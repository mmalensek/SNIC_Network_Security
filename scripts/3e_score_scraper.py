#!/usr/bin/env python3

"""
(3e/4)

Score scraping script for extracting and processing evaluation scores for retraining of models

Prerequisites:
openai >= 1.0.0
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from datetime import datetime, timedelta

ROOT = Path("json_log")

DETERMINISTIC_DIR = (
    ROOT
    / "3_evaluation_results"
    / "1_deterministic_score"
)

EXPERT_DIR = (
    ROOT
    / "3_evaluation_results"
    / "2_expert_system_score"
)

HUMAN_DIR = (
    ROOT
    / "3_evaluation_results"
    / "3_human_expert_score"
)

OUTPUT_DIR = (
    ROOT
    / "3_evaluation_results"
    / "6_score_winner"
)

HISTORY_OUTPUT_DIR = (
    ROOT
    / "3_evaluation_results"
    / "5b_history_log"
)

OLLAMA_DIR = ROOT / "2_ollama_evaluation"
OPENAI_DIR = ROOT / "2_openai_evaluation"
RETRAINED_DIR = ROOT / "2_retrained_evaluation"

HISTORY_FILE = HISTORY_OUTPUT_DIR / "training_history.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WEIGHTS_WITH_HUMAN = {
    "deterministic": 0.25,
    "expert": 0.25,
    "human": 0.50,
}

WEIGHTS_NO_HUMAN = {
    "deterministic": 0.40,
    "expert": 0.60,
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_json(directory):
    files = sorted(directory.glob("*.json"))

    if not files:
        return None

    return files[-1]

def find_model_output(model_name):
    search_dirs = [
        OLLAMA_DIR,
        OPENAI_DIR,
        RETRAINED_DIR,
    ]

    for directory in search_dirs:

        if not directory.exists():
            continue

        for file in sorted(
            directory.glob("*.json")
        ):

            try:
                data = load_json(file)

                if not isinstance(data, list):
                    continue

                for entry in data:

                    if (
                        isinstance(entry, dict)
                        and entry.get("model") == model_name
                    ):

                        return {
                            "source_file": str(file),
                            "evaluation_record": entry,
                        }

            except Exception as e:
                print(
                    f"Warning reading {file}: {e}"
                )

    return None

# --------------------------------------------------
# Load latest files
# --------------------------------------------------

deterministic_file = latest_json(
    DETERMINISTIC_DIR
)

expert_file = latest_json(
    EXPERT_DIR
)

human_file = latest_json(
    HUMAN_DIR
)

# --------------------------------------------------
# Required files
# --------------------------------------------------

if deterministic_file is None:
    raise RuntimeError(
        "No deterministic score file found"
    )

if expert_file is None:
    raise RuntimeError(
        "No expert system score file found"
    )

# --------------------------------------------------
# Check timestamps
# --------------------------------------------------

det_time = datetime.fromtimestamp(
    deterministic_file.stat().st_mtime
)

exp_time = datetime.fromtimestamp(
    expert_file.stat().st_mtime
)

# Deterministic and expert MUST belong
# to the same run

if abs(
    (det_time - exp_time).total_seconds()
) > 30 * 60:
    raise RuntimeError(
        "Deterministic and expert files "
        "differ by more than 30 minutes"
    )

# Human file is optional
if human_file is not None:

    human_time = datetime.fromtimestamp(
        human_file.stat().st_mtime
    )

    newest_core = max(
        det_time,
        exp_time
    )

    oldest_core = min(
        det_time,
        exp_time
    )

    if (
        abs(
            (
                human_time
                - newest_core
            ).total_seconds()
        ) > 30 * 60
        or
        abs(
            (
                human_time
                - oldest_core
            ).total_seconds()
        ) > 30 * 60
    ):
        print(
            "Human evaluation file is "
            "outside the 30 minute window. "
            "Ignoring it."
        )

        human_file = None

# --------------------------------------------------
# Load JSON data
# --------------------------------------------------

deterministic = load_json(
    deterministic_file
)

expert = load_json(
    expert_file
)

human = None
human_scores = {}

if human_file is not None:
    human = load_json(
        human_file
    )

# --------------------------------------------------
# Human comparison -> normalized score
# --------------------------------------------------

if human is not None:

    human_points = defaultdict(float)
    human_matches = defaultdict(int)

    for comparison in human.get(
        "comparisons",
        []
    ):

        a = comparison[
            "candidate_a_model"
        ]

        b = comparison[
            "candidate_b_model"
        ]

        winner = comparison[
            "winner"
        ]

        human_matches[a] += 1
        human_matches[b] += 1

        if winner == "A":
            human_points[a] += 1

        elif winner == "B":
            human_points[b] += 1

        else:  # TIE
            human_points[a] += 0.5
            human_points[b] += 0.5

    for model in human_matches:

        human_scores[model] = (
            human_points[model]
            / human_matches[model]
        ) * 100

ACTIVE_WEIGHTS = (
    WEIGHTS_WITH_HUMAN
    if human_file is not None
    else WEIGHTS_NO_HUMAN
)


# --------------------------------------------------
# Collect all models
# --------------------------------------------------

all_models = set()

# deterministic models
for system_data in deterministic.values():

    if not isinstance(
        system_data,
        dict
    ):
        continue

    models = system_data.get(
        "models",
        {}
    )

    all_models.update(
        models.keys()
    )

# expert models
for system_data in (
    expert.get(
        "results",
        {}
    ).values()
):

    if not isinstance(
        system_data,
        dict
    ):
        continue

    summary = system_data.get(
        "summary",
        {}
    )

    all_models.update(
        summary.keys()
    )

# human models
all_models.update(
    human_scores.keys()
)


# --------------------------------------------------
# Score lookups
# --------------------------------------------------

def find_deterministic_score(model):

    for system_data in deterministic.values():

        if not isinstance(
            system_data,
            dict
        ):
            continue

        models = system_data.get(
            "models",
            {}
        )

        if model in models:
            return (
                models[model]
                .get("overall", 0)
                * 100
            )

    return None


def find_expert_score(model):

    for system_data in (
        expert.get(
            "results",
            {}
        ).values()
    ):

        if not isinstance(
            system_data,
            dict
        ):
            continue

        summary = system_data.get(
            "summary",
            {}
        )

        if model in summary:
            return (
                summary[model]
                .get("overall", 0)
                * 10
            )

    return None


def find_human_score(model):
    return human_scores.get(model)


# --------------------------------------------------
# Combine scores
# --------------------------------------------------

combined = {}

for model in sorted(all_models):

    d = find_deterministic_score(
        model
    )

    e = find_expert_score(
        model
    )

    h = find_human_score(
        model
    )

    weighted_sum = 0
    weight_total = 0

    if d is not None:
        weighted_sum += (
            d
            * ACTIVE_WEIGHTS["deterministic"]
        )
        weight_total += (
            ACTIVE_WEIGHTS["deterministic"]
        )

    if e is not None:
        weighted_sum += (
            e
            * ACTIVE_WEIGHTS["expert"]
        )
        weight_total += (
            ACTIVE_WEIGHTS["expert"]
        )

    if human_file is not None and h is not None:
        weighted_sum += (
            h
            * ACTIVE_WEIGHTS["human"]
        )
        weight_total += (
            ACTIVE_WEIGHTS["human"]
        )

    final_score = (
        weighted_sum / weight_total
        if weight_total > 0
        else None
    )

    combined[model] = {
        "deterministic_score":
            round(d, 1) if d is not None else None,

        "expert_system_score":
            round(e, 1) if e is not None else None,

        "human_expert_score":
            round(h, 1) if h is not None else None,

        "final_combined_score":
            round(final_score, 1)
            if final_score is not None
            else None,
    }

# --------------------------------------------------
# Find winner
# --------------------------------------------------

winner_model = None
winner_score = -1

for model, scores in combined.items():

    final_score = scores.get(
        "final_combined_score"
    )

    if (
        final_score is not None
        and final_score > winner_score
    ):
        winner_model = model
        winner_score = final_score

if winner_model is None:
    raise RuntimeError(
        "No winning model found"
    )

winner_output = find_model_output(
    winner_model
)


# --------------------------------------------------
# Final report
# --------------------------------------------------

report = {
    "generated_at":
        datetime.now().isoformat(),

    "weights":
        ACTIVE_WEIGHTS,

    "human_score_included":
        human_file is not None,

    "winner": {
        "model":
            winner_model,

        "final_score":
            round(winner_score, 1),

        "scores":
            combined[winner_model],
    },

    "model_output":
        winner_output,
}


timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = (
    OUTPUT_DIR
    / f"winner_{timestamp}.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=2
    )

# --------------------------------------------------
# Save training history
# --------------------------------------------------

history_file = OUTPUT_DIR / "training_history.csv"

# score retrained modela
retrained_score = None
for model_name, scores in combined.items():
    if "retrained" in model_name.lower():
        retrained_score = scores["final_combined_score"]
        break

write_header = not history_file.exists()

# avtomatska številka iteracije
if write_header:
    iteration = 1
else:
    with open(history_file, "r", encoding="utf-8") as f:
        # odštej glavo
        iteration = max(1, sum(1 for _ in f))

with open(history_file, "a", encoding="utf-8") as f:

    if write_header:
        f.write(
            "iteration,"
            "timestamp,"
            "winner_model,"
            "winner_score,"
            "retrained_score\n"
        )

    f.write(
        f"{iteration},"
        f"{datetime.now().isoformat()},"
        f"{winner_model},"
        f"{winner_score:.1f},"
        f"{retrained_score if retrained_score is not None else ''}\n"
    )

print()
print("=" * 60)
print("WINNER")
print("=" * 60)

print(
    f"Model : {winner_model}"
)

print(
    f"Score : {winner_score:.1f}"
)

if winner_output:

    print(
        "Output found in:"
    )

    print(
        winner_output["source_file"]
    )

else:

    print(
        "No model output found."
    )

print()

print(
    f"Winner report saved to:\n"
    f"{output_file}"
)