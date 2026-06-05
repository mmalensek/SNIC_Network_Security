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
    / "4_score_combined"
)

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


def latest_json(directory, max_age_minutes=30):
    files = sorted(directory.glob("*.json"))
    
    if not files:
        return None
    
    # Get the latest file
    latest_file = files[-1]
    
    # Check if it's within the time window
    file_mtime = datetime.fromtimestamp(latest_file.stat().st_mtime)
    age = datetime.now() - file_mtime
    
    if age.total_seconds() > max_age_minutes * 60:
        return None  # File is too old
    
    return latest_file



# --------------------------------------------------
# Load latest files
# --------------------------------------------------

deterministic_file = latest_json(
    DETERMINISTIC_DIR,
    max_age_minutes=30
)

expert_file = latest_json(
    EXPERT_DIR,
    max_age_minutes=30
)

human_file = latest_json(
    HUMAN_DIR,
    max_age_minutes=30
)

if deterministic_file is None:
    raise RuntimeError(
        "No deterministic score file found"
    )

if expert_file is None:
    raise RuntimeError(
        "No expert system score file found"
    )

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

    human_points = defaultdict(float)
    human_matches = defaultdict(int)

    for comparison in human.get(
        "comparisons",
        []
    ):

        a = comparison["candidate_a_model"]
        b = comparison["candidate_b_model"]
        winner = comparison["winner"]

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
# Ranking
# --------------------------------------------------

ranking = sorted(
    [
        {
            "model": model,
            "score": round(
                data["final_combined_score"],
                1
            )
        }
        for model, data in combined.items()
        if data["final_combined_score"] is not None
    ],
    key=lambda x: x["score"],
    reverse=True
)

for rank, entry in enumerate(
    ranking,
    start=1
):
    entry["rank"] = rank


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

    "source_files": {
        "deterministic":
            str(deterministic_file),

        "expert":
            str(expert_file),

        "human":
            str(human_file) if human_file is not None else None,
    },

    "ranking":
        ranking,

    "results":
        combined,
}


timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = (
    OUTPUT_DIR
    / f"combined_scores_{timestamp}.json"
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

print(
    f"\nCombined report saved to:\n"
    f"{output_file}"
)

print(
    f"Human score included: {human_file is not None}"
)

print(
    f"Weights used: {ACTIVE_WEIGHTS}"
)
