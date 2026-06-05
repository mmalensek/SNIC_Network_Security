#!/usr/bin/env python3

"""
(3b/4)

Expert System Score Evaluation, with GPT-5 as the judge.

Prerequisites:
openai >= 1.0.0
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean
from datetime import datetime

from openai import OpenAI

ROOT = Path("json_log")

GT_DIR = ROOT / "1_groundtruth_and_xgboost_prediction"

EVALUATION_DIRS = {
    "ollama": ROOT / "2_ollama_evaluation",
    "openai": ROOT / "2_openai_evaluation",
    "retrained": ROOT / "2_retrained_evaluation",
}

OUTPUT_DIR = (
    ROOT
    / "3_evaluation_results"
    / "2_expert_system_score"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    ""
).strip()

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set"
    )

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_timestamp_and_n(filename):

    m = re.match(
        r".*?_(\d{8}_\d{6})_(\d+)\.json$",
        filename
    )

    if not m:
        return None, None

    return m.group(1), int(m.group(2))


def get_latest_timestamp(directory):

    timestamps = []

    for f in directory.glob(
        "evaluation_*.json"
    ):
        ts, _ = extract_timestamp_and_n(
            f.name
        )

        if ts:
            timestamps.append(ts)

    if not timestamps:
        return None

    return max(timestamps)


def find_ground_truth_file(n):

    matches = list(
        GT_DIR.glob(
            f"ground_truth_*_{n}.json"
        )
    )

    return matches[0] if matches else None


def find_prediction_file(n):

    matches = list(
        GT_DIR.glob(
            f"*prediction*_{n}.json"
        )
    )

    return matches[0] if matches else None


# --------------------------------------------------
# Judge Prompt
# --------------------------------------------------

def judge_evaluation(
    evaluation,
    ground_truth,
    prediction
):

    prompt = f"""
You are an expert cybersecurity analyst.

Evaluate the quality of the AI-generated analysis.

GROUND TRUTH:
{json.dumps(ground_truth, indent=2)}

XGBOOST PREDICTION:
{json.dumps(prediction, indent=2)}

AI EVALUATION:
{json.dumps(evaluation, indent=2)}

Score the following categories from 0 to 10:

1. attack_reasoning
2. feature_grounding
3. technical_accuracy
4. actionability
5. overall

Return ONLY valid JSON:

{{
  "attack_reasoning": int,
  "feature_grounding": int,
  "technical_accuracy": int,
  "actionability": int,
  "overall": int,
  "reasoning": "short explanation"
}}
"""

    response = client.chat.completions.create(
        model="gpt-5",
        temperature=1,
        messages=[
            {
                "role": "system",
                "content":
                    "You are a strict cybersecurity evaluator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    try:
        return json.loads(text)

    except Exception:
        return {
            "attack_reasoning": 0,
            "feature_grounding": 0,
            "technical_accuracy": 0,
            "actionability": 0,
            "overall": 0,
            "reasoning":
                "Judge returned invalid JSON"
        }


# --------------------------------------------------
# Evaluate directory
# --------------------------------------------------

def evaluate_directory(directory):

    latest_timestamp = (
        get_latest_timestamp(directory)
    )

    if latest_timestamp is None:
        return None

    files = sorted(
        directory.glob(
            f"evaluation_{latest_timestamp}_*.json"
        )
    )

    aggregate = defaultdict(list)

    sample_details = []

    for evaluation_file in files:

        _, n = extract_timestamp_and_n(
            evaluation_file.name
        )

        gt_file = find_ground_truth_file(n)

        pred_file = find_prediction_file(n)

        if not gt_file:
            continue

        ground_truth = load_json(
            gt_file
        )

        prediction = (
            load_json(pred_file)
            if pred_file
            else {}
        )

        evaluations = load_json(
            evaluation_file
        )

        if not isinstance(
            evaluations,
            list
        ):
            evaluations = [evaluations]

        for evaluation in evaluations:

            judge_result = judge_evaluation(
                evaluation,
                ground_truth,
                prediction
            )

            model_name = evaluation.get(
                "model",
                "unknown"
            )

            aggregate[
                model_name
            ].append(judge_result)

            sample_details.append({
                "sample_id": n,
                "model": model_name,
                "judge_result": judge_result
            })

    summary = {}

    for model, scores in aggregate.items():

        summary[model] = {
            "num_samples":
                len(scores),

            "attack_reasoning":
                mean(
                    s["attack_reasoning"]
                    for s in scores
                ),

            "feature_grounding":
                mean(
                    s["feature_grounding"]
                    for s in scores
                ),

            "technical_accuracy":
                mean(
                    s["technical_accuracy"]
                    for s in scores
                ),

            "actionability":
                mean(
                    s["actionability"]
                    for s in scores
                ),

            "overall":
                mean(
                    s["overall"]
                    for s in scores
                )
        }

    return {
        "latest_batch":
            latest_timestamp,

        "summary":
            summary,

        "samples":
            sample_details
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    report = {
        "generated_at":
            datetime.now().isoformat(),

        "judge_model":
            "gpt-5",

        "results": {}
    }

    for name, directory in (
        EVALUATION_DIRS.items()
    ):

        if not directory.exists():

            report["results"][
                name
            ] = None

            continue

        report["results"][
            name
        ] = evaluate_directory(
            directory
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        OUTPUT_DIR
        / f"expert_system_score_{timestamp}.json"
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
        f"Saved results to {output_file}"
    )


if __name__ == "__main__":
    main()