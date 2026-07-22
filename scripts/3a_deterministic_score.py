#!/usr/bin/env python3

"""
(3a/4)

Deterministic scoring script for evaluating model performance

Prerequisites:
openai >= 1.0.0
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json
"""

import re
import json
from pathlib import Path
from statistics import mean
from datetime import datetime
from collections import defaultdict

ROOT = Path("json_log")

GT_DIR = ROOT / "1_groundtruth_and_xgboost_prediction"

EVALUATION_DIRS = {
    "ollama": ROOT / "2_ollama_evaluation",
    "openai": ROOT / "2_openai_evaluation",
    "retrained": ROOT / "2_retrained_evaluation",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_label(label):
    return str(label).strip().lower()

def score_attack_alignment(entry, ground_truth):
    predicted = normalize_label(
        entry.get("xgboost_predicted_label", "")
    )

    actual = normalize_label(
        ground_truth.get("most_common_true_label", "")
    )

    return 1.0 if predicted == actual else 0.0

def extract_timestamp(filename):
    """
    Only used to pick which batch is 'latest' - not used for pairing.
    evaluation_20260506_085644_1.json -> "20260506_085644"
    """
    m = re.match(r".*?_(\d{8}_\d{6})_\d+\.json$", filename)
    return m.group(1) if m else None


def get_latest_timestamp(directory):
    timestamps = []

    for f in directory.glob("evaluation_*.json"):
        ts = extract_timestamp(f.name)
        if ts:
            timestamps.append(ts)

    if not timestamps:
        return None

    return max(timestamps)


# ----------------------------------------------------------------------
# Format Compliance
# ----------------------------------------------------------------------

REQUIRED_FIELDS = {
    "model",
    "xgboost_predicted_label",
    "actual_label",
    "is_xgboost_correct",
    "reasoning",
    "solution",
}


def score_format_compliance(entry):
    score = 0
    total = len(REQUIRED_FIELDS)

    for field in REQUIRED_FIELDS:
        if field in entry:
            score += 1

    score = score / total

    if len(str(entry.get("reasoning", ""))) < 50:
        score *= 0.9

    if len(str(entry.get("solution", ""))) < 50:
        score *= 0.9

    return score


# ----------------------------------------------------------------------
# Attack Alignment
# ----------------------------------------------------------------------

def score_attack_alignment(entry, ground_truth):
    predicted = normalize_label(
        entry.get("predicted_label", "")
    )

    actual = normalize_label(
        ground_truth.get(
            "most_common_true_label",
            ""
        )
    )

    return 1.0 if predicted == actual else 0.0


# ----------------------------------------------------------------------
# Feature Grounding
# ----------------------------------------------------------------------

FEATURE_KEYWORDS = {
    "flow duration": [
        "flow duration"
    ],
    "idle": [
        "idle"
    ],
    "ack": [
        "ack"
    ],
    "syn": [
        "syn"
    ],
    "rst": [
        "rst"
    ],
    "packet length std": [
        "packet length std",
        "packet size variation",
        "packet length variation"
    ],
    "packet length": [
        "packet length"
    ],
    "flow packets": [
        "flow packets"
    ],
    "flow bytes": [
        "flow bytes"
    ],
}

def get_feature(row_data, name, default=0):
    """
    Handle CICIDS column names with leading spaces.
    """
    for k, v in row_data.items():
        if k.strip().lower() == name.lower():
            return v

    return default


def detect_claims(reasoning):
    """
    Extract measurable claims from reasoning.
    """

    text = reasoning.lower()

    claims = []

    patterns = {
        "long_flow_duration": [
            r"long flow duration",
            r"extremely long flow duration",
            r"high flow duration",
            r"unusually long",
        ],

        "high_idle": [
            r"high idle",
            r"long idle",
            r"idle times",
            r"high idle times",
        ],

        "high_packet_rate": [
            r"high packet rate",
            r"high packet rates",
            r"high packets/s",
            r"many packets",
            r"flood",
            r"flooding",
        ],

        "low_packet_rate": [
            r"low packet rate",
            r"low packet rates",
            r"low-rate",
            r"low rate",
        ],

        "ack_activity": [
            r"ack flags",
            r"ack traffic",
            r"dominance of ack",
            r"ack packets",
        ],

        "syn_activity": [
            r"syn flags",
            r"syn traffic",
            r"syn packets",
        ],

        "rst_activity": [
            r"rst flags",
            r"rst traffic",
            r"rst packets",
        ],

        "high_packet_length_std": [
            r"packet length standard deviation",
            r"high packet length std",
            r"variable packet sizes",
            r"highly variable packet sizes",
        ],

        "large_packets": [
            r"large packets",
            r"large packet sizes",
            r"high packet size",
        ],
    }

    for claim, regexes in patterns.items():
        for pattern in regexes:
            if re.search(pattern, text):
                claims.append(claim)
                break

    return claims


def verify_claim(claim, row_data):

    flow_duration = get_feature(
        row_data,
        "Flow Duration"
    )

    idle_mean = get_feature(
        row_data,
        "Idle Mean"
    )

    idle_max = get_feature(
        row_data,
        "Idle Max"
    )

    flow_packets_s = get_feature(
        row_data,
        "Flow Packets/s"
    )

    ack_count = get_feature(
        row_data,
        "ACK Flag Count"
    )

    syn_count = get_feature(
        row_data,
        "SYN Flag Count"
    )

    rst_count = get_feature(
        row_data,
        "RST Flag Count"
    )

    packet_std = get_feature(
        row_data,
        "Packet Length Std"
    )

    avg_packet_size = get_feature(
        row_data,
        "Average Packet Size"
    )

    if claim == "long_flow_duration":
        return flow_duration > 100000

    elif claim == "high_idle":
        return max(idle_mean, idle_max) > 10000

    elif claim == "high_packet_rate":
        return flow_packets_s > 1000

    elif claim == "low_packet_rate":
        return flow_packets_s < 100

    elif claim == "ack_activity":
        return ack_count > 0

    elif claim == "syn_activity":
        return syn_count > 0

    elif claim == "rst_activity":
        return rst_count > 0

    elif claim == "high_packet_length_std":
        return packet_std > 500

    elif claim == "large_packets":
        return avg_packet_size > 1000

    return None


def score_feature_grounding(entry, prediction):

    row_data = prediction.get(
        "current_flow",
        {}
    )

    if not row_data:
        return None

    reasoning = entry.get(
        "reasoning",
        ""
    )

    claims = detect_claims(reasoning)

    if len(claims) == 0:
        return 0.0

    verified = 0

    for claim in claims:

        result = verify_claim(
            claim,
            row_data
        )

        if result is True:
            verified += 1

    return verified / len(claims)


# ----------------------------------------------------------------------
# Overall
# ----------------------------------------------------------------------

def compute_overall_score(
    format_score,
    attack_score,
    grounding_score
):
    if grounding_score is None:
        grounding_score = 0

    return (
        0.20 * format_score +
        0.50 * attack_score +
        0.30 * grounding_score
    )


# ----------------------------------------------------------------------
# Matching files
# ----------------------------------------------------------------------

def find_ground_truth_file(run_id, sample_id):
    for path in GT_DIR.glob("ground_truth_*.json"):
        try:
            data = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("run_id") == run_id and data.get("sample_id") == sample_id:
            return path
    return None


def find_prediction_file(run_id, sample_id):
    for path in GT_DIR.glob("prediction_*.json"):
        try:
            data = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("run_id") == run_id and data.get("sample_id") == sample_id:
            return path
    return None


# ----------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------

def evaluate_directory(name, directory):

    latest_timestamp = get_latest_timestamp(directory)

    if latest_timestamp is None:
        return None

    evaluation_files = sorted(
        directory.glob(f"evaluation_{latest_timestamp}_*.json")
    )

    model_results = defaultdict(list)

    for evaluation_file in evaluation_files:

        # Load evaluations FIRST so we can read run_id/sample_id from them
        evaluations = load_json(evaluation_file)

        if not isinstance(evaluations, list):
            evaluations = [evaluations]

        if not evaluations:
            continue

        run_id = evaluations[0].get("run_id")
        sample_id = evaluations[0].get("sample_id")

        gt_file = find_ground_truth_file(run_id, sample_id)
        prediction_file = find_prediction_file(run_id, sample_id)

        if gt_file is None:
            continue

        ground_truth = load_json(gt_file)

        prediction = {}
        if prediction_file:
            prediction = load_json(prediction_file)

        for entry in evaluations:
            model_name = entry.get("model", "unknown")

            format_score = score_format_compliance(entry)
            attack_score = score_attack_alignment(entry, ground_truth)
            grounding_score = score_feature_grounding(entry, prediction)
            overall = compute_overall_score(format_score, attack_score, grounding_score)

            model_results[model_name].append({
                "format_compliance": format_score,
                "attack_alignment": attack_score,
                "feature_grounding": grounding_score,
                "overall": overall,
            })

    aggregated = {}

    for model, scores in model_results.items():
        aggregated[model] = {
            "num_samples": len(scores),
            "format_compliance": mean(s["format_compliance"] for s in scores),
            "attack_alignment": mean(s["attack_alignment"] for s in scores),
            "feature_grounding": mean(
                s["feature_grounding"] for s in scores
                if s["feature_grounding"] is not None
            ),
            "overall": mean(s["overall"] for s in scores),
        }

    return {
        "latest_batch": latest_timestamp,
        "models": aggregated,
    }


def main():

    report = {}

    for name, directory in EVALUATION_DIRS.items():

        if not directory.exists():
            report[name] = None
            continue

        report[name] = evaluate_directory(
            name,
            directory
        )

    print(
        json.dumps(
            report,
            indent=2
        )
    )

    OUTPUT_DIR = (
        ROOT
        / "3_evaluation_results"
        / "1_deterministic_score"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        OUTPUT_DIR
        / f"deterministic_score_{timestamp}.json"
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
        f"Deterministic score report saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()