"""
(4/4)

Evaluation analysis script for model predictions and reasoning quality

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json
"""

import os
import json
import re
from datetime import datetime
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)

JSON_LOG_DIR = "json_log/2_ollama_evaluation"


def get_latest_file_batch():
    files = os.listdir(JSON_LOG_DIR)
    eval_pattern = re.compile(r"^evaluation_(\d{8}_\d{6})_(\d+)\.json$")

    matches = []
    for f in files:
        m = eval_pattern.match(f)
        if m:
            timestamp = m.group(1)
            idx = int(m.group(2))
            matches.append((timestamp, idx, f))

    if not matches:
        raise FileNotFoundError(f"No evaluation_<timestamp>_<n>.json files found in {JSON_LOG_DIR}")

    latest_timestamp = max(ts for ts, _, _ in matches)

    batch_files = [
        (idx, os.path.join(JSON_LOG_DIR, fname))
        for ts, idx, fname in matches
        if ts == latest_timestamp
    ]
    batch_files.sort(key=lambda x: x[0])

    return latest_timestamp, batch_files


def load_latest_evaluations():
    batch_timestamp, batch_files = get_latest_file_batch()

    all_records = []
    per_file_records = []

    for idx, filepath in batch_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Evaluation JSON must be array of objects: {filepath}")

        all_records.extend(data)
        per_file_records.append({
            "file_index": idx,
            "file_path": filepath,
            "records": data
        })

    return batch_timestamp, per_file_records, all_records


def normalize_label(label):
    if label is None:
        return "UNKNOWN"

    label = str(label).strip()

    mapping = {
        "benign": "BENIGN",
        "normal": "BENIGN",
        "ddos": "DDoS",
        "dos hulk": "DoS Hulk",
        "dos slowhttptest": "DoS Slowhttptest",
        "dos slowloris": "DoS Slowloris",
        "dos goldeneye": "DoS GoldenEye",
        "heartbleed": "Heartbleed",
        "portscan": "PortScan",
        "bot": "Bot",
        "infiltration": "Infiltration",
        "ftp-patator": "FTP-Patator",
        "ssh-patator": "SSH-Patator",
        "web attack \u2013 brute force": "Web Attack Brute Force",
        "web attack \u2013 xss": "Web Attack XSS",
        "web attack \u2013 sql injection": "Web Attack SQL Injection",
        "web attack - brute force": "Web Attack Brute Force",
        "web attack - xss": "Web Attack XSS",
        "web attack - sql injection": "Web Attack SQL Injection",
    }

    return mapping.get(label.lower(), label)


def split_claims(text):
    if not text:
        return []
    parts = re.split(r"[.\n;]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def evaluate_reasoning_text(reasoning, predicted_label, actual_label):
    text = (reasoning or "").lower()
    pred = (predicted_label or "").lower()
    actual = (actual_label or "").lower()

    score = 0
    max_score = 4
    reasons = []

    if len(text) > 80:
        score += 1
        reasons.append("sufficient_length")

    if any(word in text for word in ["traffic", "flow", "packet", "bytes", "tcp", "flag", "duration", "attack"]):
        score += 1
        reasons.append("mentions_network_evidence")

    if pred and pred != "unknown" and pred.lower() in text:
        score += 1
        reasons.append("mentions_predicted_label")
    elif actual and actual != "unknown" and actual.lower() in text:
        score += 1
        reasons.append("mentions_actual_label")

    if any(word in text for word in ["because", "indicates", "suggests", "combined with", "therefore", "pattern"]):
        score += 1
        reasons.append("contains_explanatory_language")

    claims = split_claims(reasoning)

    return {
        "reasoning_score": score / max_score,
        "reasoning_score_raw": score,
        "reasoning_score_max": max_score,
        "claim_count": len(claims),
        "reasons": reasons,
    }

x
def evaluate_solution_text(solution, predicted_label):
    text = (solution or "").lower()
    pred = (predicted_label or "").lower()

    score = 0
    max_score = 4
    reasons = []

    if len(text) > 50:
        score += 1
        reasons.append("non_empty_solution")

    if any(word in text for word in ["limit", "block", "filter", "timeout", "firewall", "ids", "ips", "monitor"]):
        score += 1
        reasons.append("contains_actionable_terms")

    if pred == "dos slowhttptest":
        if any(word in text for word in ["header timeout", "connection timeout", "slow http", "nginx", "apache"]):
            score += 1
            reasons.append("attack_specific_mitigation")
    elif pred == "ddos":
        if any(word in text for word in ["rate limit", "scrubbing", "upstream", "blackhole", "traffic shaping"]):
            score += 1
            reasons.append("attack_specific_mitigation")
    elif pred == "portscan":
        if any(word in text for word in ["scan detection", "threshold", "block scanner", "ips"]):
            score += 1
            reasons.append("attack_specific_mitigation")
    else:
        if any(word in text for word in ["rate limit", "timeout", "acl", "signature", "suricata", "snort"]):
            score += 1
            reasons.append("generic_security_mitigation")

    if any(word in text for word in ["1.", "2.", "3.", "implement", "configure", "deploy", "use"]):
        score += 1
        reasons.append("structured_recommendations")

    return {
        "solution_score": score / max_score,
        "solution_score_raw": score,
        "solution_score_max": max_score,
        "reasons": reasons,
    }


def evaluate_labels(records):
    y_true = [normalize_label(r["actual_label"]) for r in records]
    y_pred = [normalize_label(r["predicted_label"]) for r in records]

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except Exception:
        kappa = None

    return {
        "num_samples": len(records),
        "labels": labels,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "cohen_kappa": kappa,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "correct_count": sum(1 for r in records if r.get("is_model_correct") is True),
        "incorrect_count": sum(1 for r in records if r.get("is_model_correct") is False),
    }


def evaluate_text_fields(records):
    reasoning_scores = []
    solution_scores = []
    details = []

    for idx, r in enumerate(records):
        predicted_label = normalize_label(r.get("predicted_label"))
        actual_label = normalize_label(r.get("actual_label"))
        reasoning = r.get("reasoning", "")
        solution = r.get("solution", "")

        reasoning_eval = evaluate_reasoning_text(reasoning, predicted_label, actual_label)
        solution_eval = evaluate_solution_text(solution, predicted_label)

        reasoning_scores.append(reasoning_eval["reasoning_score"])
        solution_scores.append(solution_eval["solution_score"])

        details.append({
            "index": idx,
            "predicted_label": predicted_label,
            "actual_label": actual_label,
            "is_model_correct": r.get("is_model_correct"),
            "reasoning_score": reasoning_eval["reasoning_score"],
            "reasoning_score_raw": reasoning_eval["reasoning_score_raw"],
            "reasoning_reasons": reasoning_eval["reasons"],
            "claim_count": reasoning_eval["claim_count"],
            "solution_score": solution_eval["solution_score"],
            "solution_score_raw": solution_eval["solution_score_raw"],
            "solution_reasons": solution_eval["reasons"],
            "reasoning_preview": reasoning[:250],
            "solution_preview": solution[:250],
        })

    avg_reasoning_score = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
    avg_solution_score = sum(solution_scores) / len(solution_scores) if solution_scores else 0.0

    return {
        "avg_reasoning_score": avg_reasoning_score,
        "avg_solution_score": avg_solution_score,
        "details": details,
    }


def build_per_file_summary(per_file_records):
    summaries = []

    for item in per_file_records:
        idx = item["file_index"]
        filepath = item["file_path"]
        records = item["records"]

        label_eval = evaluate_labels(records)
        text_eval = evaluate_text_fields(records)

        summaries.append({
            "file_index": idx,
            "source_file": filepath,
            "num_records": len(records),
            "accuracy": label_eval["accuracy"],
            "f1_macro": label_eval["f1_macro"],
            "f1_weighted": label_eval["f1_weighted"],
            "avg_reasoning_score": text_eval["avg_reasoning_score"],
            "avg_solution_score": text_eval["avg_solution_score"],
            "correct_count": label_eval["correct_count"],
            "incorrect_count": label_eval["incorrect_count"],
        })

    return summaries


def main():
    batch_timestamp, per_file_records, all_records = load_latest_evaluations()

    label_eval = evaluate_labels(all_records)
    text_eval = evaluate_text_fields(all_records)
    per_file_summary = build_per_file_summary(per_file_records)

    final_output = {
        "source_batch_timestamp": batch_timestamp,
        "source_files": [item["file_path"] for item in per_file_records],
        "num_source_files": len(per_file_records),
        "num_records_total": len(all_records),
        "averaged_across_files": {
            "accuracy_mean": sum(x["accuracy"] for x in per_file_summary) / len(per_file_summary) if per_file_summary else 0.0,
            "f1_macro_mean": sum(x["f1_macro"] for x in per_file_summary) / len(per_file_summary) if per_file_summary else 0.0,
            "f1_weighted_mean": sum(x["f1_weighted"] for x in per_file_summary) / len(per_file_summary) if per_file_summary else 0.0,
            "avg_reasoning_score_mean": sum(x["avg_reasoning_score"] for x in per_file_summary) / len(per_file_summary) if per_file_summary else 0.0,
            "avg_solution_score_mean": sum(x["avg_solution_score"] for x in per_file_summary) / len(per_file_summary) if per_file_summary else 0.0,
        },
        "combined_record_evaluation": {
            "num_records": len(all_records),
            "label_evaluation": label_eval,
            "text_evaluation": {
                "avg_reasoning_score": text_eval["avg_reasoning_score"],
                "avg_solution_score": text_eval["avg_solution_score"],
            },
            "label_distribution_actual": dict(Counter(normalize_label(r["actual_label"]) for r in all_records)),
            "label_distribution_predicted": dict(Counter(normalize_label(r["predicted_label"]) for r in all_records)),
        },
        "per_file_summary": per_file_summary,
        "sample_details": text_eval["details"][:20],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = os.path.normpath(os.path.join(JSON_LOG_DIR, "..", "3_analysis_of_evaluation"))
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"evaluation_metrics_{batch_timestamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(json.dumps(final_output, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
