"""
(2a/4)

Ollama explanation script for evaluating model reasoning and solution quality

* this script is similar to oll_test.py but focuses on extracting and saving the reasoning 
and solution from the model response, without asking for a label prediction from the LLM

** the evaluation will still compare the XGBoost predicted label to the true label, 
but the LLM's response will be evaluated qualitatively based on the reasoning and solution sections

Prerequisites:
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json
"""

import os
import re
import json
import requests
import argparse
import subprocess
from datetime import datetime

OLLAMA_API = "http://localhost:11434/api/generate"
JSON_LOG_DIR = "json_log/1_groundtruth_and_xgboost_prediction"
EVAL_LOG_DIR = "json_log/2_ollama_evaluation"

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Ollama explanation models."
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model index or 'all'. Example: 0, 1 or all"
    )

    return parser.parse_args()

# getting local Ollama models
def get_ollama_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")[1:]
    models = [line.split()[0] for line in lines if line]
    return models


# get latest batch of numbered prediction/ground truth files
def get_latest_file_pairs():
    files = os.listdir(JSON_LOG_DIR)

    pred_pattern = re.compile(r"^prediction_(\d{8}_\d{6})_(\d+)\.json$")
    gt_pattern = re.compile(r"^ground_truth_(\d{8}_\d{6})_(\d+)\.json$")

    pred_matches = []
    gt_matches = []

    for f in files:
        pred_match = pred_pattern.match(f)
        if pred_match:
            pred_matches.append((pred_match.group(1), int(pred_match.group(2)), f))

        gt_match = gt_pattern.match(f)
        if gt_match:
            gt_matches.append((gt_match.group(1), int(gt_match.group(2)), f))

    if not pred_matches or not gt_matches:
        raise FileNotFoundError("No numbered prediction/ground truth files found.")

    latest_timestamp = max(ts for ts, _, _ in pred_matches)

    pred_map = {
        idx: os.path.join(JSON_LOG_DIR, fname)
        for ts, idx, fname in pred_matches
        if ts == latest_timestamp
    }

    gt_map = {
        idx: os.path.join(JSON_LOG_DIR, fname)
        for ts, idx, fname in gt_matches
        if ts == latest_timestamp
    }

    common_indices = sorted(set(pred_map.keys()) & set(gt_map.keys()))

    if not common_indices:
        raise FileNotFoundError("No matching prediction/ground truth pairs found for latest timestamp.")

    return latest_timestamp, [(idx, pred_map[idx], gt_map[idx]) for idx in common_indices]


# query Ollama model
def query_model(model, prompt):
    response = requests.post(
        OLLAMA_API,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# extract reasoning and solution from model response (no label expected from LLM)
def extract_response_parts(text):
    reasoning = ""
    solution = ""

    if "REASONING:" in text:
        reasoning_start = text.index("REASONING:") + len("REASONING:")
        reasoning_end = text.find("SOLUTION:") if "SOLUTION:" in text else len(text)
        reasoning = text[reasoning_start:reasoning_end].strip()

    if "SOLUTION:" in text:
        solution_start = text.index("SOLUTION:") + len("SOLUTION:")
        solution = text[solution_start:].strip()

    return {
        "reasoning": reasoning,
        "solution": solution
    }


# build prompt for model
def build_prompt(pred_json):
    return f"""
You are a cybersecurity expert.

Analyze the following network traffic summary produced by an XGBoost classifier.

Provide your response EXACTLY and ONLY in the following format:

REASONING:
[Analyze what is happening in the traffic, whether it is an attack, and what type of attack if applicable]

SOLUTION:
[Provide specific recommendations or mitigation strategies for this traffic pattern]

JSON:
{json.dumps(pred_json, indent=2)}
"""


# run evaluation for one prediction/ground truth pair
def evaluate(models, pred_json, ground_truth):
    results = []

    true_label = ground_truth["most_common_true_label"]
    # XGBoost predicted label comes from the prediction JSON, not the LLM
    xgboost_label = pred_json.get("model_prediction", "UNKNOWN")
    xgboost_correct = xgboost_label == true_label

    for model in models:
        print(f"\n--- Testing model: {model} ---")

        prompt = build_prompt(pred_json)
        response = query_model(model, prompt)

        response_parts = extract_response_parts(response)
        reasoning = response_parts["reasoning"]
        solution = response_parts["solution"]

        results.append({
            "model": model,
            "xgboost_predicted_label": xgboost_label,
            "actual_label": true_label,
            "is_xgboost_correct": xgboost_correct,
            "reasoning": reasoning,
            "solution": solution
        })

        print(f"XGBoost Predicted: {xgboost_label}")
        print(f"True Label:        {true_label}")
        print(f"XGBoost Correct:   {xgboost_correct}")
        print("\n--- Reasoning ---")
        print(reasoning)
        print("\n--- Solution ---")
        print(solution)

    return results


# main
def main():

    args = parse_args()

    print("Starting Ollama evaluation script...")
    models = get_ollama_models()
    print(f"Found {len(models)} local Ollama models.")

    print("\nAvailable models:")
    for i, m in enumerate(models):
        print(f"{i}: {m}")

    if args.model:
     choice = args.model
    else:
      choice = input("\nSelect model index or 'all': ")

    if choice == "all":
        selected_models = models
    else:
        selected_models = [models[int(choice)]]

    latest_timestamp, file_pairs = get_latest_file_pairs()

    print(f"\nFound {len(file_pairs)} prediction/ground truth pairs for batch {latest_timestamp}")

    os.makedirs(EVAL_LOG_DIR, exist_ok=True)

    overall_correct = 0
    overall_total = 0

    for idx, pred_file, gt_file in file_pairs:
        print(f"\n==============================")
        print(f"Processing pair {idx}")
        print(f"Prediction file: {pred_file}")
        print(f"Ground truth file: {gt_file}")
        print(f"==============================\n")

        with open(pred_file) as f:
            pred_json = json.load(f)

        with open(gt_file) as f:
            ground_truth = json.load(f)

        results = evaluate(selected_models, pred_json, ground_truth)

        correct = sum(r["is_xgboost_correct"] for r in results)
        total = len(results)

        overall_correct += correct
        overall_total += total

        print("\n--- Pair Summary ---")
        print(f"XGBoost accuracy for pair {idx}: {correct}/{total} = {correct/total:.2f}")

        out_file = os.path.join(EVAL_LOG_DIR, f"evaluation_{latest_timestamp}_{idx}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {out_file}")

    print("\n=== Overall Summary ===")
    print(f"XGBoost Accuracy: {overall_correct}/{overall_total} = {overall_correct/overall_total:.2f}")


if __name__ == "__main__":
    main()
