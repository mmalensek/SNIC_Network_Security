"""
(2b/4)

OpenAI explanation script for evaluating model reasoning and solution quality

Prerequisites:
openai >= 1.0.0
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json
"""

import os
import re
import json
import argparse
from openai import OpenAI

# configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

AVAILABLE_MODELS = [
    "gpt-5.2",
    "gpt-5.1",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]

JSON_LOG_DIR = "json_log/1_groundtruth_and_xgboost_prediction"
EVAL_LOG_DIR = "json_log/2_openai_evaluation"

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

def get_openai_client():
    api_key = (OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise ValueError(
            "OpenAI API key not set. "
            "Set the OPENAI_API_KEY environment variable."
        )
    return OpenAI(api_key=api_key)


def get_latest_file_pairs():
    files = os.listdir(JSON_LOG_DIR)

    pred_pattern = re.compile(r"^prediction_(\d{8}_\d{6})_(\d+)\.json$")
    gt_pattern   = re.compile(r"^ground_truth_(\d{8}_\d{6})_(\d+)\.json$")

    pred_matches, gt_matches = [], []

    for f in files:
        pm = pred_pattern.match(f)
        gm = gt_pattern.match(f)
        if pm:
            pred_matches.append((pm.group(1), int(pm.group(2)), f))
        if gm:
            gt_matches.append((gm.group(1), int(gm.group(2)), f))

    if not pred_matches or not gt_matches:
        raise FileNotFoundError("No numbered prediction/ground truth files found.")

    latest_timestamp = max(ts for ts, _, _ in pred_matches)

    pred_map = {idx: os.path.join(JSON_LOG_DIR, fname)
                for ts, idx, fname in pred_matches if ts == latest_timestamp}
    gt_map   = {idx: os.path.join(JSON_LOG_DIR, fname)
                for ts, idx, fname in gt_matches  if ts == latest_timestamp}

    common_indices = sorted(set(pred_map.keys()) & set(gt_map.keys()))

    if not common_indices:
        raise FileNotFoundError("No matching pairs found for latest timestamp.")

    return latest_timestamp, [(idx, pred_map[idx], gt_map[idx]) for idx in common_indices]


# use openai.chat.completions instead of requests.post to Ollama
def query_model(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity expert specializing in "
                    "network traffic analysis and intrusion detection."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def extract_response_parts(text):
    reasoning, solution = "", ""

    if "REASONING:" in text:
        start = text.index("REASONING:") + len("REASONING:")
        end   = text.find("SOLUTION:") if "SOLUTION:" in text else len(text)
        reasoning = text[start:end].strip()

    if "SOLUTION:" in text:
        start    = text.index("SOLUTION:") + len("SOLUTION:")
        solution = text[start:].strip()

    return {"reasoning": reasoning, "solution": solution}


def build_prompt(pred_json):
    return f"""
Analyze the following network traffic summary produced by an XGBoost classifier.

Provide your response EXACTLY and ONLY in the following format:

REASONING:
[Analyze what is happening in the traffic, whether it is an attack, and what type of attack if applicable]

SOLUTION:
[Provide specific recommendations or mitigation strategies for this traffic pattern]

JSON:
{json.dumps(pred_json, indent=2)}
"""


def evaluate(client, models, pred_json, ground_truth):
    results = []

    true_label     = ground_truth["most_common_true_label"]
    xgboost_label  = pred_json.get("model_prediction", "UNKNOWN")
    xgboost_correct = xgboost_label == true_label

    run_id = pred_json.get("run_id")
    sample_id = pred_json.get("sample_id")

    for model in models:
        print(f"\n--- Testing model: {model} ---")

        prompt   = build_prompt(pred_json)
        response = query_model(client, model, prompt)

        parts     = extract_response_parts(response)
        reasoning = parts["reasoning"]
        solution  = parts["solution"]

        results.append({
            "run_id": run_id,              
            "sample_id": sample_id, 
            "model": model,
            "xgboost_predicted_label": xgboost_label,
            "actual_label":            true_label,
            "is_xgboost_correct":      xgboost_correct,
            "reasoning":               reasoning,
            "solution":                solution,
        })

        print(f"XGBoost Predicted: {xgboost_label}")
        print(f"True Label:        {true_label}")
        print(f"XGBoost Correct:   {xgboost_correct}")
        print("\n--- Reasoning ---")
        print(reasoning)
        print("\n--- Solution ---")
        print(solution)

    return results


def main():
    print("Starting OpenAI evaluation script...")

    client = get_openai_client()

    args = parse_args()

    print("\nAvailable OpenAI models:")
    for i, m in enumerate(AVAILABLE_MODELS):
        print(f"  {i}: {m}")

    if args.model is None:
        choice = input("\nSelect model index or 'all': ").strip()
    else:
        choice = args.model
        print(f"\nUsing model selection from CLI: {choice}")

    if choice == "all":
        selected_models = AVAILABLE_MODELS

    elif choice.isdigit():
        selected_models = [AVAILABLE_MODELS[int(choice)]]

    elif choice in AVAILABLE_MODELS:
        selected_models = [choice]

    else:
        raise ValueError(
            f"Unknown model '{choice}'. "
            f"Choose 'all', an index, or one of:\n"
            + "\n".join(AVAILABLE_MODELS)
        )

    print(f"Selected: {selected_models}")

    latest_timestamp, file_pairs = get_latest_file_pairs()
    print(f"\nFound {len(file_pairs)} pairs for batch {latest_timestamp}")

    os.makedirs(EVAL_LOG_DIR, exist_ok=True)

    overall_correct, overall_total = 0, 0

    for idx, pred_file, gt_file in file_pairs:
        print(f"\n{'='*30}")
        print(f"Processing pair {idx}")
        print(f"Prediction:   {pred_file}")
        print(f"Ground truth: {gt_file}")
        print(f"{'='*30}\n")

        with open(pred_file)  as f: pred_json    = json.load(f)
        with open(gt_file)    as f: ground_truth = json.load(f)

        results = evaluate(client, selected_models, pred_json, ground_truth)

        correct = sum(r["is_xgboost_correct"] for r in results)
        total   = len(results)
        overall_correct += correct
        overall_total   += total

        print(f"\n--- Pair Summary ---")
        print(f"XGBoost accuracy for pair {idx}: {correct}/{total} = {correct/total:.2f}")

        out_file = os.path.join(EVAL_LOG_DIR, f"evaluation_{latest_timestamp}_{idx}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_file}")

    print("\n=== Overall Summary ===")
    if overall_total > 0:
        print(f"XGBoost Accuracy: {overall_correct}/{overall_total} = {overall_correct/overall_total:.2f}")
    else:
        print("No pairs evaluated.")


if __name__ == "__main__":
    main()
