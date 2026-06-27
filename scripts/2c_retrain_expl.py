"""
(2c/4)

Evaluation script for the retrained DeepSeek-R1 LoRA model.

This script replaces the previous Ollama-based evaluation by loading the
fine-tuned LoRA adapter directly with Unsloth and Transformers.

The evaluation still compares:

- XGBoost predicted label
- Ground truth label

The LLM itself is only evaluated qualitatively based on its generated
REASONING and SOLUTION.

Prerequisites:
    unsloth
    transformers
    torch
    datasets
    peft
    json
"""

import os
import json
import re
from datetime import datetime
import torch
from unsloth import FastLanguageModel

# CONFIGURATION

# path to trained LoRA adapter
MODEL_PATH = "/mnt/share/tmp/intrusion_lora/checkpoint-3"
JSON_LOG_DIR = "json_log/1_groundtruth_and_xgboost_prediction"
EVAL_LOG_DIR = "json_log/2_retrained_evaluation"
MAX_NEW_TOKENS = 512


# LOAD MODEL

print("Loading retrained model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=4096,
    load_in_4bit=True,
)

base_model = "unsloth/DeepSeek-R1-Distill-Llama-8B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=4096,
    load_in_4bit=True,
)

from peft import PeftModel

model = PeftModel.from_pretrained(
    model,
    MODEL_PATH,
)

print("Model loaded successfully.\n")



# FIND LATEST PREDICTION/GROUND TRUTH FILES

def get_latest_file_pairs():
    files = os.listdir(JSON_LOG_DIR)

    pred_pattern = re.compile(
        r"^prediction_(\d{8}_\d{6})_(\d+)\.json$"
    )

    gt_pattern = re.compile(
        r"^ground_truth_(\d{8}_\d{6})_(\d+)\.json$"
    )

    pred_matches = []
    gt_matches = []

    for f in files:

        pred_match = pred_pattern.match(f)

        if pred_match:
            pred_matches.append(
                (
                    pred_match.group(1),
                    int(pred_match.group(2)),
                    f,
                )
            )

        gt_match = gt_pattern.match(f)

        if gt_match:
            gt_matches.append(
                (
                    gt_match.group(1),
                    int(gt_match.group(2)),
                    f,
                )
            )

    if not pred_matches or not gt_matches:
        raise FileNotFoundError(
            "No numbered prediction/ground truth files found."
        )

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

    common_indices = sorted(
        set(pred_map.keys()) &
        set(gt_map.keys())
    )

    if not common_indices:
        raise FileNotFoundError(
            "No matching prediction/ground truth pairs found."
        )

    return latest_timestamp, [
        (idx, pred_map[idx], gt_map[idx])
        for idx in common_indices
    ]



# BUILD PROMPT

def build_prompt(pred_json):

    return f"""
You are a cybersecurity expert.

Analyze the following network traffic summary produced by an XGBoost classifier.

Provide your response EXACTLY and ONLY in the following format.

REASONING:
[Explain what is happening in the traffic, whether it indicates an attack,
what evidence supports your conclusion, and if possible identify the attack.]

SOLUTION:
[Provide concrete recommendations or mitigation strategies.]

JSON:
{json.dumps(pred_json, indent=2)}
"""



# GENERATE MODEL RESPONSE

@torch.inference_mode()
def query_model(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return response


# EXTRACT REASONING + SOLUTION

def extract_response_parts(text):

    reasoning = ""
    solution = ""

    if "REASONING:" in text:

        start = text.index("REASONING:") + len("REASONING:")

        end = (
            text.find("SOLUTION:")
            if "SOLUTION:" in text
            else len(text)
        )

        reasoning = text[start:end].strip()

    if "SOLUTION:" in text:

        start = text.index("SOLUTION:") + len("SOLUTION:")

        solution = text[start:].strip()

    return {
        "reasoning": reasoning,
        "solution": solution,
    }


# EVALUATE A SINGLE PREDICTION / GROUND TRUTH PAIR

def evaluate(pred_json, ground_truth):

    results = []

    true_label = ground_truth["most_common_true_label"]

    # XGBoost prediction (the LLM does NOT predict the label)
    xgboost_label = pred_json.get("model_prediction", "UNKNOWN")

    xgboost_correct = (xgboost_label == true_label)

    print("\nGenerating response from retrained model...\n")

    prompt = build_prompt(pred_json)

    response = query_model(prompt)

    response_parts = extract_response_parts(response)

    reasoning = response_parts["reasoning"]
    solution = response_parts["solution"]

    result = {
        "model": os.path.basename(MODEL_PATH),
        "xgboost_predicted_label": xgboost_label,
        "actual_label": true_label,
        "is_xgboost_correct": xgboost_correct,
        "reasoning": reasoning,
        "solution": solution,
        "raw_response": response,
    }

    results.append(result)

    print(f"XGBoost Predicted: {xgboost_label}")
    print(f"True Label:        {true_label}")
    print(f"XGBoost Correct:   {xgboost_correct}")

    print("\n-------------------------------")
    print("REASONING")
    print("-------------------------------")
    print(reasoning)

    print("\n-------------------------------")
    print("SOLUTION")
    print("-------------------------------")
    print(solution)

    return results



# MAIN

def main():

    print("=" * 60)
    print("Retrained DeepSeek-R1 LoRA Evaluation")
    print("=" * 60)

    latest_timestamp, file_pairs = get_latest_file_pairs()

    print(
        f"\nFound {len(file_pairs)} prediction/ground truth "
        f"pairs for batch {latest_timestamp}"
    )

    os.makedirs(EVAL_LOG_DIR, exist_ok=True)

    overall_correct = 0
    overall_total = 0

    for idx, pred_file, gt_file in file_pairs:

        print("\n" + "=" * 60)
        print(f"Processing pair {idx}")
        print("=" * 60)

        print(f"Prediction file : {pred_file}")
        print(f"Ground truth    : {gt_file}")

        with open(pred_file, "r") as f:
            pred_json = json.load(f)

        with open(gt_file, "r") as f:
            ground_truth = json.load(f)

        results = evaluate(
            pred_json,
            ground_truth,
        )

        correct = sum(
            r["is_xgboost_correct"]
            for r in results
        )

        total = len(results)

        overall_correct += correct
        overall_total += total

        print("\nPair Summary")
        print("-------------------------------")
        print(
            f"XGBoost Accuracy: "
            f"{correct}/{total} = {correct/total:.2f}"
        )

        out_file = os.path.join(
            EVAL_LOG_DIR,
            f"evaluation_{latest_timestamp}_{idx}.json",
        )

        with open(out_file, "w") as f:
            json.dump(
                results,
                f,
                indent=2,
            )

        print(f"\nSaved evaluation to:\n{out_file}")

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    print(
        f"XGBoost Accuracy: "
        f"{overall_correct}/{overall_total} = "
        f"{overall_correct/overall_total:.2f}"
    )

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()