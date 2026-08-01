"""
(2c/4)

Evaluation script for the retrained DeepSeek-R1 LoRA model.

This script replaces the previous Ollama-based evaluation by loading the
fine-tuned LoRA adapter directly with Unsloth and Transformers.

By default it evaluates the most recently trained adapter under
BASE_LORA_DIR (as pointed to by "latest_run.txt", written by
4b_unsloth_finetune.py). Pass --model-path to evaluate a specific adapter
instead.

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
import re
import json
import time
import argparse
import torch
from pathlib import Path
from datetime import datetime
from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizerFast

# CONFIGURATION

# base directory holding timestamped run_* adapters produced by
# 4b_unsloth_finetune.py, plus its "latest_run.txt" pointer
BASE_LORA_DIR = "/mnt/share/tmp/intrusion_lora"
JSON_LOG_DIR = "json_log/1_groundtruth_and_xgboost_prediction"
EVAL_LOG_DIR = "json_log/2_retrained_evaluation"
MAX_NEW_TOKENS = 512

# must match whatever --max-seq-length 4b_unsloth_finetune.py was trained
# with, or the adapter will see prompts shaped differently than in training
MAX_SEQ_LENGTH = 16384

model = None
tokenizer = None
MODEL_PATH = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the most recently retrained LoRA adapter."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a specific LoRA adapter directory. Defaults to the most "
             "recently trained run under --base-dir."
    )
    parser.add_argument(
        "--base-dir",
        default=BASE_LORA_DIR,
        help="Base directory containing timestamped run_* adapters from "
             "4b_unsloth_finetune.py."
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Should match the --max-seq-length used to train the adapter.",
    )
    return parser.parse_args()


def resolve_model_path(base_dir, explicit_path=None):
    if explicit_path:
        return explicit_path

    base = Path(base_dir)
    pointer_file = base / "latest_run.txt"

    if pointer_file.exists():
        run_name = pointer_file.read_text(encoding="utf-8").strip()
        candidate = base / run_name
        if candidate.exists():
            return str(candidate)

    # fall back to the most recently modified run_* directory
    run_dirs = sorted(
        (p for p in base.glob("run_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if run_dirs:
        return str(run_dirs[-1])

    raise FileNotFoundError(
        f"No retrained LoRA adapter found under {base}. "
        f"Run 4b_unsloth_finetune.py first, or pass --model-path explicitly."
    )


def load_model(model_path, max_seq_length=MAX_SEQ_LENGTH):
    print("Loading retrained model...")

    base_model = "unsloth/DeepSeek-R1-Distill-Llama-8B"

    base, tok = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # see matching comment in 4b_unsloth_finetune.py: AutoTokenizer resolves
    # this checkpoint to a broken slow tokenizer that corrupts word
    # boundaries on both encode and decode. Must use the same fix here or
    # eval prompts won't even match what the model was trained on.
    tok = PreTrainedTokenizerFast.from_pretrained(base_model)

    from peft import PeftModel

    merged = PeftModel.from_pretrained(base, model_path)

    print("Model loaded successfully.\n")
    return merged, tok



# FIND LATEST PREDICTION/GROUND TRUTH FILES

def get_latest_file_pairs():
    files = os.listdir(JSON_LOG_DIR)

    pred_pattern = re.compile(
        r"^prediction_(\d{8}_\d{6})(?:_(\d+))?\.json$"
    )

    gt_pattern = re.compile(
        r"^ground_truth_(\d{8}_\d{6})(?:_(\d+))?\.json$"
    )

    pred_matches = []
    gt_matches = []

    for f in files:

        pred_match = pred_pattern.match(f)

        if pred_match:
            pred_matches.append(
                (
                    pred_match.group(1),
                    int(pred_match.group(2)) if pred_match.group(2) else 1,
                    f,
                )
            )

        gt_match = gt_pattern.match(f)

        if gt_match:
            gt_matches.append(
                (
                    gt_match.group(1),
                    int(gt_match.group(2)) if gt_match.group(2) else 1,
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
#
# This MUST match the prompt/message structure that 4a_training_prepare.py
# used to build the training set (user_text + DEFAULT_SYSTEM), otherwise the
# retrained model is evaluated out-of-distribution and produces garbage.

DEFAULT_SYSTEM = (
    "You are a network security expert specializing in intrusion detection. "
    "Given a network flow, an actual label, and an XGBoost prediction, "
    "produce a LABEL, REASONING, and SOLUTION section. "
    "The REASONING should explain the traffic characteristics and why they "
    "match the label. The SOLUTION should provide practical mitigation and "
    "response recommendations."
)


def build_prompt(pred_json, actual_label):

    current_flow = pred_json.get("current_flow") or pred_json.get("row_data")
    # keep only the single closest neighboring flow on each side — must match
    # the trimming in 4a_training_prepare.py's extract_example()
    previous_flows = pred_json.get("previous_flows", [])[-1:]
    next_flows = pred_json.get("next_flows", [])[:1]
    probabilities = pred_json.get("probabilities", {})
    xgb_label = pred_json.get(
        "predicted_class_label",
        pred_json.get("model_prediction", ""),
    )

    return (
        "Analyze this network traffic.\n\n"
        "Provide your response EXACTLY in the following format:\n\n"
        "LABEL:\n"
        "[attack label]\n\n"
        "REASONING:\n"
        "[analysis]\n\n"
        "SOLUTION:\n"
        "[recommended mitigations]\n\n"
        f"Actual label: {actual_label}\n"
        f"XGBoost prediction: {xgb_label}\n\n"
        "Current flow:\n"
        f"{json.dumps(current_flow, indent=2)}\n\n"
        "Previous flows:\n"
        f"{json.dumps(previous_flows, indent=2)}\n\n"
        "Next flows:\n"
        f"{json.dumps(next_flows, indent=2)}\n\n"
        "XGBoost probabilities:\n"
        f"{json.dumps(probabilities, indent=2)}"
    )



# GENERATE MODEL RESPONSE

@torch.inference_mode()
def query_model(prompt):

    messages = [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM,
        },
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
    xgboost_label = pred_json.get("model_prediction", "UNKNOWN")
    xgboost_correct = (xgboost_label == true_label)

    run_id = pred_json.get("run_id")
    sample_id = pred_json.get("sample_id")

    print("\nGenerating response from retrained model...\n")

    prompt = build_prompt(pred_json, true_label)

    # cas od flowa (prompta) do odgovora modela
    start_time = time.time()
    response = query_model(prompt)
    response_time_sec = time.time() - start_time

    response_parts = extract_response_parts(response)
    reasoning = response_parts["reasoning"]
    solution = response_parts["solution"]

    # dolzina odgovora
    response_length_chars = len(response)
    response_length_words = len(response.split())

    result = {
        "run_id": run_id,
        "sample_id": sample_id,
        "model": f"retrained-{os.path.basename(MODEL_PATH)}",
        "xgboost_predicted_label": xgboost_label,
        "actual_label": true_label,
        "is_xgboost_correct": xgboost_correct,
        "reasoning": reasoning,
        "solution": solution,
        "raw_response": response,
        # NOVO: dodatne metrike
        "response_time_sec": round(response_time_sec, 4),
        "response_length_chars": response_length_chars,
        "response_length_words": response_length_words,
    }

    results.append(result)

    print(f"XGBoost Predicted: {xgboost_label}")
    print(f"True Label:        {true_label}")
    print(f"XGBoost Correct:   {xgboost_correct}")
    print(f"Response Time:     {response_time_sec:.2f}s")
    print(f"Response Length:   {response_length_chars} chars / {response_length_words} words")

    print("\n-------------------------------")
    print("REASONING")
    print("-------------------------------")
    print(reasoning)

    print("\n-------------------------------")
    print("SOLUTION")
    print("-------------------------------")
    print(solution)

    return results


def main():
    global model, tokenizer, MODEL_PATH

    args = parse_args()
    MODEL_PATH = resolve_model_path(args.base_dir, args.model_path)

    print("=" * 60)
    print("Retrained DeepSeek-R1 LoRA Evaluation")
    print("=" * 60)
    print(f"Using adapter: {MODEL_PATH}\n")

    model, tokenizer = load_model(MODEL_PATH, args.max_seq_length)

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

        suffix = "" if len(file_pairs) == 1 else f"_{idx}"

        out_file = os.path.join(
            EVAL_LOG_DIR,
            f"evaluation_{latest_timestamp}{suffix}.json",
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