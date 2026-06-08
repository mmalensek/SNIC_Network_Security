#!/usr/bin/env python3
"""
(4/4)

Create an Ollama training dataset from selected evaluation JSON outputs.

This script prepares:
- a JSONL conversation dataset
- a Modelfile for ollama create
- a test command template

Usage:
  python train_ollama_intrusion_detection.py \
    --input-glob "json_log/3_evaluation_results/6_score_winner/*.json" \
    --max-samples 50 \
    --model-name network-intrusion-ollama \
    --base-model llama3.1 \
    --output-dir output/ollama_training
"""

import argparse
import glob
import json
from pathlib import Path
from collections import Counter


DEFAULT_SYSTEM = (
    "You are a network security expert specializing in intrusion detection. "
    "Given a network flow, an actual label, and an XGBoost prediction, "
    "produce a LABEL, REASONING, and SOLUTION section. "
    "The REASONING should explain the traffic characteristics and why they "
    "match the label. The SOLUTION should provide practical mitigation and "
    "response recommendations."
)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_nested_record(obj):
    if isinstance(obj, dict):
        if "model_output" in obj and isinstance(obj["model_output"], dict):
            mo = obj["model_output"]
            if "evaluation_record" in mo and isinstance(mo["evaluation_record"], dict):
                return mo["evaluation_record"]
        for v in obj.values():
            found = find_nested_record(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_nested_record(item)
            if found is not None:
                return found
    return None


def extract_example(eval_obj, source_file):
    record = find_nested_record(eval_obj)
    if not record:
        raise ValueError(f"No evaluation_record found in {source_file}")

    row_data = record.get("row_data", {})
    label = record.get("predicted_class_label") or record.get("model_prediction") or "UNKNOWN"
    reasoning = record.get("reasoning", "")
    solution = record.get("solution", "")
    xgb_label = record.get("xgboost_predicted_label", "")
    actual_label = record.get("actual_label", "")

    user_text = (
        "Analyze this network flow.\n\n"
        "Provide your response EXACTLY in the following format:\n\n"
        "LABEL:\n"
        "[attack label]\n\n"
        "REASONING:\n"
        "[analysis]\n\n"
        "SOLUTION:\n"
        "[recommended mitigations]\n\n"
        f"Source: {source_file}\n"
        f"Actual label: {actual_label}\n"
        f"XGBoost label: {xgb_label}\n\n"
        f"Flow features:\n"
        f"{json.dumps(row_data, indent=2, ensure_ascii=False)}"
    )

    assistant_text = (
        f"LABEL:\n{label}\n\n"
        f"REASONING:\n{reasoning}\n\n"
        f"SOLUTION:\n{solution}"
    )

    return {
        "source_file": source_file,
        "label": label,
        "actual_label": actual_label,
        "user": user_text,
        "assistant": assistant_text,
        "row_data": row_data,
    }


def build_dataset(paths, max_samples=None):
    examples = []
    for path in paths:
        try:
            obj = read_json(path)
            examples.append(extract_example(obj, path))
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


def write_jsonl(examples, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            rec = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(examples, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)


def write_modelfile(out_path, base_model, system_prompt):
    content = f"""FROM {base_model}

SYSTEM {json.dumps(system_prompt)}
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_predict 1024
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def summarize(examples):
    labels = Counter(ex["label"] for ex in examples)
    actuals = Counter(ex["actual_label"] for ex in examples)
    print(f"Samples: {len(examples)}")
    print("Predicted labels:")
    for k, v in labels.most_common():
        print(f"  {k}: {v}")
    print("Actual labels:")
    for k, v in actuals.most_common():
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--model-name", default="deepseek-r1:8b-intrusion-retrained")
    ap.add_argument("--base-model", default="deepseek-r1:8b")
    ap.add_argument("--output-dir", default="ollama_training")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    examples = build_dataset(paths, args.max_samples)
    if not examples:
        raise SystemExit("No usable training examples were extracted.")

    json_path = outdir / "training_dataset.json"
    jsonl_path = outdir / "training_dataset.jsonl"
    modelfile_path = outdir / "Modelfile"
    commands_path = outdir / "ollama_commands.txt"

    write_json(examples, json_path)
    write_jsonl(examples, jsonl_path)
    write_modelfile(modelfile_path, args.base_model, args.system_prompt)

    commands = [
        f"ollama create {args.model_name} -f {modelfile_path}",
        f"ollama run {args.model_name} 'Classify this network flow ...'",
    ]
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

    summarize(examples)
    print(f"Saved: {json_path}")
    print(f"Saved: {jsonl_path}")
    print(f"Saved: {modelfile_path}")
    print(f"Saved: {commands_path}")
    print("\nNote: Ollama itself does not support classic supervised fine-tuning; this script prepares a dataset and a model wrapper you can use with ollama create.")


if __name__ == "__main__":
    main()
