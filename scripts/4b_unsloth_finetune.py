#!/usr/bin/env python3
"""
(4b/4)

Fine-tune a model using Unsloth on the dataset produced by
4a_training_prepare.py (ollama_training/training_dataset.jsonl).

Each run is saved to its own timestamped subdirectory under --output-dir
(e.g. .../run_20260731_101500), and a "latest_run.txt" pointer file is
updated to name it. This lets 2c_retrain_expl.py automatically evaluate
the most recently trained adapter, and lets training_history.csv actually
track whether retraining is improving across iterations, instead of every
run silently overwriting the previous one.

Usage:
  python 4b_unsloth_finetune.py \
    --dataset ollama_training/training_dataset.jsonl \
    --output-dir /mnt/share/tmp/intrusion_lora
"""

import argparse
from pathlib import Path
from datetime import datetime

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the intrusion-detection explainer model with Unsloth."
    )
    parser.add_argument(
        "--dataset",
        default="ollama_training/training_dataset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/share/tmp/intrusion_lora",
        help="Base directory; each run is saved to a timestamped subdirectory inside it.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    run_label = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = Path(args.output_dir)
    run_dir = base_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
    )

    dataset = load_dataset(
        "json",
        data_files=args.dataset,
        split="train",
    )

    def format_example(example):
        messages = example["messages"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": text}

    dataset = dataset.map(format_example)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        args=TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
        ),
    )

    trainer.train()

    model.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    (base_dir / "latest_run.txt").write_text(f"{run_label}\n", encoding="utf-8")

    print(f"\nSaved retrained adapter to: {run_dir}")
    print(f"Updated pointer: {base_dir / 'latest_run.txt'}")


if __name__ == "__main__":
    main()