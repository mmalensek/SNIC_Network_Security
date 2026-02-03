#!/usr/bin/env python3

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer

import warnings
warnings.filterwarnings("ignore", category=torch.cuda._utils._CreationPolicyWarning) 

# -------------------------------------------------------------------
# Environment (safe on ARM)
# -------------------------------------------------------------------
os.environ["HF_HOME"] = "/mnt/share/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/share/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/mnt/share/huggingface/datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Starting LoRA fine-tuning (ARM-safe)...")
print("CUDA available:", torch.cuda.is_available())

# -------------------------------------------------------------------
# Metadata
# -------------------------------------------------------------------
MODEL_ID = "microsoft/DialoGPT-small"
OUTPUT_DIR = "./lora_models/nids-dialogpt-lora"

CSV_PATH = {
    "train": "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv",
    "eval": "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv",
}

LABEL_COL = " Label"

COLUMN_DESCS = """<your column description here>""".strip()

# -------------------------------------------------------------------
# Load model & tokenizer (NO quantization)
# -------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.model_max_length = 512

print("Model loaded on:", next(model.parameters()).device)

# -------------------------------------------------------------------
# LoRA configuration
# -------------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
dataset = load_dataset("csv", data_files=CSV_PATH)
print(f"Loaded {len(dataset['train'])} train / {len(dataset['eval'])} eval")

# -------------------------------------------------------------------
# Prompt formatting
# -------------------------------------------------------------------
def formatting_func(example):
    label = str(example[LABEL_COL]).strip()

    flow_dict = {
        k.strip(): example[k]
        for k in example.keys()
        if k.strip() != "Label"
    }

    prompt = f"""Analyze this network flow for attacks.

Column descriptors:
{COLUMN_DESCS}

Flow:
{flow_dict}

Is this traffic malicious?
Answer ONLY with either BENIGN or MALICIOUS.
"""

    return prompt + "\n" + label

# -------------------------------------------------------------------
# Training arguments
# -------------------------------------------------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch=8
    learning_rate=5e-4,
    max_steps=50,
    logging_steps=5,
    save_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    fp16=True,  # Enforce for A30X
    dataloader_pin_memory=False,
    dataloader_num_workers=0,  # Stable on shared FS
)


# -------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    formatting_func=formatting_func,
    args=args,
)

# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------
print("Starting training...")
trainer.train()

# -------------------------------------------------------------------
# Save LoRA adapter
# -------------------------------------------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapter saved to {OUTPUT_DIR}")

# -------------------------------------------------------------------
# Merge LoRA with base model
# -------------------------------------------------------------------
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
)
merged_model = merged_model.merge_and_unload()

merged_path = OUTPUT_DIR + "-merged"
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

print(f"Merged model saved to {merged_path}")
