#!/usr/bin/env python3

import os
import gc
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer

# Disable parallelism and set memory-safe environment
os.environ["HF_HOME"] = "/mnt/share/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/share/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/mnt/share/huggingface/datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Starting stable LoRA fine-tuning (ARM-safe)...")

# Metadata
MODEL_ID = "microsoft/DialoGPT-small"
OUTPUT_DIR = "./lora_models/nids-dialogpt-lora"
CSV_PATH = {
    "train": "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv",
    "eval": "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv",
}
LABEL_COL = " Label"

# Load model with minimal memory usage
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True,
)

# Force CPU placement initially
model = model.cpu()
logger.info(f"Model loaded on CPU with {sum(p.numel() for p in model.parameters()):,} parameters")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 512
tokenizer.pad_token = tokenizer.eos_token

# Conservative LoRA config for stability
lora_config = LoraConfig(
    r=4,  # Reduced rank
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"],  # DialoGPT-specific
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
logger.info("LoRA model prepared successfully")

# Load and process dataset (small subset first)
dataset = load_dataset("csv", data_files=CSV_PATH)
print(f"Full dataset: {len(dataset['train'])} train / {len(dataset['eval'])} eval")

# Take small subset for testing stability
train_dataset = dataset["train"].select(range(min(1000, len(dataset["train"]))))
eval_dataset = dataset["eval"].select(range(min(200, len(dataset["eval"]))))
print(f"Using subset: {len(train_dataset)} train / {len(eval_dataset)} eval")

def formatting_func(example):
    """Simplified formatting to avoid memory issues"""
    label = str(example[LABEL_COL]).strip()
    
    # Select only key features to reduce prompt size
    key_features = {
        "Flow Duration": example.get(" Flow Duration", 0),
        "Total Fwd Packets": example.get(" Total Fwd Packets", 0),
        "Total Bwd Packets": example.get(" Total Backward Packets", 0),
        "Flow Bytes/s": example.get("Flow Bytes/s", 0),
        " Label": label
    }
    
    prompt = f"""Analyze network flow:
{key_features}

Malicious? {label}"""
    
    return prompt

# Ultra-conservative training args
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Reduced
    learning_rate=2e-4,  # Lower LR
    max_steps=20,  # Very small for testing
    warmup_steps=5,
    logging_steps=1,
    save_steps=10,
    evaluation_strategy="steps",
    eval_steps=10,
    fp16=False,
    bf16=False,  # Disable mixed precision
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    gradient_checkpointing=False,
    optim="adamw_hf",
    remove_unused_columns=False,
    report_to=None,  # Disable wandb/tensorboard
    save_total_limit=2,
    load_best_model_at_end=False,
)

# Create trainer with explicit args
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    args=args,
    max_seq_length=256,  # Reduced
    packing=False,
    tokenizer=tokenizer,
)

# Clear memory before training
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# Save LoRA adapter
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")

# Skip merging for now (common crash point on ARM)
print("Skipping model merge to avoid core dump. Use PEFT model directly.")
