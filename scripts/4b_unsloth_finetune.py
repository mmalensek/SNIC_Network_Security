#!/usr/bin/env python3
"""
(4b/4)

Fine-tune a model using Unsloth.

Usage:
  python 4b_unsloth_finetune.py \
    --input-glob "json_log/3_evaluation_results/6_score_winner/*.json" \
    --max-samples 50 \
    --model-name network-intrusion-ollama \
    --base-model llama3.1 \
    --output-dir output/ollama_training
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B"

# everything is saved to external storage
OUTPUT_DIR = "/mnt/share/tmp/intrusion_lora"

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
    data_files="/home/ubuntu/martinmalensek_diploma/SNIC_Network_Security/scripts/ollama_training/training_dataset.jsonl",
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
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
    ),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)