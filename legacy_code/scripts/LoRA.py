#!/usr/bin/env python3

"""
pip install \
  torch==2.1.2 \
  transformers==4.36.2 \
  trl==0.7.11 \
  peft==0.7.1 \
  tokenizers==0.15.2 \
  accelerate==0.27.2
"""

import os
import gc
import torch
import logging
import subprocess
import shutil
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,AutoTokenizer,TrainingArguments)
from pathlib import Path
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
import peft
peft.utils.other.is_bnb_available = lambda: False

# exporting hugging face cache to external drive
os.environ["HF_HOME"] = "/mnt/share/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/share/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/mnt/share/huggingface/datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Starting stable LoRA fine-tuning (ARM-safe)...")

# metadata
MODEL_ID = "microsoft/DialoGPT-small"
OUTPUT_DIR = "./lora_models/nids-dialogpt-lora"
CSV_PATH = {
    "train": "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv",
    "eval": "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv",
}
LABEL_COL = " Label"
OLLAMA_MODELS_DIR = Path("/mnt/share/ollama/models")

# load model with minimal memory usage
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True,
)

# force CPU placement initially
model = model.cpu()
logger.info(f"Model loaded on CPU with {sum(p.numel() for p in model.parameters()):,} parameters")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 128 # set lower for testing!!!!
tokenizer.pad_token = tokenizer.eos_token

# LoRA config 
lora_config = LoraConfig(
    r=4,  # Reduced rank
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"],  # DialoGPT-specific
    inference_mode=False
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
logger.info("LoRA model prepared successfully")

# load and process dataset - small subset currently!
dataset = load_dataset("csv", data_files=CSV_PATH)
print(f"Full dataset: {len(dataset['train'])} train / {len(dataset['eval'])} eval")

# take small subset for testing stability
train_dataset = dataset["train"].select(range(min(100, len(dataset["train"])))) # min set as 100 for testing, raise later
eval_dataset = dataset["eval"].select(range(min(100, len(dataset["eval"])))) # min set as 100 for testing, raise later
print(f"Using subset: {len(train_dataset)} train / {len(eval_dataset)} eval")

# setting up the formatting function for the model
def formatting_func(batch):
    prompts = []

    labels = batch[LABEL_COL]
    flow_durations = batch.get(" Flow Duration", [0] * len(labels))
    fwd_packets = batch.get(" Total Fwd Packets", [0] * len(labels))
    bwd_packets = batch.get(" Total Backward Packets", [0] * len(labels))
    flow_bytes = batch.get("Flow Bytes/s", [0] * len(labels))

    for i in range(len(labels)):
        label = str(labels[i]).strip()

        key_features = {
            "Flow Duration": flow_durations[i],
            "Total Fwd Packets": fwd_packets[i],
            "Total Bwd Packets": bwd_packets[i],
            "Flow Bytes/s": flow_bytes[i],
            "Label": label,
        }

        prompt = f"""Analyze network flow:
{key_features}

Malicious? {label}"""

        prompts.append(prompt)

    return prompts  



# setting up training arguments - currently conservative for testing
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # reduced
    learning_rate=2e-4,  # lower LR
    max_steps=20,  # very small for testing
    warmup_steps=5,
    logging_steps=1,
    save_steps=10,
    evaluation_strategy="steps",
    eval_steps=10,
    fp16=False,
    bf16=False,  # disable mixed precision
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    gradient_checkpointing=False,
    optim="adamw_hf",
    remove_unused_columns=True,
    report_to=None,  # disable wandb/tensorboard
    save_total_limit=2,
    load_best_model_at_end=False,
)

# creating trainer with explicit args
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

# clear memory before training
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# save LoRA adapter
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")

# exporting LoRA parameters to ollama
model_name = "nids_dialogpt" # change model_name here aswell!!!
lora_path = Path(OUTPUT_DIR)

print(f"\nExporting '{model_name}' to Ollama...")

# 1. merge LoRA with base model
print("1. Merging LoRA adapter into base model...")
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    lora_path,
    torch_dtype=torch.float32,
    device_map=None,
)
merged_model = merged_model.merge_and_unload()
merged_path = lora_path / "merged"
merged_path.mkdir(exist_ok=True)
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

print(f"Merged model saved to: {merged_path}")

# Free memory early (important on ARM)
del merged_model
gc.collect()

# 2. convert merged HF model to GGUF (model file format for ollama)
print("2. Converting merged model to GGUF...")

gguf_path = lora_path / f"{model_name}.gguf"

convert_cmd = [
    "python", "-m", "llama_cpp.convert",
    "--model-path", str(merged_path),
    "--outfile", str(gguf_path),
    "--outtype", "f16",
    "--vocab-type", "bpe",
]

result = subprocess.run(convert_cmd, capture_output=True, text=True)

if result.returncode != 0:
    logger.error("GGUF conversion failed")
    logger.error(result.stderr)
    raise RuntimeError("GGUF conversion failed")

print(f"GGUF created at: {gguf_path}")

# 3. Create Ollama Modelfile
print("3. Creating Ollama Modelfile...")

modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9

SYSTEM \"\"\"You are a Network Intrusion Detection System (NIDS).
Analyze network flows and respond ONLY with BENIGN or MALICIOUS.\"\"\"
"""

modelfile_path = lora_path / "Modelfile"
modelfile_path.write_text(modelfile_content)

print(f"Modelfile written to: {modelfile_path}")

# 4. Copy GGUF to Ollama blobs directory
print("4. Copying model into Ollama directory...")

OLLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
ollama_blobs_dir = OLLAMA_MODELS_DIR / "blobs"
ollama_blobs_dir.mkdir(parents=True, exist_ok=True)

shutil.copy2(gguf_path, ollama_blobs_dir / gguf_path.name)

print("GGUF copied to Ollama blobs")

# 5. Create Ollama model
print("5. Creating Ollama model...")

ollama_cmd = [
    "ollama", "create", model_name,
    "-f", str(modelfile_path)
]

result = subprocess.run(ollama_cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"\nOllama model '{model_name}' is ready")
else:
    logger.error("Ollama model creation failed")
    logger.error(result.stderr)
    raise RuntimeError("Ollama create failed")