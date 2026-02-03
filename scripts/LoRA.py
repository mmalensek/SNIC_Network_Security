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

# export hugging face cache to external drive
os.environ["HF_HOME"] = "/mnt/share/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/share/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/mnt/share/huggingface/datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Starting LoRA fine-tuning (ARM-safe)...")
print("CUDA available:", torch.cuda.is_available())

# meta data
MODEL_ID = "microsoft/DialoGPT-small"
OUTPUT_DIR = "./lora_models/nids-dialogpt-lora"
CSV_PATH = {
    "train": "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv",
    "eval": "../../dataset/TrafficLabelling/Friday-DDos-SHORTENED.csv",
}
LABEL_COL = " Label"
COLUMN_DESCS = """
- Destination Port: Destination TCP/UDP port number
- Flow Duration: Total duration of the flow in microseconds
- Total Fwd Packets: Total number of packets sent in the forward direction
- Total Backward Packets: Total number of packets sent in the backward direction
- Total Length of Fwd Packets: Total bytes sent in the forward direction
- Total Length of Bwd Packets: Total bytes sent in the backward direction
- Fwd Packet Length Max: Maximum packet length in the forward direction
- Fwd Packet Length Min: Minimum packet length in the forward direction
- Fwd Packet Length Mean: Mean packet length in the forward direction
- Fwd Packet Length Std: Standard deviation of packet length in the forward direction
- Bwd Packet Length Max: Maximum packet length in the backward direction
- Bwd Packet Length Min: Minimum packet length in the backward direction
- Bwd Packet Length Mean: Mean packet length in the backward direction
- Bwd Packet Length Std: Standard deviation of packet length in the backward direction
- Flow Bytes/s: Number of flow bytes per second
- Flow Packets/s: Number of flow packets per second
- Flow IAT Mean: Mean inter-arrival time between packets in the flow
- Flow IAT Std: Standard deviation of flow inter-arrival times
- Flow IAT Max: Maximum inter-arrival time in the flow
- Flow IAT Min: Minimum inter-arrival time in the flow
- Fwd IAT Total: Total inter-arrival time of forward packets
- Fwd IAT Mean: Mean inter-arrival time of forward packets
- Fwd IAT Std: Standard deviation of forward inter-arrival times
- Fwd IAT Max: Maximum inter-arrival time of forward packets
- Fwd IAT Min: Minimum inter-arrival time of forward packets
- Bwd IAT Total: Total inter-arrival time of backward packets
- Bwd IAT Mean: Mean inter-arrival time of backward packets
- Bwd IAT Std: Standard deviation of backward inter-arrival times
- Bwd IAT Max: Maximum inter-arrival time of backward packets
- Bwd IAT Min: Minimum inter-arrival time of backward packets
- Fwd PSH Flags: Number of PSH flags set in forward packets
- Bwd PSH Flags: Number of PSH flags set in backward packets
- Fwd URG Flags: Number of URG flags set in forward packets
- Bwd URG Flags: Number of URG flags set in backward packets
- Fwd Header Length: Total bytes used for headers in forward packets
- Bwd Header Length: Total bytes used for headers in backward packets
- Fwd Packets/s: Forward packet rate per second
- Bwd Packets/s: Backward packet rate per second
- Min Packet Length: Minimum packet length in the flow
- Max Packet Length: Maximum packet length in the flow
- Packet Length Mean: Mean packet length of the flow
- Packet Length Std: Standard deviation of packet length in the flow
- Packet Length Variance: Variance of packet lengths in the flow
- FIN Flag Count: Number of FIN flags set
- SYN Flag Count: Number of SYN flags set
- RST Flag Count: Number of RST flags set
- PSH Flag Count: Number of PSH flags set
- ACK Flag Count: Number of ACK flags set
- URG Flag Count: Number of URG flags set
- CWE Flag Count: Number of CWE flags set
- ECE Flag Count: Number of ECE flags set
- Down/Up Ratio: Ratio of backward to forward packets
- Average Packet Size: Average packet size of the flow
- Avg Fwd Segment Size: Average size of forward TCP segments
- Avg Bwd Segment Size: Average size of backward TCP segments
- Fwd Header Length.1: Duplicate forward header length feature
- Fwd Avg Bytes/Bulk: Average bytes per bulk transfer in forward direction
- Fwd Avg Packets/Bulk: Average packets per bulk transfer in forward direction
- Fwd Avg Bulk Rate: Average bulk transfer rate in forward direction
- Bwd Avg Bytes/Bulk: Average bytes per bulk transfer in backward direction
- Bwd Avg Packets/Bulk: Average packets per bulk transfer in backward direction
- Bwd Avg Bulk Rate: Average bulk transfer rate in backward direction
- Subflow Fwd Packets: Number of forward packets in subflows
- Subflow Fwd Bytes: Number of forward bytes in subflows
- Subflow Bwd Packets: Number of backward packets in subflows
- Subflow Bwd Bytes: Number of backward bytes in subflows
- Init_Win_bytes_forward: Initial TCP window size in forward direction
- Init_Win_bytes_backward: Initial TCP window size in backward direction
- act_data_pkt_fwd: Number of forward packets carrying application data
- min_seg_size_forward: Minimum segment size in forward direction
- Active Mean: Mean active time of the flow
- Active Std: Standard deviation of active time
- Active Max: Maximum active time
- Active Min: Minimum active time
- Idle Mean: Mean idle time of the flow
- Idle Std: Standard deviation of idle time
- Idle Max: Maximum idle time
- Idle Min: Minimum idle time
- Label: Traffic label (BENIGN or attack type)
""".strip()

# loading of model and tokenizer
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

# lora configuration
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

# loading dataset
dataset = load_dataset("csv", data_files=CSV_PATH)
print(f"Loaded {len(dataset['train'])} train / {len(dataset['eval'])} eval")

# formatting the prompt
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

# setting the training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    max_steps=50,
    logging_steps=5,
    save_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    fp16=True,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)


# setting the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    formatting_func=formatting_func,
    args=args,
)

# actually running the training
print("Starting training...")
trainer.train()

# saving the LoRA adapter
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")

# merging the LoRA layer with the base model
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
)
merged_model = merged_model.merge_and_unload()
merged_path = OUTPUT_DIR + "-merged"
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"Merged model saved to {merged_path}")