#!/usr/bin/env python

# set all huging face cache to external drive
import os
os.environ["HF_HOME"] = "/mnt/share/huggingface" 
os.environ["TRANSFORMERS_CACHE"] = "/mnt/share/huggingface/hub"  
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer

print("Starting LoRA fine-tuning...")

# meta data 
# !!! EVAL PATH CURRENTLY SET TO THE SAME DATASET JUST FOR TESTING, CHANGE LATER !!!
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
MODEL_ID = "microsoft/DialoGPT-small" # testing model, change later
CSV_PATH = {"train": "../../dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "eval": "../../dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"} # current dataset paths
FLOW_COL = "flow_data" # CSV column
LABEL_COL = " Label" # either benign or malicious !!! "_LABEL" !!!
OUTPUT_DIR = "./lora_models/nids-deepseek-lora"

# 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# load model/tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto", 
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# preparation for QLoRA
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# LoRA config
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["c_attn"]  # GPT2 combined attention layer
)
model = get_peft_model(model, lora_config)

# load CSV dataset
dataset = load_dataset("csv", data_files=CSV_PATH)
print(f"Loaded {len(dataset['train'])} train, {len(dataset['eval'])} eval samples")

# prompt formatter
def formatting_func(example):
    texts = []

    labels = example[LABEL_COL] 
    keys = example.keys()

    for i in range(len(labels)):
        flow_dict = {
            k.strip(): example[k][i]
            for k in keys
            if k.strip() != "Label"
        }

        record_str = str(flow_dict)

        prompt = f"""Analyze this network flow for attacks.
Column descriptors:
{COLUMN_DESCS}

Flow:
{record_str}

Is this traffic malicious?
Answer ONLY with either BENIGN or MALICIOUS."""
        
        # label must be string
        label = str(labels[i]).strip()

        texts.append(prompt + "\n" + label)

    return texts

# training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=5,
    max_steps=50,  # starting number, change later
    learning_rate=2e-4,
    logging_steps=5,
    save_steps=25,
    eval_strategy="steps",
    eval_steps=25,
    fp16=torch.cuda.is_available(),
    report_to=None, 
    remove_unused_columns=False,
)

# setup trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    formatting_func=formatting_func,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=args,
    max_seq_length=2048,
    packing=False
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Saved LoRA model to {OUTPUT_DIR}")

# merge (for Ollama)
merged_model = AutoPeftModelForCausalLM.from_pretrained(OUTPUT_DIR, device_map="auto")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR + "-merged")
tokenizer.save_pretrained(OUTPUT_DIR + "-merged")
print("Merged model ready for GGUF to Ollama")