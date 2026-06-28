## Explainable Network Intrusion Detection with Large Language Models

This repository contains the implementation developed for my diploma thesis on **Explainable Network Intrusion Detection using Large Language Models (LLMs)**.

The goal of this work is to investigate how LLMs can improve the interpretability of machine learning-based intrusion detection systems by generating human-readable explanations and mitigation strategies for detected network attacks.

The project combines a traditional XGBoost-based intrusion detection model with multiple large language models (both local and cloud-hosted), evaluates the quality of their explanations using several scoring methods, and automatically prepares high-quality datasets for iterative fine-tuning of local LLMs.

## Features

* XGBoost binary and multiclass intrusion detection
* Support for both local (Ollama) and OpenAI language models
* Automatic generation of:

  * reasoning
  * mitigation recommendations
* Multiple evaluation methods:

  * deterministic scoring
  * expert-system scoring
  * human evaluation
  * weighted score aggregation
* Automatic selection of the highest-quality explanations
* Training dataset generation for supervised fine-tuning
* LoRA fine-tuning using Unsloth
* Support for iterative improvement of local LLMs

## Project Pipeline

The complete workflow consists of the following stages:

1. **XGBoost classification**

   * Generate predictions for selected network traffic samples.

2. **LLM explanation generation**

   * Ollama models
   * OpenAI models
   * (optional) previously fine-tuned local model

3. **Evaluation**

   * Deterministic scoring
   * Expert-system scoring
   * Human evaluation (optional)
   * Combined weighted score

4. **Winner selection**

   * Select the highest-quality explanation for each sample.

5. **Dataset preparation**

   * Convert winning explanations into a JSONL dataset suitable for supervised fine-tuning.

6. **Fine-tuning**

   * Train a local model using Unsloth LoRA.

## Requirements

* Python 3.11+
* XGBoost
* Ollama
* CUDA-capable GPU (recommended for fine-tuning)
* OpenAI API key (optional, for OpenAI evaluation)
* CICIDS2017 (or compatible) dataset

The project currently uses two Conda environments:

* **xgboost**

  * evaluation pipeline
  * XGBoost
  * OpenAI/Ollama scripts
  * scoring
  * dataset preparation

* **retrain**

  * Unsloth
  * PyTorch
  * CUDA
  * LoRA fine-tuning

## Running the Evaluation Pipeline

The complete evaluation pipeline can be executed using:

```bash
python main_pipeline.py \
    --classifier multiclass \
    --labels 2 \
    --limit 1 \
    --pairs 100 \
    --ollama-model deepseek-r1:8b \
    --openai-model gpt-5.2 \
    --skip-retrained \
    --skip-human-evaluation
```

This pipeline performs:

* XGBoost prediction
* explanation generation
* automatic scoring
* winner selection
* training dataset generation

By default, the fine-tuning step is executed separately inside the `retrain` Conda environment.

## Fine-Tuning

After the evaluation pipeline has generated the training dataset, activate the retraining environment and run:

```bash
conda activate retrain

python 4b_unsloth_finetune.py
```

The resulting LoRA adapter can then be converted into an Ollama model and used as an additional explanation model in future evaluation iterations.

## Keywords

* Explainable Artificial Intelligence (XAI)
* Large Language Models (LLMs)
* Network Intrusion Detection
* Cybersecurity
* Machine Learning
* XGBoost
* Fine-Tuning
* LoRA
* Unsloth
* Ollama

## Acknowledgements

**Mentor**

* Assoc. Prof. Dr. Veljko Pejović

**Co-mentor**

* Assist. Miha Grohar

## Author

[Martin Malenšek](https://github.com/mmalensek)
