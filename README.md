## FlowExplain: Explainable Network Intrusion Detection with LLMs

Implementation for my diploma thesis on explainable network intrusion detection. An XGBoost flow classifier is paired with a small, locally fine-tuned LLM that turns each prediction into a natural-language explanation and mitigation recommendation, intended to run on a smart NIC (SNIC).

Candidate explanations are generated in parallel by a local model (Ollama), a cloud reference model (OpenAI), and the current fine-tuned checkpoint, then scored automatically (deterministic checks + LLM-as-judge, optionally a human expert comparison) to select training examples for the next LoRA fine-tuning round.

## Pipeline

1. `1_xgb_agg.py` — XGBoost prediction on selected CICIDS2017 flows
2. `2a_ollama_expl.py` / `2b_openai_expl.py` / `2c_retrain_expl.py` — explanation generation (local, cloud, current fine-tuned model)
3. `3a`–`3e` — deterministic, LLM-judge, and optional human-expert scoring, combined into a weighted score and a per-round winner
4. `4a_training_prepare.py` — build the fine-tuning dataset from winning explanations
5. `4b_unsloth_finetune.py` — LoRA fine-tuning (separate `retrain` environment)

`main_pipeline.py` runs steps 1–4a in one command:

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

Fine-tuning is run separately:

```bash
conda activate retrain
python 4b_unsloth_finetune.py
```

## Environments

- **xgboost** — classifier, explanation generation, scoring, dataset prep
- **retrain** — Unsloth, PyTorch, LoRA fine-tuning

Requires Python 3.11+, a CUDA GPU for fine-tuning, an OpenAI API key (optional, for the cloud reference model), and the CICIDS2017 dataset.

## Acknowledgements

Mentor: Assoc. Prof. Dr. Veljko Pejović
Co-mentor: Assist. Miha Grohar

## Author

[Martin Malenšek](https://github.com/mmalensek)
