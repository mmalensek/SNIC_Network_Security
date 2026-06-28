#!/usr/bin/env python3

"""
Main pipeline for the intrusion detection evaluation and retraining workflow.

Pipeline:

1_xgb_agg.py
2a_ollama_expl.py
2b_openai_expl.py
(optional) 2c_retrain_expl.py
3a_deterministic_score.py
3b_expert_system_score.py
3c_human_expert_score.py
3d_score_combined.py
3e_score_scraper.py
4a_training_prepare.py
4b_unsloth_finetune.py

USAGE: 

Without a retrained model:

python main_pipeline.py \
    --classifier multiclass \
    --labels 2 \
    --limit 1 \
    --pairs 100 \
    --ollama-model deepseek-r1:8b \
    --openai-model gpt-5.2 \
    --skip-retrained

With a retrained model:

python main_pipeline.py \
    --classifier multiclass \
    --labels 2 \
    --limit 1 \
    --pairs 100 \
    --ollama-model deepseek-r1:8b \
    --openai-model gpt-5.2

"""

import argparse
import subprocess
import sys


def run(command, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Running:")
    print(" ".join(command))
    print()

    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete evaluation and retraining pipeline."
    )

    # ---------- 1_xgb_agg ----------
    parser.add_argument(
        "--classifier",
        default="multiclass",
        choices=["binary", "multiclass"],
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated label indices."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--print-settings",
        type=int,
        default=2,
        choices=[1,2,3],
    )

    # ---------- explanation models ----------

    parser.add_argument(
        "--ollama-model",
        default="all",
        help="'all' or model index"
    )

    parser.add_argument(
        "--openai-model",
        default="0",
        help="'all' or model index"
    )

    parser.add_argument(
        "--skip-retrained",
        action="store_true",
        help="Skip 2c_retrain_expl.py"
    )

    # ---------- training ----------

    parser.add_argument(
        "--input-glob",
        default="json_log/3_evaluation_results/6_score_winner/*.json"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None
    )

    parser.add_argument(
        "--model-name",
        default="deepseek-r1-8b-intrusion"
    )

    parser.add_argument(
        "--base-model",
        default="deepseek-r1:8b"
    )

    parser.add_argument(
        "--output-dir",
        default="ollama_training"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    run(
        [
            "python",
            "1_xgb_agg.py",
            "--model-key", args.classifier,
            "--labels", args.labels,
            "--limit", str(args.limit),
            "--pairs", str(args.pairs),
            "--print-settings", str(args.print_settings),
        ],
        "STEP 1/10 : XGBoost aggregation",
    )

    run(
        [
            "python",
            "2a_ollama_expl.py",
            "--model",
            args.ollama_model,
        ],
        "STEP 2/10 : Ollama explanations",
    )

    run(
        [
            "python",
            "2b_openai_expl.py",
            "--model",
            args.openai_model,
        ],
        "STEP 3/10 : OpenAI explanations",
    )

    if not args.skip_retrained:

        run(
            [
                "python",
                "2c_retrain_expl.py",
            ],
            "STEP 4/10 : Retrained model explanations",
        )

    run(
        [
            "python",
            "3a_deterministic_score.py",
        ],
        "STEP 5/10 : Deterministic scoring",
    )

    run(
        [
            "python",
            "3b_expert_system_score.py",
        ],
        "STEP 6/10 : Expert scoring",
    )

    run(
        [
            "python",
            "3c_human_expert_score.py",
        ],
        "STEP 7/10 : Human scoring",
    )

    run(
        [
            "python",
            "3d_score_combined.py",
        ],
        "STEP 8/10 : Combined scoring",
    )

    run(
        [
            "python",
            "3e_score_scraper.py",
        ],
        "STEP 9/10 : Winner extraction",
    )

    cmd = [
        "python",
        "4a_training_prepare.py",
        "--input-glob",
        args.input_glob,
        "--model-name",
        args.model_name,
        "--base-model",
        args.base_model,
        "--output-dir",
        args.output_dir,
    ]

    if args.max_samples is not None:
        cmd.extend([
            "--max-samples",
            str(args.max_samples),
        ])

    run(
        cmd,
        "STEP 10/10 : Dataset generation",
    )

    run(
        [
            "python",
            "4b_unsloth_finetune.py",
        ],
        "STEP 11/11 : LoRA fine-tuning",
    )

    print("\n" + "=" * 80)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("\nPipeline failed.")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)