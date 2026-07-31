#!/usr/bin/env python3
"""
(5/5)

Plot the retraining progress recorded in training_history.csv
(written by 3e_score_scraper.py, one row per main_pipeline.py run).

Produces:
- winner_score and retrained_score vs. iteration, so you can see whether the
  retrained model is closing the gap on (or beating) the best off-the-shelf
  model each iteration.
- winning model per iteration annotated on the chart, so you can see which
  model tends to win over time.

Usage:
  python 5_plot_training_history.py \
    --history json_log/3_evaluation_results/5b_history_log/training_history.csv \
    --output json_log/3_evaluation_results/5b_history_log/training_progress.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot retraining progress from training_history.csv"
    )
    parser.add_argument(
        "--history",
        default="json_log/3_evaluation_results/5b_history_log/training_history.csv",
    )
    parser.add_argument(
        "--output",
        default="json_log/3_evaluation_results/5b_history_log/training_progress.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive window (requires a display).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        raise SystemExit(f"No history file found at {history_path}")

    df = pd.read_csv(history_path)

    if df.empty:
        raise SystemExit(f"{history_path} has no rows yet.")

    df["retrained_score"] = pd.to_numeric(df["retrained_score"], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["iteration"], df["winner_score"],
        linewidth=0.75, color="lightgray", zorder=1,
    )

    palette = plt.get_cmap("tab10")
    for i, winner_model in enumerate(sorted(df["winner_model"].unique())):
        subset = df[df["winner_model"] == winner_model]
        ax.scatter(
            subset["iteration"], subset["winner_score"],
            label=f"winner: {winner_model}", color=palette(i % 10),
            s=25, zorder=2,
        )

    retrained = df.dropna(subset=["retrained_score"])
    if not retrained.empty:
        ax.plot(
            retrained["iteration"], retrained["retrained_score"],
            marker="s", label="retrained_score", color="tab:red", zorder=3,
        )
    else:
        print(
            "No retrained_score values yet — run 2c_retrain_expl.py at least "
            "once after training a LoRA adapter to start populating this line."
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("final_combined_score")
    ax.set_title("Retraining progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()

    # ----- text summary -----
    latest = df.iloc[-1]
    print("\n=== Latest iteration ===")
    print(f"Iteration:       {latest['iteration']}")
    print(f"Winner:          {latest['winner_model']} ({latest['winner_score']})")
    if pd.notna(latest["retrained_score"]):
        gap = latest["winner_score"] - latest["retrained_score"]
        print(f"Retrained score: {latest['retrained_score']} (gap to winner: {gap:+.1f})")
    else:
        print("Retrained score: not evaluated this iteration")

    retrained_wins = df["winner_model"].astype(str).str.contains("retrained", case=False)
    print(f"\nRetrained model has won {int(retrained_wins.sum())} / {len(df)} iterations so far.")


if __name__ == "__main__":
    main()
