#!/usr/bin/env bash
# Run 10 independent multiclass pipeline invocations for labels 0 through 14.
#
# Usage (from the directory that contains main_pipeline.py):
#   bash run_multiclass_batch.sh
#
# Optional environment overrides:
#   ITERATIONS=20 FIRST_LABEL=2 LAST_LABEL=14 bash run_multiclass_batch.sh

set -uo pipefail

iterations="${ITERATIONS:-10}"
first_label="${FIRST_LABEL:-0}"
last_label="${LAST_LABEL:-14}"
log_dir="${LOG_DIR:-batch-logs-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f main_pipeline.py ]]; then
  echo "Error: run this script from the directory containing main_pipeline.py." >&2
  exit 1
fi

mkdir -p "$log_dir"
failures=0

for ((label = first_label; label <= last_label; label++)); do
  for ((iteration = 1; iteration <= iterations; iteration++)); do
    log_file="$log_dir/label-${label}_iteration-${iteration}.log"
    echo "[$(date '+%F %T')] label=$label iteration=$iteration/$iterations"

    if python main_pipeline.py \
      --classifier multiclass \
      --labels "$label" \
      --limit 1 \
      --pairs 1 \
      --ollama-model deepseek-r1:8b \
      --openai-model gpt-5.2 \
      --skip-retrained \
      --skip-human-evaluation \
      >"$log_file" 2>&1; then
      echo "  completed (log: $log_file)"
    else
      echo "  FAILED (log: $log_file)" >&2
      ((failures++))
    fi
  done
done

echo "Finished $(( (last_label - first_label + 1) * iterations )) runs; failures: $failures"
exit "$failures"
