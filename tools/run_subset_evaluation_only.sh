#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/run_subset_evaluation_only.sh [judge_model] [judge_api_key]
#
# Example:
#   bash tools/run_subset_evaluation_only.sh \
#     "gpt-5.1-2025-11-13" \
#     "$OPENAI_API_KEY"

JUDGE_MODEL="${1:-gpt-5.1-2025-11-13}"
JUDGE_API_KEY="${2:-${OPENAI_API_KEY:-}}"

if [[ -z "$JUDGE_API_KEY" ]]; then
  echo "Missing judge API key. Pass it as 2nd arg or set OPENAI_API_KEY."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SUBSET_DIR="results/subset_backbone_sensitivity"
CASE_IDS_FILE="$SUBSET_DIR/case_ids_n30_seed57.txt"
OUT_DIR="results/backbone_gemma4_subset30_sectioned_direct_cap"

if [[ ! -f "$CASE_IDS_FILE" ]]; then
  echo "Missing subset file: $CASE_IDS_FILE"
  exit 1
fi

CASE_IDS=($(cat "$CASE_IDS_FILE"))

python3 code/run_template_rendering_experiments.py \
  --output-dir "$OUT_DIR" \
  --methods direct cap \
  --templates sectioned \
  --case-ids "${CASE_IDS[@]}" \
  --model "google/gemma-4-31b-it" \
  --judge-model "$JUDGE_MODEL" \
  --judge-api-key "$JUDGE_API_KEY" \
  --request-timeout 240 \
  --task-workers 1 \
  --case-workers 1 \
  --evaluation-only \
  --write-case-files

python3 tools/summarize_latency_cost.py \
  --case-metrics "$OUT_DIR/case_metrics.csv" \
  --case-ids-file "$CASE_IDS_FILE" \
  --template sectioned \
  --methods direct cap \
  --output-csv "$OUT_DIR/latency_cost_summary.csv"

echo "[DONE] evaluation-only run complete: $OUT_DIR"
echo "[DONE] latency summary: $OUT_DIR/latency_cost_summary.csv"

