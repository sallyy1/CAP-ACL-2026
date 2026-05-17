#!/usr/bin/env bash
set -euo pipefail

# Goal:
# - Reproduce generation-only runs on the paper model (Gemma-3-4B-Instruct)
# - 30-case stratified subset
# - methods: direct, cap, cap_event
# - output latency + token usage summary
#
# Prereq:
# - .env has RUNPOD_POD_ID set (or pass OPENAI_BASE_URL explicitly before run)
#
# Usage:
#   bash tools/run_subset_gemma3_latency_tokens.sh [api_key]
#
# Example:
#   bash tools/run_subset_gemma3_latency_tokens.sh dummy

API_KEY="${1:-dummy}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SUBSET_DIR="results/subset_backbone_sensitivity"
mkdir -p "$SUBSET_DIR"

python3 tools/build_backbone_subset.py \
  --subset-size 30 \
  --seed 57 \
  --output-dir "$SUBSET_DIR"

CASE_IDS_FILE="$SUBSET_DIR/case_ids_n30_seed57.txt"
CASE_IDS=($(cat "$CASE_IDS_FILE"))

OUT_DIR="results/backbone_gemma3_subset30_sectioned_direct_cap_capevent"
PS_OUT_DIR="outputs/problem_state_tracking_subset30_seed57_gemma3"
mkdir -p "$OUT_DIR"

python3 code/run_problem_state_tracking_experiments.py \
  --cases-path outputs/shared/run_aci_all_v1_realistic/cases.jsonl \
  --output-dir "$PS_OUT_DIR" \
  --case-ids "${CASE_IDS[@]}" \
  --model "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g" \
  --extractor-model "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g" \
  --api-key "$API_KEY" \
  --temperature 0.0 \
  --request-timeout 240 \
  --deployment-cost-only

python3 code/run_template_rendering_experiments.py \
  --cases-path outputs/shared/run_aci_all_v1_realistic/cases.jsonl \
  --problem-state-dir "$PS_OUT_DIR" \
  --output-dir "$OUT_DIR" \
  --methods direct cap cap_event \
  --templates sectioned \
  --case-ids "${CASE_IDS[@]}" \
  --model "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g" \
  --api-key "$API_KEY" \
  --temperature 0.0 \
  --request-timeout 240 \
  --task-workers 1 \
  --case-workers 1 \
  --generation-only \
  --write-case-files

python3 tools/summarize_pipeline_latency_tokens.py \
  --render-case-metrics "$OUT_DIR/case_metrics.csv" \
  --problem-state-case-metrics "$PS_OUT_DIR/case_metrics.csv" \
  --case-ids-file "$CASE_IDS_FILE" \
  --template sectioned \
  --output-csv "$OUT_DIR/pipeline_latency_tokens_n30_seed57.csv"

python3 tools/export_pipeline_table_latex.py \
  --input-csv "$OUT_DIR/pipeline_latency_tokens_n30_seed57.csv" \
  --output-tex "$OUT_DIR/pipeline_latency_tokens_n30_seed57.tex" \
  --caption "Offline latency and token-cost proxy on a 30-case stratified subset (paper model, sectioned template)." \
  --label "tab:latency_token_subset_gemma3"

echo "[DONE] generation-only run complete: $OUT_DIR"
echo "[DONE] CSV: $OUT_DIR/pipeline_latency_tokens_n30_seed57.csv"
echo "[DONE] LaTeX: $OUT_DIR/pipeline_latency_tokens_n30_seed57.tex"
