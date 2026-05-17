#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/run_subset_gemma4_generation_only.sh <runpod_base_url> [api_key]
#
# Example:
#   bash tools/run_subset_gemma4_generation_only.sh \
#     "https://<pod>-40080.proxy.runpod.net/v1" \
#     "dummy"

BASE_URL="${1:-}"
API_KEY="${2:-dummy}"

if [[ -z "$BASE_URL" ]]; then
  echo "Missing BASE_URL"
  exit 1
fi

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

OUT_DIR="results/backbone_gemma4_subset30_sectioned_direct_cap_capevent"
mkdir -p "$OUT_DIR"

python3 code/run_template_rendering_experiments.py \
  --output-dir "$OUT_DIR" \
  --methods direct cap cap_event \
  --templates sectioned \
  --case-ids "${CASE_IDS[@]}" \
  --model "google/gemma-4-31b-it" \
  --api-base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --request-timeout 240 \
  --temperature 0 \
  --task-workers 1 \
  --case-workers 1 \
  --generation-only \
  --write-case-files

echo "[DONE] generation-only run complete: $OUT_DIR"
echo "[NEXT] run evaluation later with tools/run_subset_evaluation_only.sh"
