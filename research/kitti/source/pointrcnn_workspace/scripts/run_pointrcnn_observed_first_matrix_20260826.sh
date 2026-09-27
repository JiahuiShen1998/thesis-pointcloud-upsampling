#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python
EXP="$ROOT/results/pugcn_full_retrain_20260824"

count_bins() {
  find "$1" -maxdepth 1 -type f -name '*.bin' | wc -l
}

TRAIN_A="$EXP/centerpoint_inputs/line_a_pugcn_observed_first"
TRAIN_B="$EXP/centerpoint_inputs/line_b_pugcn_observed_first"
EVAL_A="$EXP/evaluations/centerpoint/line_a_pugcn/observed_first_input"
EVAL_B="$EXP/evaluations/centerpoint/line_b_pugcn/observed_first_input"

if [[ "$(count_bins "$TRAIN_A")" -ne 3712 || "$(count_bins "$TRAIN_B")" -ne 3712 ]]; then
  echo "Observed-first training inputs are incomplete" >&2
  exit 1
fi
if [[ "$(count_bins "$EVAL_A")" -ne 256 || "$(count_bins "$EVAL_B")" -ne 256 ]]; then
  echo "Observed-first evaluation inputs are incomplete" >&2
  exit 1
fi

"$ROOT/scripts/run_pointrcnn_full_adaptation_20260824.sh" line_a_pugcn_observed_first
"$ROOT/scripts/run_pointrcnn_full_adaptation_20260824.sh" line_b_pugcn_observed_first
"$PY" "$ROOT/scripts/run_pointrcnn_observed_first_eval_20260826.py"
"$PY" "$ROOT/scripts/summarize_pointrcnn_observed_first_20260826.py"

echo "POINT_RCNN_OBSERVED_FIRST_MATRIX_PASS"
