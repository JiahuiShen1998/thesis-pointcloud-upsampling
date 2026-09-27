#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_OLD=/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python
PY_CP=/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python
EXP="$ROOT/results/pugcn_full_retrain_20260824"
LINE_A="$EXP/generation/train3712_linea/pu_gcn_linea_surface_pr1_c32/line_a_original_x4_up/final_bin"
LINE_B="$EXP/generation/train3712_lineb/pu_gcn_lineb_c2048_r4/line_b_downsampled_x4_up/final_bin"

count_bins() {
  find "$1" -maxdepth 1 -type f -name '*.bin' | wc -l
}

if [[ "$(count_bins "$LINE_A")" -ne 3712 ]]; then
  echo "Line A generation is incomplete" >&2
  exit 1
fi
if [[ "$(count_bins "$LINE_B")" -ne 3712 ]]; then
  echo "Line B generation is incomplete" >&2
  exit 1
fi

# PointRCNN is first because it is the primary detector in the current study.
"$ROOT/scripts/run_pointrcnn_full_adaptation_20260824.sh" line_a_pugcn
"$ROOT/scripts/run_pointrcnn_full_adaptation_20260824.sh" line_b_pugcn
"$ROOT/scripts/run_pointrcnn_full_adaptation_20260824.sh" line_b_baseline

"$ROOT/scripts/run_centerpoint_full_adaptation_20260824.sh" line_a_pugcn
"$ROOT/scripts/run_centerpoint_full_adaptation_20260824.sh" line_b_pugcn
"$ROOT/scripts/run_centerpoint_full_adaptation_20260824.sh" line_b_baseline

"$PY_OLD" "$ROOT/scripts/run_pointrcnn_full_adaptation_eval_20260824.py"
"$ROOT/scripts/run_centerpoint_full_adaptation_eval_20260824.sh"
"$PY_CP" "$ROOT/scripts/summarize_pugcn_full_retraining_20260824.py"

echo "PU_GCN_FULL_DETECTOR_MATRIX_PASS"
