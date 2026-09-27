#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python
RESULT_ROOT="$ROOT/results/pugcn_full_retrain_20260824/evaluations/centerpoint"
PUGCN_A="$ROOT/results/kitti_patch_punet_causal_ablation_v1_20260731/surface_candidate256_v1/pu_gcn_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin"
PUGCN_B="$ROOT/results/kitti_patch_punet_causal_ablation_v1_20260731/c2048_r4_pilot256_v1/pu_gcn_c2048_r4/line_b_downsampled_x4_up/final_bin"

run_eval() {
  local arm=$1
  local line=$2
  local mode=$3
  local checkpoint="$ROOT/results/pugcn_full_retrain_20260824/centerpoint/$arm/openpcdet_output/ckpt/checkpoint_epoch_3.pth"
  shift 3
  env CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$ROOT/scripts/run_patch_causal_centerpoint_eval.py" \
    --name "$arm" --line "$line" --source-mode "$mode" \
    --checkpoint "$checkpoint" --result-root "$RESULT_ROOT" \
    --batch-size 1 --workers 0 --resume "$@"
}

run_eval line_a_pugcn line_a upsampled_observed_first \
  --predicted-dir "$PUGCN_A" --reference-token original_x4_pu_gcn
run_eval line_b_pugcn line_b upsampled_observed_first \
  --predicted-dir "$PUGCN_B" --reference-token downsampled_x4_pu_gcn
run_eval line_b_baseline line_b baseline

echo "CENTERPOINT_FULL_EVAL_PASS"
