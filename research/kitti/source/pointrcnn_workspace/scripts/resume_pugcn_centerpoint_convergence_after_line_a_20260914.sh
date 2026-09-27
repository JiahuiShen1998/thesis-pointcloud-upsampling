#!/usr/bin/env bash
# Resume the 2026-09-14 convergence run after Line A training completed.
# This script deliberately never calls Line A training, so its 12 checkpoints
# remain immutable while validation and the Line B arm continue.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CENTER_PY=/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python
REPORT_PY=/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python
RUN_ROOT="$ROOT/results/pugcn_detector_convergence_20260914"
TRAIN_ROOT="$RUN_ROOT/centerpoint"
EVAL_ROOT="$RUN_ROOT/evaluations/centerpoint"
OLD_EXP="$ROOT/results/pugcn_full_retrain_20260824"
FULL_VAL="$ROOT/results/pugcn_detector_adaptation_full_val_20260908"
TRAIN_SPLIT="$ROOT/data/KITTI/ImageSets/train.txt"
TRAIN_INFOS="$ROOT/external/OpenPCDet/data/kitti/kitti_infos_train.pkl"
INIT="$ROOT/external/centerpoint_hpc_bundle_20260717/outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth"
MAX_EPOCHS="${MAX_EPOCHS:-12}"

A_VAL="$FULL_VAL/inputs/line_a_pugcn_observed_first"
B_TRAIN="$OLD_EXP/centerpoint_inputs/line_b_pugcn_observed_first"
B_VAL="$FULL_VAL/inputs/line_b_pugcn_observed_first"
A_PROTOCOL="$FULL_VAL/generation/val3769_linea/pu_gcn_linea_surface_pr1_c32/protocol.json"
B_PROTOCOL="$FULL_VAL/generation/val3769_lineb/pu_gcn_lineb_c2048_r4/protocol.json"

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT" "$RUN_ROOT/reports"

eval_arm() {
  local arm="$1" line="$2" lidar="$3" protocol="$4"
  local epoch checkpoint name
  for epoch in $(seq 1 "$MAX_EPOCHS"); do
    checkpoint="$TRAIN_ROOT/$arm/openpcdet_output/ckpt/checkpoint_epoch_${epoch}.pth"
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      return 1
    fi
    name="${arm}_epoch_${epoch}"
    CUDA_VISIBLE_DEVICES=0 "$CENTER_PY" "$ROOT/scripts/run_patch_causal_centerpoint_eval.py" \
      --name "$name" --line "$line" --source-mode direct --predicted-dir "$lidar" \
      --protocol-json "$protocol" --result-root "$EVAL_ROOT" \
      --checkpoint "$checkpoint" --batch-size 4 --workers 4 --resume
  done
}

# Phase 1: evaluate the already completed Line A checkpoints.
eval_arm line_a_pugcn_observed_first line_a "$A_VAL" "$A_PROTOCOL"

# Phase 2: train Line B only if all 12 checkpoints are not already present.
line_b_ckpt_dir="$TRAIN_ROOT/line_b_pugcn_observed_first/openpcdet_output/ckpt"
line_b_ckpt_count=$(find "$line_b_ckpt_dir" -maxdepth 1 -name 'checkpoint_epoch_*.pth' -type f 2>/dev/null | wc -l || true)
if [[ "$line_b_ckpt_count" -eq "$MAX_EPOCHS" ]]; then
  echo "LINE_B_TRAIN_SKIP existing_checkpoints=$line_b_ckpt_count"
elif [[ "$line_b_ckpt_count" -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES=0 "$CENTER_PY" "$ROOT/scripts/run_centerpoint_full_train.py" \
    --lidar-dir "$B_TRAIN" --split-file "$TRAIN_SPLIT" \
    --output-dir "$TRAIN_ROOT/line_b_pugcn_observed_first/openpcdet_output" \
    --init-ckpt "$INIT" --tag pugcn_conv_20260914_line_b --epochs "$MAX_EPOCHS" \
    --workers 2 --learning-rate 0.0003 --ckpt-save-interval 1 \
    --max-ckpt-save-num "$MAX_EPOCHS"
else
  echo "partial Line B checkpoint set ($line_b_ckpt_count/$MAX_EPOCHS); refusing an ambiguous restart" >&2
  exit 2
fi

# Phase 3: evaluate every Line B checkpoint and generate the evidence report.
eval_arm line_b_pugcn_observed_first line_b "$B_VAL" "$B_PROTOCOL"

"$REPORT_PY" "$ROOT/scripts/summarize_pugcn_centerpoint_convergence_20260914.py" \
  --run-root "$RUN_ROOT" --expected-epochs "$MAX_EPOCHS" \
  --min-delta 0.2 --patience 3

echo "PU_GCN_CENTERPOINT_CONVERGENCE_STAGE_PASS result=$RUN_ROOT"
