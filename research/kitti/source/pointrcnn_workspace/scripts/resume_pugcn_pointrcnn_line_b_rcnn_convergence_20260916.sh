#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python
RUN_ROOT="$ROOT/results/pugcn_detector_convergence_e18_20260915"
ARM=line_b_pugcn_observed_first
ARM_ROOT="$RUN_ROOT/pointrcnn/$ARM"
EVAL_ROOT="$RUN_ROOT/evaluations/pointrcnn"
REPORT_ROOT="$RUN_ROOT/reports"
VAL_LIDAR="$ROOT/results/pugcn_detector_adaptation_full_val_20260908/inputs/$ARM"
VAL_SPLIT="$ROOT/data/KITTI/ImageSets/val.txt"
OFFICIAL="$ROOT/tools/PointRCNN.pth"
RPN_EPOCH=14
RPN_CKPT="$ARM_ROOT/rpn/ckpt/checkpoint_epoch_${RPN_EPOCH}.pth"
RCNN_DIR="$ARM_ROOT/rcnn_from_rpn_epoch_${RPN_EPOCH}"
COMBINED_DIR="$ARM_ROOT/combined_rcnn_curve_rpn_epoch_${RPN_EPOCH}"

mkdir -p "$COMBINED_DIR" "$EVAL_ROOT" "$REPORT_ROOT"

for epoch in $(seq 1 18); do
  rcnn_ckpt="$RCNN_DIR/ckpt/checkpoint_epoch_${epoch}.pth"
  combined="$COMBINED_DIR/checkpoint_epoch_${epoch}.pth"
  output_dir="$EVAL_ROOT/${ARM}_rcnn_epoch_${epoch}"
  marker="$output_dir/run_complete.json"
  log="$output_dir/runner_stdout.log"

  [[ -s "$rcnn_ckpt" ]] || {
    echo "MISSING_RCNN_CHECKPOINT epoch=$epoch path=$rcnn_ckpt" >&2
    exit 91
  }

  if [[ ! -s "$combined" ]]; then
    "$PY" "$ROOT/scripts/combine_pointrcnn_rpn_with_official_rcnn.py" \
      --official "$OFFICIAL" \
      --rpn "$RPN_CKPT" \
      --rcnn "$rcnn_ckpt" \
      --output "$combined"
  fi

  if [[ -f "$marker" ]] && jq -e \
      --arg checkpoint "$combined" \
      '.status == "PASS" and .frame_count == 3769 and
       .actual_consumed_frame_count == 3769 and
       .checkpoint == $checkpoint and
       .heavy_artifacts_retained == false' \
      "$marker" >/dev/null; then
    ap="$(jq -r '.metrics_percent.Car."3d_ap_r40".moderate' "$marker")"
    echo "EVAL_SKIP_PASS epoch=$epoch car_3d_moderate=$ap"
    continue
  fi

  mkdir -p "$output_dir"
  echo "EVAL_START epoch=$epoch checkpoint=$combined"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/run_pointrcnn_checkpoint_split_eval.py" \
    --lidar-dir "$VAL_LIDAR" \
    --checkpoint "$combined" \
    --split-file "$VAL_SPLIT" \
    --split-name val \
    --output-dir "$output_dir" \
    --batch-size 1 \
    --workers 0 \
    --seed 20260908 \
    --resume \
    --cleanup-heavy-artifacts >> "$log" 2>&1

  jq -e \
    --arg checkpoint "$combined" \
    '.status == "PASS" and .frame_count == 3769 and
     .actual_consumed_frame_count == 3769 and
     .checkpoint == $checkpoint and
     .heavy_artifacts_retained == false' \
    "$marker" >/dev/null
  ap="$(jq -r '.metrics_percent.Car."3d_ap_r40".moderate' "$marker")"
  echo "EVAL_PASS epoch=$epoch car_3d_moderate=$ap"
done

"$PY" "$ROOT/scripts/summarize_pugcn_pointrcnn_convergence_20260914.py" stage \
  --run-root "$RUN_ROOT" \
  --arm "$ARM" \
  --phase rcnn \
  --expected-epochs 18 \
  --min-delta 0.2 \
  --patience 3 > "$REPORT_ROOT/pointrcnn_${ARM}_rcnn_summary_stdout.json"

jq -e '.status == "CONVERGED"' \
  "$REPORT_ROOT/pointrcnn_${ARM}_rcnn_convergence.json" >/dev/null

"$PY" "$ROOT/scripts/summarize_pugcn_pointrcnn_convergence_20260914.py" aggregate \
  --run-root "$RUN_ROOT" \
  --expected-epochs 18 > "$REPORT_ROOT/pointrcnn_convergence_summary_stdout.json"

jq -e '.status == "CONVERGED"' \
  "$REPORT_ROOT/pointrcnn_convergence_summary.json" >/dev/null

echo "PU_GCN_POINTRCNN_LINE_B_RCNN_CONVERGENCE_PASS result=$RUN_ROOT"
