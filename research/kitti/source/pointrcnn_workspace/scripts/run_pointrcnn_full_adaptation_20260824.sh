#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python
SPLIT="$ROOT/data/KITTI/ImageSets/train.txt"
OFFICIAL="$ROOT/tools/PointRCNN.pth"
RUN_ROOT="$ROOT/results/pugcn_full_retrain_20260824/pointrcnn"

case "${1:-}" in
  line_a_pugcn)
    LIDAR="$ROOT/results/pugcn_full_retrain_20260824/generation/train3712_linea/pu_gcn_linea_surface_pr1_c32/line_a_original_x4_up/final_bin"
    ;;
  line_b_pugcn)
    LIDAR="$ROOT/results/pugcn_full_retrain_20260824/generation/train3712_lineb/pu_gcn_lineb_c2048_r4/line_b_downsampled_x4_up/final_bin"
    ;;
  line_b_baseline)
    LIDAR="$ROOT/results/pugcn_full_retrain_20260824/inputs/velodyne_downsampled_x4_train"
    ;;
  line_a_pugcn_observed_first)
    LIDAR="$ROOT/results/pugcn_full_retrain_20260824/centerpoint_inputs/line_a_pugcn_observed_first"
    ;;
  line_b_pugcn_observed_first)
    LIDAR="$ROOT/results/pugcn_full_retrain_20260824/centerpoint_inputs/line_b_pugcn_observed_first"
    ;;
  *)
    echo "Usage: $0 {line_a_pugcn|line_b_pugcn|line_b_baseline|line_a_pugcn_observed_first|line_b_pugcn_observed_first}" >&2
    exit 2
    ;;
esac

ARM_ROOT="$RUN_ROOT/$1"
RPN_CKPT="$ARM_ROOT/rpn/ckpt/checkpoint_epoch_3.pth"
COMBINED="$ARM_ROOT/combined_rpn_rcnn_init.pth"
RCNN_CKPT="$ARM_ROOT/rcnn/ckpt/checkpoint_epoch_3.pth"
EXPORT_ROOT="$ARM_ROOT/rpn_export"
EXPORT_DATA="$EXPORT_ROOT/eval/epoch_3/train"
FEATURE_DIR="$EXPORT_DATA/features"
ROI_DIR="$EXPORT_DATA/detections/data"
EXPORT_MARKER="$EXPORT_ROOT/export_complete.json"
FINAL_CKPT="$ARM_ROOT/combined_rpn_rcnn_adapted.pth"

if [[ ! -f "$RPN_CKPT" ]]; then
  "$PY" "$ROOT/scripts/run_pointrcnn_full_train_stage.py" \
    --mode rpn --lidar-dir "$LIDAR" --split-file "$SPLIT" \
    --output-dir "$ARM_ROOT/rpn" --init-ckpt "$OFFICIAL" --epochs 3 --workers 0 --batch-size 1
fi

if [[ ! -f "$COMBINED" ]]; then
  "$PY" "$ROOT/scripts/combine_pointrcnn_rpn_with_official_rcnn.py" \
    --official "$OFFICIAL" --rpn "$RPN_CKPT" --output "$COMBINED"
fi

if [[ ! -f "$EXPORT_MARKER" ]]; then
  "$PY" "$ROOT/scripts/run_pointrcnn_rpn_feature_export.py" \
    --lidar-dir "$LIDAR" --split-file "$SPLIT" --rpn-ckpt "$RPN_CKPT" \
    --output-dir "$EXPORT_ROOT" --workers 0
fi

if [[ ! -f "$RCNN_CKPT" ]]; then
  "$PY" "$ROOT/scripts/run_pointrcnn_full_train_stage.py" \
    --mode rcnn_offline --lidar-dir "$LIDAR" --split-file "$SPLIT" \
    --output-dir "$ARM_ROOT/rcnn" --init-ckpt "$COMBINED" --epochs 3 --workers 0 --batch-size 1 \
    --rcnn-training-roi-dir "$ROI_DIR" --rcnn-training-feature-dir "$FEATURE_DIR"
fi

if [[ ! -f "$FINAL_CKPT" ]]; then
  "$PY" "$ROOT/scripts/combine_pointrcnn_rpn_with_official_rcnn.py" \
    --official "$OFFICIAL" --rpn "$RPN_CKPT" --rcnn "$RCNN_CKPT" --output "$FINAL_CKPT"
fi

echo "POINT_RCNN_FULL_ADAPTATION_PASS arm=$1 checkpoint=$FINAL_CKPT"
