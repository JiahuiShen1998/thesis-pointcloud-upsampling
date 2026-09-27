#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python
SPLIT="$ROOT/data/KITTI/ImageSets/train.txt"
ORIGINAL="$ROOT/data/KITTI/object/training/velodyne_original"
DOWNSAMPLED="$ROOT/results/pugcn_full_retrain_20260824/inputs/velodyne_downsampled_x4_train"
PUGCN_A="$ROOT/results/pugcn_full_retrain_20260824/generation/train3712_linea/pu_gcn_linea_surface_pr1_c32/line_a_original_x4_up/final_bin"
PUGCN_B="$ROOT/results/pugcn_full_retrain_20260824/generation/train3712_lineb/pu_gcn_lineb_c2048_r4/line_b_downsampled_x4_up/final_bin"
INIT="$ROOT/external/centerpoint_hpc_bundle_20260717/outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth"
RUN_ROOT="$ROOT/results/pugcn_full_retrain_20260824/centerpoint"
INPUT_ROOT="$ROOT/results/pugcn_full_retrain_20260824/centerpoint_inputs"
TRAIN_INFOS="$ROOT/external/OpenPCDet/data/kitti/kitti_infos_train.pkl"

count_bins() {
  find "$1" -maxdepth 1 -type f -name '*.bin' | wc -l
}

"$PY" "$ROOT/scripts/prepare_centerpoint_train_infos.py" \
  --split-file "$SPLIT" --output "$TRAIN_INFOS" --workers 8

case "${1:-}" in
  line_a_pugcn)
    LIDAR="$INPUT_ROOT/line_a_pugcn_observed_first"
    if [[ "$(count_bins "$LIDAR")" -ne 3712 ]]; then
      "$PY" "$ROOT/scripts/prepare_centerpoint_observed_first_train.py" \
        --split-file "$SPLIT" --observed-dir "$ORIGINAL" --predicted-dir "$PUGCN_A" \
        --output-dir "$LIDAR" --reference-token original_x4_pu_gcn \
        --manifest "$INPUT_ROOT/line_a_pugcn_manifest.csv"
    fi
    ;;
  line_b_pugcn)
    LIDAR="$INPUT_ROOT/line_b_pugcn_observed_first"
    if [[ "$(count_bins "$LIDAR")" -ne 3712 ]]; then
      "$PY" "$ROOT/scripts/prepare_centerpoint_observed_first_train.py" \
        --split-file "$SPLIT" --observed-dir "$DOWNSAMPLED" --predicted-dir "$PUGCN_B" \
        --output-dir "$LIDAR" --reference-token downsampled_x4_pu_gcn \
        --manifest "$INPUT_ROOT/line_b_pugcn_manifest.csv"
    fi
    ;;
  line_b_baseline)
    LIDAR="$DOWNSAMPLED"
    ;;
  *)
    echo "Usage: $0 {line_a_pugcn|line_b_pugcn|line_b_baseline}" >&2
    exit 2
    ;;
esac

OUTPUT="$RUN_ROOT/$1/openpcdet_output"
CKPT="$OUTPUT/ckpt/checkpoint_epoch_3.pth"
if [[ ! -f "$CKPT" ]]; then
  env CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$ROOT/scripts/run_centerpoint_full_train.py" \
    --lidar-dir "$LIDAR" --split-file "$SPLIT" --output-dir "$OUTPUT" \
    --init-ckpt "$INIT" --tag "full_adapt_$1" --epochs 3 --workers 0 \
    --learning-rate 0.0003
fi

echo "CENTERPOINT_FULL_ADAPTATION_PASS arm=$1 checkpoint=$CKPT"
