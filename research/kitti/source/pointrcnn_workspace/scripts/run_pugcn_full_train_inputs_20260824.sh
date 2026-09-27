#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv_pointrcnn/bin/python"
GENERATOR="$ROOT/scripts/run_patch_causal_upsampling.py"
EXPERIMENT_ROOT="$ROOT/results/pugcn_full_retrain_20260824/generation"
TRAIN_SPLIT="$ROOT/data/KITTI/ImageSets/train.txt"
LINE_A_INPUT="$ROOT/data/KITTI/object/training/velodyne_original"
LINE_B_INPUT="$ROOT/results/pugcn_full_retrain_20260824/inputs/velodyne_downsampled_x4_train"
NVIDIA_LIB=/tmp/nvidia61043_root/usr/lib64

common_args=(
  --experiment-root "$EXPERIMENT_ROOT"
  --frames-file "$TRAIN_SPLIT"
  --method pu_gcn
  --patch-selection fps_ball_cover_knn_v3
  --patch-num-ratio 1
  --min-ball-points 2048
  --cover-min-points 32
  --cover-radius-m 6
  --normalization-mode legacy_none
  --extract-workers "${EXTRACT_WORKERS:-8}"
  --drop-merged-after-final
  --cuda-visible-devices 0
  --gpu-memory-fraction 0.45
  --min-free-gib 120
)

case "${1:-}" in
  line_a)
    exec env LD_LIBRARY_PATH="$NVIDIA_LIB" "$PY" "$GENERATOR" \
      "${common_args[@]}" \
      --run-kind train3712_linea \
      --variant-name pu_gcn_linea_surface_pr1_c32 \
      --ball-radius-m 2 \
      --lines line_a_original_x4_up \
      --line-a-input "$LINE_A_INPUT"
    ;;
  line_b)
    exec env LD_LIBRARY_PATH="$NVIDIA_LIB" "$PY" "$GENERATOR" \
      "${common_args[@]}" \
      --run-kind train3712_lineb \
      --variant-name pu_gcn_lineb_c2048_r4 \
      --ball-radius-m 4 \
      --lines line_b_downsampled_x4_up \
      --line-b-input "$LINE_B_INPUT"
    ;;
  *)
    echo "Usage: $0 {line_a|line_b}" >&2
    exit 2
    ;;
esac
