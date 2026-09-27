#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PU_NET_FORCE_CPU || true

exec /home/ra87racy/miniconda3/envs/punet_tf/bin/python \
  /home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/punet_tf_gpu_bootstrap.py \
  "$@"
