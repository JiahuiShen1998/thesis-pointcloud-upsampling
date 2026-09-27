#!/bin/bash
# Evaluate PointNet++ checkpoint on a ModelNet40 variant.
# Usage: ./eval_pointnet2.sh <variant> [data_root_override]
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
VARIANT="${1:?variant required}"
DATA_ROOT="${2:-${PROJECT_ROOT}/datasets/modelnet40_original}"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
pip install --user -q tqdm

CHECKPOINT="${PROJECT_ROOT}/outputs/${VARIANT}/checkpoints/best_model.pth"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${VARIANT}"
LOG_FILE="${PROJECT_ROOT}/logs/${VARIANT}_eval.log"

python "${PROJECT_ROOT}/scripts/eval_pointnet2.py" \
  --variant "${VARIANT}" \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-file "${LOG_FILE}" \
  --model pointnet2_cls_ssg \
  --num-category 40 \
  --num-point 1024 \
  --batch-size 24 \
  --num-workers 4
