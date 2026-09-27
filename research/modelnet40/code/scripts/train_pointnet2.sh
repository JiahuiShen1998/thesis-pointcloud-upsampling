#!/bin/bash
# Train PointNet++ on a ModelNet40 variant.
# Usage: ./train_pointnet2.sh [variant] [extra args...]
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
VARIANT="${1:-original_baseline}"
shift || true

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
pip install --user -q tqdm

DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_original"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${VARIANT}"
LOG_FILE="${PROJECT_ROOT}/logs/${VARIANT}.log"
NUM_POINT=1024
ALLOW_RESAMPLE="--allow-resample"
REPORTS_DIR="${PROJECT_ROOT}/reports"

case "${VARIANT}" in
  original_baseline)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_original"
    NUM_POINT=1024
    ALLOW_RESAMPLE="--allow-resample"
    ;;
  downsampled50_baseline)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_downsampled50"
    NUM_POINT=1024
    ALLOW_RESAMPLE="--allow-resample"
    ;;
  downsampled50_native512_pointnet2)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_downsampled50"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/downsampled50_native512_pointnet2"
    LOG_FILE="${PROJECT_ROOT}/logs/downsampled50_native512_pointnet2/train.log"
    REPORTS_DIR="${PROJECT_ROOT}/reports/downsampled50_native512_pointnet2"
    NUM_POINT=512
    ALLOW_RESAMPLE="--no-allow-resample"
    ;;
  downsampled50_ear_pointnet2)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_downsampled50_ear"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/step8_downsampled50_ear_pointnet2"
    LOG_FILE="${PROJECT_ROOT}/logs/step8_downsampled50_ear_pointnet2/train.log"
    REPORTS_DIR="${PROJECT_ROOT}/reports/step8_downsampled50_ear_pointnet2"
    NUM_POINT=1024
    ALLOW_RESAMPLE="--no-allow-resample"
    ;;
  original_ear_x4_pointnet2)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_original_up/ear_x4"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/original_ear_x4_pointnet2"
    LOG_FILE="${PROJECT_ROOT}/logs/original_ear_x4_pointnet2/train.log"
    REPORTS_DIR="${PROJECT_ROOT}/reports/original_ear_x4_pointnet2"
    NUM_POINT=4096
    ALLOW_RESAMPLE="--no-allow-resample"
    ;;
  downsampled50_ear_x4_pointnet2)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_downsampled50_up/ear_x4"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/downsampled50_ear_x4_pointnet2"
    LOG_FILE="${PROJECT_ROOT}/logs/downsampled50_ear_x4_pointnet2/train.log"
    REPORTS_DIR="${PROJECT_ROOT}/reports/downsampled50_ear_x4_pointnet2"
    NUM_POINT=2048
    ALLOW_RESAMPLE="--no-allow-resample"
    ;;
  original_pdans_x4_pointnet2)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_original_up/pdans_x4"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/original_pdans_x4_pointnet2"
    LOG_FILE="${PROJECT_ROOT}/logs/original_pdans_x4_pointnet2/train.log"
    REPORTS_DIR="${PROJECT_ROOT}/reports/original_pdans_x4_pointnet2"
    NUM_POINT=4096
    ALLOW_RESAMPLE="--no-allow-resample"
    ;;
  downsampled50_pdans_x4_pointnet2)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_downsampled50_up/pdans_x4"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/downsampled50_pdans_x4_pointnet2"
    LOG_FILE="${PROJECT_ROOT}/logs/downsampled50_pdans_x4_pointnet2/train.log"
    REPORTS_DIR="${PROJECT_ROOT}/reports/downsampled50_pdans_x4_pointnet2"
    NUM_POINT=2048
    ALLOW_RESAMPLE="--no-allow-resample"
    ;;
  original_punet_x4_pointnet2)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_original_up/punet_x4"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/original_punet_x4_pointnet2"
    LOG_FILE="${PROJECT_ROOT}/logs/original_punet_x4_pointnet2/train.log"
    REPORTS_DIR="${PROJECT_ROOT}/reports/original_punet_x4_pointnet2"
    NUM_POINT=4096
    ALLOW_RESAMPLE="--no-allow-resample"
    ;;
  downsampled50_punet_x4_pointnet2)
    DATA_ROOT="${PROJECT_ROOT}/datasets/modelnet40_downsampled50_up/punet_x4"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/downsampled50_punet_x4_pointnet2"
    LOG_FILE="${PROJECT_ROOT}/logs/downsampled50_punet_x4_pointnet2/train.log"
    REPORTS_DIR="${PROJECT_ROOT}/reports/downsampled50_punet_x4_pointnet2"
    NUM_POINT=2048
    ALLOW_RESAMPLE="--no-allow-resample"
    ;;
  original_*|downsampled50_*)
    if [[ -n "${DATA_ROOT_OVERRIDE:-}" ]]; then
      DATA_ROOT="${DATA_ROOT_OVERRIDE}"
    fi
    ;;
esac

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")" "${REPORTS_DIR}"

python "${PROJECT_ROOT}/scripts/train_pointnet2.py" \
  --variant "${VARIANT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-file "${LOG_FILE}" \
  --reports-dir "${REPORTS_DIR}" \
  --model pointnet2_cls_ssg \
  --num-category 40 \
  --num-point "${NUM_POINT}" \
  ${ALLOW_RESAMPLE} \
  --batch-size 24 \
  --epoch 200 \
  --learning-rate 0.001 \
  --decay-rate 1e-4 \
  --optimizer Adam \
  --seed 42 \
  --num-workers 4 \
  "$@"
