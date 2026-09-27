#!/bin/bash
#SBATCH --job-name=pn2_lineB_downsampled_x4_baseline_256
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_full/lineB_downsampled_x4_baseline_256/slurm_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_full/lineB_downsampled_x4_baseline_256/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
BRANCH_NAME="lineB_downsampled_x4_baseline"
INPUT_PATH="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_baseline"
OUTPUT_DIR="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/lineB_downsampled_x4_baseline"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "${OUTPUT_DIR}" "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_full/lineB_downsampled_x4_baseline_256"

echo "=== PointNet++ full training job ==="
echo "branch=${BRANCH_NAME}"
echo "input_path=${INPUT_PATH}"
echo "expected_point_count=256"
echo "output_dir=${OUTPUT_DIR}"
echo "start=$(date)"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
pip install --user -q tqdm

cd "${PROJECT_ROOT}"
python scripts/train_pointnet2.py \
  --variant "${BRANCH_NAME}" \
  --data-root "${INPUT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-file "${LOG_FILE}" \
  --reports-dir "${OUTPUT_DIR}" \
  --model pointnet2_cls_ssg \
  --num-category 40 \
  --num-point 256 \
  --no-allow-resample \
  --batch-size 24 \
  --epoch 200 \
  --learning-rate 0.001 \
  --decay-rate 1e-4 \
  --optimizer Adam \
  --seed 42 \
  --num-workers 4

echo "end=$(date)"
