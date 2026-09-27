#!/bin/bash
#SBATCH --job-name=pn2_mesh_ref_baseline_4096
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_full/mesh_ref_baseline_4096/slurm_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_full/mesh_ref_baseline_4096/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
BRANCH_NAME="mesh_ref_baseline_4096"
INPUT_PATH="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/mesh_ref_baseline_4096"
OUTPUT_DIR="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/mesh_ref_baseline_4096"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "${OUTPUT_DIR}" "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_full/mesh_ref_baseline_4096"

echo "=== PointNet++ mesh-ref baseline ==="
echo "branch=${BRANCH_NAME}"
echo "input_path=${INPUT_PATH}"
echo "expected_point_count=4096"
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
  --num-point 4096 \
  --no-allow-resample \
  --batch-size 24 \
  --epoch 200 \
  --learning-rate 0.001 \
  --decay-rate 1e-4 \
  --optimizer Adam \
  --seed 42 \
  --num-workers 4

echo "end=$(date)"
