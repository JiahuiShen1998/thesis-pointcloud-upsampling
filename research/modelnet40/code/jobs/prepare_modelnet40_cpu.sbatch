#!/bin/bash
#SBATCH --job-name=mn40_prepare
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/prepare_modelnet40_%j.log
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/prepare_modelnet40_%j.err

# CPU-only ModelNet40 preprocessing.
# NOTE: TinyGPU policy requires --gres=gpu:1 even though this job does not use the GPU.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
SCRIPT="${PROJECT_ROOT}/scripts/prepare_modelnet40.py"
CHECK_SCRIPT="${PROJECT_ROOT}/scripts/check_modelnet40_original.py"

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"

module load python/3.12-conda 2>/dev/null || module load python/pytorch2.6py3.12
python --version

cd "${PROJECT_ROOT}"

python "${SCRIPT}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --seed 42 \
  --num-points 1024 \
  --overwrite \
  2>&1 | tee "${PROJECT_ROOT}/logs/prepare_modelnet40_latest.log"

python "${CHECK_SCRIPT}"

echo "=== Job finished: $(date) ==="
