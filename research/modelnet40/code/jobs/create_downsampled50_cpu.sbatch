#!/bin/bash
#SBATCH --job-name=mn40_down50
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/create_downsampled50_%j.log
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/create_downsampled50_%j.err

# CPU-only downsampling. TinyGPU requires --gres=gpu:1 but this job does not train.
# NOTE: Do NOT use this if you want to avoid occupying a GPU slot while Step 4 runs.
# Prefer login-node execution for this lightweight task (~1-2 min).
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"

module purge 2>/dev/null || true
module load python/3.12-conda 2>/dev/null || module load python/pytorch2.6py3.12

cd "${PROJECT_ROOT}"

python scripts/create_downsampled50.py --overwrite --num-workers 8
python scripts/check_downsampled50.py

echo "Step 5 finished: $(date)"
