#!/bin/bash
#SBATCH --job-name=ear_lineB_fin
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/ear_full_lineB/finalize_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/ear_full_lineB/finalize_%j.err

# Merge chunk audits, validate outputs, update report.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
ARRAY_JOB_ID="${ARRAY_JOB_ID:?ARRAY_JOB_ID must be set}"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12

cd "${PROJECT_ROOT}"

python scripts/merge_ear_full_lineB_audit.py --array-job-id "${ARRAY_JOB_ID}"
python scripts/check_ear_full_lineB_outputs.py

echo "=== Finalize finished: $(date) ==="
