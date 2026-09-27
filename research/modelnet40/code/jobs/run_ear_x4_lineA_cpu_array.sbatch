#!/bin/bash
#SBATCH --job-name=ear_x4_a
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=12:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/ear_x4_generation/ear_x4_a_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/ear_x4_generation/ear_x4_a_%A_%a.err

# EAR ×4 Line A: original 1024 → 4096
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
NUM_CHUNKS=16
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

echo "=== EAR ×4 Line A chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
cd "${PROJECT_ROOT}"

python scripts/run_ear_x4_chunk.py \
  --line A \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}" \
  --skip-existing

echo "=== Chunk ${CHUNK_ID} finished: $(date) ==="
