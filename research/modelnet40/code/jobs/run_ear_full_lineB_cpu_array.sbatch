#!/bin/bash
#SBATCH --job-name=ear_lineB
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=06:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/ear_full_lineB/ear_chunk_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/ear_full_lineB/ear_chunk_%A_%a.err

# Step 7b: EAR full Line B (downsampled50 -> 1024) CPU upsampling.
# EAR uses numpy/sklearn only; GPU slot is required by work partition policy but unused.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
NUM_CHUNKS=16
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

echo "=== EAR Line B chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12

cd "${PROJECT_ROOT}"

python scripts/run_ear_full_lineB_chunk.py \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}" \
  --skip-existing

echo "=== Chunk ${CHUNK_ID} finished: $(date) ==="
