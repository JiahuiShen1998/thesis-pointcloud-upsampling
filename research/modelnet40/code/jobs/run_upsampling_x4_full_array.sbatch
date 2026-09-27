#!/bin/bash
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/%x_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/%x_%A_%a.err

# Parameterized ×4 full generation array job.
# Submit example:
#   sbatch --job-name=punet_x4_a --export=ALL,METHOD=punet,LINE=A jobs/run_upsampling_x4_full_array.sbatch
set -euo pipefail

: "${METHOD:?METHOD must be set (pdans|punet|pugcn)}"
: "${LINE:?LINE must be set (A|B)}"

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
METHOD="$("${PROJECT_ROOT}/scripts/validate_upsampling_method.sh" "${METHOD}")"
echo "Validated METHOD=${METHOD} LINE=${LINE}"
NUM_CHUNKS=16
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

echo "=== ${METHOD} ×4 line ${LINE} chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
cd "${PROJECT_ROOT}"

source scripts/activate_upsampling_env.sh "${METHOD}"

python scripts/run_upsampling_x4_chunk.py \
  --method "${METHOD}" \
  --line "${LINE}" \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}" \
  --skip-existing

echo "=== Chunk ${CHUNK_ID} finished: $(date) ==="
