#!/bin/bash
#SBATCH --job-name=lineB_x4_pu_net
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A_%a.err

# Line B PU-Net full: 256 -> 1024, 12311 samples, resume/skip-existing
# Submit ONLY after smoke PASS.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
NUM_CHUNKS=16
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

echo "=== Line B PU-Net full chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
cd "${PROJECT_ROOT}"
mkdir -p datasets/lineB_downsampled_x4_up/raw/pu_net datasets/lineB_downsampled_x4_up/strict_N/pu_net

source scripts/activate_upsampling_env.sh punet

PUNET_CODE="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-Net/code"
cd "${PUNET_CODE}"
if ! python -c "from tf_ops.sampling.tf_sampling import farthest_point_sample" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh punet
fi
cd "${PROJECT_ROOT}"

python scripts/run_lineB_downsampled_x4_upsampling_chunk.py \
  --method pu_net \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}" \
  --skip-existing

echo "=== Chunk ${CHUNK_ID} finished: $(date) ==="
