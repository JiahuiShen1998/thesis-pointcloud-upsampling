#!/bin/bash
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A_%a.err

# Line B new protocol full generation: 256→1024
# Submit: sbatch --job-name=lineB_x4_pdans --export=ALL,METHOD=pdans jobs/run_lineB_downsampled_x4_full_array.sbatch
set -euo pipefail

: "${METHOD:?METHOD must be ear|pu_net|pdans}"

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
METHOD_KEY="${METHOD%\}}"
NUM_CHUNKS=16
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

echo "=== Line B x4 full METHOD=${METHOD_KEY} chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
cd "${PROJECT_ROOT}"
mkdir -p logs/lineB_downsampled_x4_up datasets/lineB_downsampled_x4_up/raw datasets/lineB_downsampled_x4_up/strict_N

if [[ "${METHOD_KEY}" == "ear" ]]; then
  module purge 2>/dev/null || true
  module load python/pytorch2.6py3.12
else
  ALIAS="$("${PROJECT_ROOT}/scripts/validate_upsampling_method.sh" "${METHOD_KEY/pu_net/punet}")"
  source scripts/activate_upsampling_env.sh "${ALIAS}"
  if [[ "${ALIAS}" == "punet" ]]; then
    ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-Net/code"
    cd "${ROOT}"
    if ! python -c "from tf_ops.sampling.tf_sampling import farthest_point_sample" 2>/dev/null; then
      cd "${PROJECT_ROOT}"
      bash scripts/recompile_tf_ops.sh punet
    fi
    cd "${PROJECT_ROOT}"
  fi
fi

python scripts/run_lineB_downsampled_x4_upsampling_chunk.py \
  --method "${METHOD_KEY}" \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}" \
  --skip-existing

echo "=== Chunk ${CHUNK_ID} finished: $(date) ==="
