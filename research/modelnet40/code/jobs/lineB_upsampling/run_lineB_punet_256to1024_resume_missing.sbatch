#!/bin/bash
#SBATCH --job-name=lineB_x4_pu_net_resume
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_upsampling_resume/pu_net_resume_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_upsampling_resume/pu_net_resume_%A_%a.err

# Resume PU-Net Line B: only missing samples (3161), skip existing 9150
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
NUM_CHUNKS=8
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"
MISSING_LIST="${PROJECT_ROOT}/reports/modelnet40_lineB_punet_missing_samples.txt"

echo "=== PU-Net resume chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
mkdir -p "${PROJECT_ROOT}/logs/lineB_upsampling_resume"
cd "${PROJECT_ROOT}"

source scripts/activate_upsampling_env.sh punet

PUNET_CODE="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-Net/code"
cd "${PUNET_CODE}"
if ! python -c "from tf_ops.sampling.tf_sampling import farthest_point_sample" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh punet
fi
cd "${PROJECT_ROOT}"

python scripts/run_lineB_punet_resume_missing.py \
  --missing-list "${MISSING_LIST}" \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}"

echo "=== Resume chunk ${CHUNK_ID} finished: $(date) ==="
