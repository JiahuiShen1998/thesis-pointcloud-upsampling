#!/bin/bash
#SBATCH --job-name=mn40_B_pugcn_256to1024
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_%A_%a.err

# Line B PU-GCN full: 256 -> 1024, 12311 samples, resume/skip-existing
# Submit ONLY after Line B smoke PASS.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
PUGCN_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
CHECKPOINT="${PUGCN_ROOT}/pretrained/pu1k-pugcn"
NUM_CHUNKS=16
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

cd "${PROJECT_ROOT}"
mkdir -p logs/pugcn_x4 datasets/lineB_downsampled_x4_up/raw/pu_gcn datasets/lineB_downsampled_x4_up/strict_N/pu_gcn

echo "=== Line B PU-GCN full chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
nvidia-smi || true
echo "checkpoint: ${CHECKPOINT}"
echo "input: ${PROJECT_ROOT}/datasets/modelnet40_downsampled_x4"
echo "raw output: ${PROJECT_ROOT}/datasets/lineB_downsampled_x4_up/raw/pu_gcn"
echo "strict output: ${PROJECT_ROOT}/datasets/lineB_downsampled_x4_up/strict_N/pu_gcn"

source scripts/activate_upsampling_env.sh pugcn

cd "${PUGCN_ROOT}"
if ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh pugcn
fi
cd "${PROJECT_ROOT}"

python scripts/run_pugcn_modelnet40_x4.py \
  --line lineB \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}" \
  --resume \
  --seed 42 \
  --checkpoint "${CHECKPOINT}" \
  --failed-csv "logs/pugcn_x4/lineB_chunk_${CHUNK_ID}_failed.csv"

echo "=== Line B chunk ${CHUNK_ID} finished: $(date) ==="
find datasets/lineB_downsampled_x4_up/strict_N/pu_gcn -name "*.npy" 2>/dev/null | wc -l
