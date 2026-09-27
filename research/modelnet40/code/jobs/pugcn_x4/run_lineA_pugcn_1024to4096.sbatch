#!/bin/bash
#SBATCH --job-name=mn40_A_pugcn_1024to4096
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineA_pugcn_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineA_pugcn_%A_%a.err

# Line A PU-GCN full: 1024 -> 4096, 12311 samples, resume/skip-existing
# Submit ONLY after Line A smoke PASS.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
PUGCN_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
CHECKPOINT="${PUGCN_ROOT}/pretrained/pu1k-pugcn"
NUM_CHUNKS=16
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

cd "${PROJECT_ROOT}"
mkdir -p logs/pugcn_x4 datasets/lineA_original_up/raw/pu_gcn datasets/lineA_original_up/strict_4N/pu_gcn

echo "=== Line A PU-GCN full chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
nvidia-smi || true
echo "checkpoint: ${CHECKPOINT}"
echo "input: ${PROJECT_ROOT}/datasets/modelnet40_original"
echo "raw output: ${PROJECT_ROOT}/datasets/lineA_original_up/raw/pu_gcn"
echo "strict output: ${PROJECT_ROOT}/datasets/lineA_original_up/strict_4N/pu_gcn"

source scripts/activate_upsampling_env.sh pugcn

cd "${PUGCN_ROOT}"
if ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh pugcn
fi
cd "${PROJECT_ROOT}"

python scripts/run_pugcn_modelnet40_x4.py \
  --line lineA \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}" \
  --resume \
  --seed 42 \
  --checkpoint "${CHECKPOINT}" \
  --failed-csv "logs/pugcn_x4/lineA_chunk_${CHUNK_ID}_failed.csv"

echo "=== Line A chunk ${CHUNK_ID} finished: $(date) ==="
find datasets/lineA_original_up/strict_4N/pu_gcn -name "*.npy" 2>/dev/null | wc -l
