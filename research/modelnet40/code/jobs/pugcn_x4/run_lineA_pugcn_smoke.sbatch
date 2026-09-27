#!/bin/bash
#SBATCH --job-name=mn40_A_pugcn_smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineA_pugcn_smoke_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineA_pugcn_smoke_%j.err

# Line A PU-GCN smoke: 1024 -> 4096 (10 samples)
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
PUGCN_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
CHECKPOINT="${PUGCN_ROOT}/pretrained/pu1k-pugcn"

cd "${PROJECT_ROOT}"
mkdir -p logs/pugcn_x4 datasets/lineA_original_up/raw/pu_gcn datasets/lineA_original_up/strict_4N/pu_gcn

echo "=== Line A PU-GCN smoke started: $(date) ==="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
nvidia-smi || true
echo "checkpoint: ${CHECKPOINT}"
echo "input: ${PROJECT_ROOT}/datasets/modelnet40_original"
echo "raw output: ${PROJECT_ROOT}/datasets/lineA_original_up/raw/pu_gcn"
echo "strict output: ${PROJECT_ROOT}/datasets/lineA_original_up/strict_4N/pu_gcn"

source scripts/activate_upsampling_env.sh pugcn
bash scripts/print_gpu_cuda_diagnostics.sh

cd "${PUGCN_ROOT}"
if ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh pugcn
fi
cd "${PROJECT_ROOT}"

python scripts/run_pugcn_modelnet40_x4.py \
  --line lineA \
  --max-per-split 5 \
  --seed 42 \
  --checkpoint "${CHECKPOINT}" \
  --failed-csv logs/pugcn_x4/lineA_smoke_failed.csv

python scripts/audit_pugcn_smoke.py --line lineA --max-samples 10

echo "=== Line A PU-GCN smoke finished: $(date) ==="
