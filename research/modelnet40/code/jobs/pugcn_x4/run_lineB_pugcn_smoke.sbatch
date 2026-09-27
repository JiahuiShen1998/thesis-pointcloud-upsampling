#!/bin/bash
#SBATCH --job-name=mn40_B_pugcn_smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_smoke_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_smoke_%j.err

# Line B PU-GCN smoke: 256 -> 1024 (10 samples)
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
PUGCN_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
CHECKPOINT="${PUGCN_ROOT}/pretrained/pu1k-pugcn"

cd "${PROJECT_ROOT}"
mkdir -p logs/pugcn_x4 datasets/lineB_downsampled_x4_up/raw/pu_gcn datasets/lineB_downsampled_x4_up/strict_N/pu_gcn

echo "=== Line B PU-GCN smoke started: $(date) ==="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
nvidia-smi || true
echo "checkpoint: ${CHECKPOINT}"
echo "input: ${PROJECT_ROOT}/datasets/modelnet40_downsampled_x4"
echo "raw output: ${PROJECT_ROOT}/datasets/lineB_downsampled_x4_up/raw/pu_gcn"
echo "strict output: ${PROJECT_ROOT}/datasets/lineB_downsampled_x4_up/strict_N/pu_gcn"

source scripts/activate_upsampling_env.sh pugcn
bash scripts/print_gpu_cuda_diagnostics.sh

cd "${PUGCN_ROOT}"
if ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh pugcn
fi
cd "${PROJECT_ROOT}"

python scripts/run_pugcn_modelnet40_x4.py \
  --line lineB \
  --max-per-split 5 \
  --seed 42 \
  --checkpoint "${CHECKPOINT}" \
  --failed-csv logs/pugcn_x4/lineB_smoke_failed.csv

python scripts/audit_pugcn_smoke.py --line lineB --max-samples 10

echo "=== Line B PU-GCN smoke finished: $(date) ==="
