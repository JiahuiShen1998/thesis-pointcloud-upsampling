#!/bin/bash
#SBATCH --job-name=lineB_punet_smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A.err

# Line B PU-Net smoke only: 256 -> 1024 (fixed mkdir; 10 samples)
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"
mkdir -p logs/lineB_downsampled_x4_up datasets/lineB_downsampled_x4_up_smoke

echo "=== Line B PU-Net smoke (256->1024) started: $(date) ==="

source scripts/activate_upsampling_env.sh punet
bash scripts/print_gpu_cuda_diagnostics.sh

PUNET_CODE="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-Net/code"
cd "${PUNET_CODE}"
if ! python -c "from tf_ops.sampling.tf_sampling import farthest_point_sample" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh punet
fi
cd "${PROJECT_ROOT}"

python scripts/run_lineB_downsampled_x4_smoke.py \
  --method pu_net \
  --max-per-split 5 \
  --smoke-tag rerun_mkdir_fix \
  --audit reports/modelnet40_lineB_punet_smoke_audit.csv

python scripts/audit_lineB_downsampled_x4_smoke.py \
  --method pu_net \
  --smoke-tag rerun_mkdir_fix \
  --expected-train 5 \
  --expected-test 5

cp -f reports/lineB_downsampled_x4_smoke_audit_pu_net_rerun_mkdir_fix.csv reports/modelnet40_lineB_punet_smoke_audit.csv 2>/dev/null || true
cp -f reports/lineB_downsampled_x4_smoke_audit_pu_net_rerun_mkdir_fix.md reports/modelnet40_lineB_punet_smoke_audit.md 2>/dev/null || true

echo "=== Line B PU-Net smoke finished: $(date) ==="
