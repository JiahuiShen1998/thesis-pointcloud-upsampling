#!/bin/bash
#SBATCH --job-name=mn40_puedge_smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pu_edgeformer/labparams_smoke_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pu_edgeformer/labparams_smoke_%j.err

# ModelNet40 PU-EdgeFormer lab-params smoke ONLY (no full run, no detector/classifier/AP).
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"
mkdir -p logs/pu_edgeformer

echo "=== PU-EdgeFormer ModelNet40 smoke started: $(date) ==="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
nvidia-smi || true

# Reuse pugcn TF1.15 activation path (same env as PU-GCN/PU-Net on this HPC).
source scripts/activate_upsampling_env.sh pugcn
bash scripts/print_gpu_cuda_diagnostics.sh || true

# Ensure PU-GCN tf_ops load (EdgeFormer reuses them via symlink).
PUGCN_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
cd "${PUGCN_ROOT}"
if ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh pugcn
fi
cd "${PROJECT_ROOT}"

# Quick EdgeFormer repo import check
python - <<'PY'
import sys
sys.path.insert(0, "external/pu_edgeformer_ops_reuse")
from tf_ops.grouping.tf_grouping import knn_point_2
from tf_ops.sampling.tf_sampling import farthest_point_sample
print("EdgeFormer tf_ops import: OK")
PY

python scripts/run_pu_edgeformer_modelnet40_labparams_smoke.py \
  --max-per-split 5 \
  --gpu-memory-fraction 0.45 \
  --modes lineB A1 A2

echo "=== PU-EdgeFormer ModelNet40 smoke finished: $(date) ==="
echo "DETECTOR_EVAL_STARTED=NO"
echo "POINTNET_CLASSIFIER_STARTED=NO"
echo "KITTI_AP_EVAL_STARTED=NO"
