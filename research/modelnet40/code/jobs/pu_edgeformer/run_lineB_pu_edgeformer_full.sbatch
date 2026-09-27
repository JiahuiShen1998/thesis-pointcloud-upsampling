#!/bin/bash
#SBATCH --job-name=mn40_B_puedge_full
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=08:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pu_edgeformer/lineB_full_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pu_edgeformer/lineB_full_%A_%a.err

# Line B PU-EdgeFormer FULL: 256 -> 1024 direct (12311 samples, 16 chunks)
# No detector / classifier / KITTI AP / geometry metrics.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
NUM_CHUNKS=16
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

cd "${PROJECT_ROOT}"
mkdir -p logs/pu_edgeformer \
  datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_20260716/{train,test,logs,manifests,audits,method_raw}

echo "=== Line B PU-EdgeFormer FULL chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
echo "hostname: $(hostname)"
nvidia-smi || true

source scripts/activate_upsampling_env.sh pugcn
bash scripts/print_gpu_cuda_diagnostics.sh || true

PUGCN_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
cd "${PUGCN_ROOT}"
if ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh pugcn
fi
cd "${PROJECT_ROOT}"

python - <<'PY'
import sys
sys.path.insert(0, "external/pu_edgeformer_ops_reuse")
from tf_ops.grouping.tf_grouping import knn_point_2
print("EdgeFormer tf_ops import: OK")
PY

python scripts/run_pu_edgeformer_modelnet40_labparams_full.py \
  --line lineB \
  --mode direct_256_to_1024 \
  --chunk-id "${CHUNK_ID}" \
  --num-chunks "${NUM_CHUNKS}" \
  --skip-existing \
  --batch-size 200 \
  --gpu-memory-fraction 0.45

echo "=== Line B chunk ${CHUNK_ID} finished: $(date) ==="
echo "DETECTOR_EVAL_STARTED=NO"
echo "POINTNET_CLASSIFIER_STARTED=NO"
echo "KITTI_AP_EVAL_STARTED=NO"
