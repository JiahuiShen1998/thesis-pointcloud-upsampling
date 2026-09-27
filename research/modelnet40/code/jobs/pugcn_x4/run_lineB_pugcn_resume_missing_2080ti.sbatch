#!/bin/bash
#SBATCH --job-name=mn40_B_pugcn_resume
#SBATCH --partition=work
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --time=24:00:00
#SBATCH --exclude=tg080,tg081,tg082,tg083,tg084,tg085
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_resume_%A_%a.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_resume_%A_%a.err

# Resume PU-GCN Line B: only missing 2301 samples (256→1024), skip existing 10010.
# Avoid tg084 / RTX 3080 — previous full job failed at knn_point_2 on that node.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
PUGCN_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
CHECKPOINT="${PUGCN_ROOT}/pretrained/pu1k-pugcn"
MISSING_LIST="${PROJECT_ROOT}/reports/modelnet40_lineB_pugcn_missing_samples.txt"
NUM_CHUNKS=8
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"
FAILED_CSV="${PROJECT_ROOT}/reports/modelnet40_lineB_pugcn_resume_failed_samples_${CHUNK_ID}.csv"

cd "${PROJECT_ROOT}"
mkdir -p logs/pugcn_x4 reports \
  datasets/lineB_downsampled_x4_up/raw/pu_gcn \
  datasets/lineB_downsampled_x4_up/strict_N/pu_gcn

echo "=== Line B PU-GCN resume chunk ${CHUNK_ID}/${NUM_CHUNKS} started: $(date) ==="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "SLURM_ARRAY_TASK_ID: ${CHUNK_ID}"
echo "missing list: ${MISSING_LIST}"
echo "checkpoint: ${CHECKPOINT}"
echo "failed csv: ${FAILED_CSV}"
nvidia-smi || true

source scripts/activate_upsampling_env.sh pugcn
python -c "import tensorflow as tf; print('tensorflow', tf.__version__)" || true
python -c "import torch; print('torch', torch.__version__)" 2>/dev/null || echo "torch: not installed in pugcn env"

cd "${PUGCN_ROOT}"
if ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh pugcn
fi
cd "${PROJECT_ROOT}"

python scripts/run_lineB_pugcn_resume_missing.py \
  --missing-list "${MISSING_LIST}" \
  --input-root datasets/modelnet40_downsampled_x4 \
  --raw-output-root datasets/lineB_downsampled_x4_up/raw/pu_gcn \
  --strict-output-root datasets/lineB_downsampled_x4_up/strict_N/pu_gcn \
  --expected-input-points 256 \
  --expected-output-points 1024 \
  --checkpoint "${CHECKPOINT}" \
  --array-id "${CHUNK_ID}" \
  --array-count "${NUM_CHUNKS}" \
  --resume \
  --seed 42 \
  --failed-csv "${FAILED_CSV}"

echo "=== Line B PU-GCN resume chunk ${CHUNK_ID} finished: $(date) ==="
echo -n "raw count: "
find datasets/lineB_downsampled_x4_up/raw/pu_gcn -name "*.npy" 2>/dev/null | wc -l
echo -n "strict count: "
find datasets/lineB_downsampled_x4_up/strict_N/pu_gcn -name "*.npy" 2>/dev/null | wc -l
