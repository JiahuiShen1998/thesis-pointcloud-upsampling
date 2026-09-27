#!/bin/bash
#SBATCH --job-name=mn40_ef_qmetrics
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/quality_metrics/pu_edgeformer_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/quality_metrics/pu_edgeformer_%j.err

# PU-EdgeFormer ONLY geometry metrics (CD/HD/NUC/exact P2F) + tables + delta figures.
# No PointNet++ / detector / KITTI AP.
set -euo pipefail
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling
mkdir -p logs/quality_metrics

echo "=== PU-EdgeFormer geometry metrics started: $(date) ==="
echo "GEOMETRY_METRICS_STARTED=YES"
echo "POINTNET_CLASSIFIER_STARTED=NO"
echo "DETECTOR_EVAL_STARTED=NO"
echo "KITTI_AP_EVAL_STARTED=NO"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12

python scripts/compute_pu_edgeformer_quality_metrics.py \
  --p2f exact \
  --metrics-workers 8

python scripts/generate_pu_edgeformer_geometry_delta_figures.py

echo "=== PU-EdgeFormer geometry metrics finished: $(date) ==="
echo "POINTNET_CLASSIFIER_STARTED=NO"
echo "DETECTOR_EVAL_STARTED=NO"
echo "KITTI_AP_EVAL_STARTED=NO"
