#!/bin/bash
#SBATCH --job-name=mn40_qmetrics
#SBATCH --partition=work
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/quality_metrics/%x_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/quality_metrics/%x_%j.err

set -euo pipefail
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling
module purge 2>/dev/null || true
module load python/pytorch2.6py3.12

python scripts/compute_modelnet40_x4_quality_metrics.py \
  --p2f exact \
  --gt-workers 16 \
  --metrics-workers 8
