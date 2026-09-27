#!/bin/bash
#SBATCH --job-name=lineB_qmetrics
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A.err

set -euo pipefail
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling
module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
python scripts/compute_lineB_completed_methods_quality_metrics.py
