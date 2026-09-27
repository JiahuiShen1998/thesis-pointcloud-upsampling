#!/bin/bash
#SBATCH --job-name=pn2_down50
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/downsampled50_baseline_%j.log
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/downsampled50_baseline_%j.err

# Step 6: Downsampled50 baseline (Line B)
# Dataset: modelnet40_downsampled50 (512-point .npy)
# Loader resamples 512 -> 1024 with fixed seed before PointNet++ (num_point=1024)
#
# Submit ONLY after Step 4 (Job 1709666) completes:
#   sbatch.tinygpu /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/train_downsampled50_baseline_gpu.sbatch

set -euo pipefail
PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
bash "${PROJECT_ROOT}/scripts/train_pointnet2.sh" downsampled50_baseline
