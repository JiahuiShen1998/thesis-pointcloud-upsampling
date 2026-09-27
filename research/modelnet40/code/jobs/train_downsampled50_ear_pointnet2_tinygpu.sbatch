#!/bin/bash
#SBATCH --job-name=p2_ear_b
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/step8_downsampled50_ear_pointnet2/train_%j.log
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/step8_downsampled50_ear_pointnet2/train_%j.err

# Step 8: Downsampled50 + EAR → PointNet++ (Line B)
# Native point count: 1024 (EAR upsampled from 512-point downsampled input)
# Comparison: downstream classification vs downsampled50_baseline (512 native)
#
# Submit:
#   sbatch.tinygpu /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/train_downsampled50_ear_pointnet2_tinygpu.sbatch

set -euo pipefail
PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
mkdir -p "${PROJECT_ROOT}/logs/step8_downsampled50_ear_pointnet2"
bash "${PROJECT_ROOT}/scripts/train_pointnet2.sh" downsampled50_ear_pointnet2
