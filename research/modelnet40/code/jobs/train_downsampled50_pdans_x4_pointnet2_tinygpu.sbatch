#!/bin/bash
#SBATCH --job-name=p2_pdans4_b
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/downsampled50_pdans_x4_pointnet2/train_%j.log
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/downsampled50_pdans_x4_pointnet2/train_%j.err

set -euo pipefail
PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
mkdir -p "${PROJECT_ROOT}/logs/downsampled50_pdans_x4_pointnet2"
bash "${PROJECT_ROOT}/scripts/train_pointnet2.sh" downsampled50_pdans_x4_pointnet2
