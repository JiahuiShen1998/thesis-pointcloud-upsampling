#!/bin/bash
#SBATCH --job-name=pn2_original
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/original_baseline_%j.log
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/original_baseline_%j.err

set -euo pipefail
PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
bash "${PROJECT_ROOT}/scripts/train_pointnet2.sh" original_baseline
