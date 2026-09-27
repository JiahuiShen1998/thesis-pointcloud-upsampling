#!/bin/bash
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=x4_env_validate_pdans2
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/x4_env_validate_pdans2_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/x4_env_validate_pdans2_%j.err

set -euo pipefail
PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
bash scripts/validate_pdans_gpu_env.sh
