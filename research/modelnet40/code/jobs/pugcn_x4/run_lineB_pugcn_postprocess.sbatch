#!/bin/bash
#SBATCH --job-name=mn40_B_pugcn_qmetrics
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_qmetrics_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_qmetrics_%j.err

# Line B PU-GCN quality metrics — submit after final count audit PASS
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"

echo "=== Line B PU-GCN quality metrics started: $(date) ==="
module purge 2>/dev/null || true
module load python/pytorch2.6py3.12 2>/dev/null || true

python3 scripts/audit_pugcn_final_count.py
python3 scripts/compute_lineB_pugcn_quality_metrics.py
python3 scripts/prepare_pugcn_pointnet2_inputs.py
python3 scripts/audit_pugcn_provenance.py

echo "=== Line B PU-GCN post-process finished: $(date) ==="
