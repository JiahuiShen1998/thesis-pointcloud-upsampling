#!/bin/bash
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=pdans_x4_smoke
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pdans_x4/smoke_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pdans_x4/smoke_%j.err

set -euo pipefail
PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"
mkdir -p logs/pdans_x4 reports/pdans_x4

source scripts/activate_upsampling_env.sh pdans
bash scripts/print_gpu_cuda_diagnostics.sh

SMOKE_RC=0
AUDIT_RC=0

for LINE in A B; do
  echo "=== PDANS smoke Line ${LINE} ==="
  python scripts/run_upsampling_x4_smoke.py \
    --method pdans --line "${LINE}" \
    --max-per-split 40 --overwrite || SMOKE_RC=1
done

python scripts/audit_upsampling_x4_smoke.py \
  --method pdans \
  --expected-train 40 --expected-test 40 \
  --csv-out reports/pdans_x4/pdans_x4_smoke_audit.csv \
  --md-out reports/pdans_x4/pdans_x4_smoke_audit.md || AUDIT_RC=1

if [[ ${SMOKE_RC} -ne 0 || ${AUDIT_RC} -ne 0 ]]; then
  echo "PDANS smoke FAIL: smoke=${SMOKE_RC} audit=${AUDIT_RC}"
  exit 1
fi
echo "PDANS smoke PASS: $(date)"
