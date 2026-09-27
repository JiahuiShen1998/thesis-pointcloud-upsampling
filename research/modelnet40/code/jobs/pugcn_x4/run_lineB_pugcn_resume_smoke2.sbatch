#!/bin/bash
#SBATCH --job-name=mn40_B_pugcn_resume_smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --exclude=tg080,tg081,tg082,tg083,tg084,tg085
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_resume_smoke_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pugcn_x4/lineB_pugcn_resume_smoke_%j.err

# Smoke test for fixed resume runner: 2 missing samples only.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
PUGCN_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
CHECKPOINT="${PUGCN_ROOT}/pretrained/pu1k-pugcn"
SMOKE_LIST="${PROJECT_ROOT}/reports/modelnet40_lineB_pugcn_missing_samples_smoke2.txt"

cd "${PROJECT_ROOT}"
mkdir -p logs/pugcn_x4 reports

echo "=== Line B PU-GCN resume smoke started: $(date) ==="
echo "hostname: $(hostname)"
nvidia-smi || true
python3 -m py_compile scripts/run_lineB_pugcn_resume_missing.py

source scripts/activate_upsampling_env.sh pugcn

cd "${PUGCN_ROOT}"
if ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
  cd "${PROJECT_ROOT}"
  bash scripts/recompile_tf_ops.sh pugcn
fi
cd "${PROJECT_ROOT}"

python scripts/run_lineB_pugcn_resume_missing.py \
  --missing-list "${SMOKE_LIST}" \
  --input-root datasets/modelnet40_downsampled_x4 \
  --raw-output-root datasets/lineB_downsampled_x4_up/raw/pu_gcn \
  --strict-output-root datasets/lineB_downsampled_x4_up/strict_N/pu_gcn \
  --expected-input-points 256 \
  --expected-output-points 1024 \
  --resume \
  --seed 42 \
  --max-samples 2 \
  --failed-csv reports/modelnet40_lineB_pugcn_resume_smoke_failed.csv

python3 - <<'PY'
from pathlib import Path
import numpy as np

smoke = Path("reports/modelnet40_lineB_pugcn_missing_samples_smoke2.txt")
ok = True
for rel in smoke.read_text().splitlines():
    rel = rel.strip()
    if not rel:
        continue
    p = Path("datasets/lineB_downsampled_x4_up/strict_N/pu_gcn") / rel
    if not p.is_file():
        print(f"FAIL missing strict: {p}")
        ok = False
        continue
    arr = np.load(p)
    if arr.shape != (1024, 3) or np.isnan(arr).any() or np.isinf(arr).any():
        print(f"FAIL bad strict: {p} shape={arr.shape}")
        ok = False
    else:
        print(f"PASS {p}")
if not ok:
    raise SystemExit(1)
print("SMOKE_AUDIT_PASS")
PY

echo "=== Line B PU-GCN resume smoke finished: $(date) ==="
