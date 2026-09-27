#!/usr/bin/env bash
# GPU validation for PDANS: CUDA, pytorch3d, import, single-sample smoke (no full generation).
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${PROJECT_ROOT}/reports/x4_parallel_generation/x4_env_validate_pdans2_${TS}.md"
mkdir -p "${PROJECT_ROOT}/reports/x4_parallel_generation"
exec > >(tee -a "${REPORT}") 2>&1

echo "# PDANS GPU Environment Validate (pdans2)"
echo ""
echo "- Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "- Job purpose: PASS_PDANS_READY_FOR_SMOKE gate"
echo ""

fail=0
pass_msg() { echo "PASS: $*"; }
fail_msg() { echo "FAIL: $*"; fail=1; }

echo "## Diagnostics"
source scripts/activate_upsampling_env.sh pdans
bash scripts/print_gpu_cuda_diagnostics.sh
echo ""

echo "## torch.cuda"
if python -c "import torch; assert torch.cuda.is_available(), 'no cuda'"; then
  pass_msg "torch.cuda.is_available()"
else
  fail_msg "torch.cuda.is_available()"
fi

echo "## pytorch3d GPU knn_points"
python - <<'PY'
import torch
from pytorch3d.ops import knn_points
assert torch.cuda.is_available()
x = torch.randn(1, 16, 3, device="cuda")
y = torch.randn(1, 32, 3, device="cuda")
d, idx, nn = knn_points(x, y, K=1, return_nn=True)
print("knn gpu ok:", d.shape, idx.shape, nn.shape)
PY
if [[ $? -eq 0 ]]; then pass_msg "pytorch3d knn_points on GPU"; else fail_msg "pytorch3d knn_points on GPU"; fi

echo "## PDANS import"
python - <<'PY'
import sys
from pathlib import Path
root = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
sys.path.insert(0, str(root / "scripts"))
sys.path[:0] = [
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS/pointnet2_ops_lib",
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS",
]
from pointnet2_ops import pointnet2_utils
from upsampling_x4_factory import load_upsampler
import torch
assert torch.cuda.is_available()
u = load_upsampler("pdans")
print("PDANS upsampler:", type(u).__name__)
PY
if [[ $? -eq 0 ]]; then pass_msg "PDANS import + load_upsampler"; else fail_msg "PDANS import + load_upsampler"; fi

if [[ ${fail} -ne 0 ]]; then
  echo ""
  echo "## Conclusion: FAIL_PDANS_IMPORT_OR_COMPILE"
  exit 1
fi

SMOKE_TAG="validate2"
for LINE in A B; do
  echo "## Single-sample smoke line ${LINE}"
  python scripts/run_upsampling_x4_smoke.py \
    --method pdans --line "${LINE}" \
    --smoke-tag "${SMOKE_TAG}" --max-per-split 1 --overwrite
  python scripts/audit_upsampling_x4_smoke.py \
    --method pdans --line "${LINE}" --smoke-tag "${SMOKE_TAG}"
  EXPECT=$([[ "${LINE}" == "A" ]] && echo 4096 || echo 2048)
  python - <<PY
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("${PROJECT_ROOT}") / "scripts"))
from upsampling_x4_common import smoke_variant_config
v = smoke_variant_config("pdans", "${LINE}", smoke_tag="${SMOKE_TAG}")
files = list((v["output_root"] / "train").rglob("*.npy"))
assert files, "no output"
arr = np.load(files[0])
assert arr.shape == (${EXPECT}, 3), arr.shape
assert np.isfinite(arr).all()
print("shape ok:", arr.shape)
PY
  if [[ $? -eq 0 ]]; then pass_msg "PDANS line ${LINE} one-sample ${EXPECT}x3"; else fail_msg "PDANS line ${LINE} smoke"; fail=1; fi
  echo ""
done

echo "## Conclusion"
if [[ ${fail} -eq 0 ]]; then
  echo "**PASS_PDANS_READY_FOR_SMOKE**"
  exit 0
fi
echo "**FAIL** (see sections above)"
exit 1
