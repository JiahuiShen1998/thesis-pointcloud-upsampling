#!/usr/bin/env bash
# GPU env validation: compile TF ops + one-sample smoke per method (no full generation).
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"

REPORT="${PROJECT_ROOT}/reports/x4_parallel_generation/env_validate_$(date -u +%Y%m%dT%H%M%SZ).md"
mkdir -p "${PROJECT_ROOT}/reports/x4_parallel_generation"
exec > >(tee -a "${REPORT}") 2>&1

echo "# Upsampling ×4 Environment Validate"
echo ""
echo "- Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

fail=0
pass_msg() { echo "PASS: $*"; }
fail_msg() { echo "FAIL: $*"; fail=1; }

echo "## Shared diagnostics"
bash scripts/print_gpu_cuda_diagnostics.sh || true
echo ""

validate_pdans_cuda() {
  echo "## PDANS CUDA + one-sample"
  source scripts/activate_upsampling_env.sh pdans
  bash scripts/print_gpu_cuda_diagnostics.sh
  if python -c "import torch; assert torch.cuda.is_available(), 'no cuda'"; then
    pass_msg "PDANS torch.cuda.is_available()"
  else
    fail_msg "PDANS torch.cuda.is_available()"
    echo ""
    return
  fi
  python scripts/run_upsampling_x4_smoke.py \
    --method pdans --line A --smoke-tag envfix --max-per-split 1 --overwrite
  python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path("${PROJECT_ROOT}") / "scripts"))
import numpy as np
from upsampling_x4_common import smoke_variant_config
v = smoke_variant_config("pdans", "A", smoke_tag="envfix")
files = list((v["output_root"] / "train").rglob("*.npy"))
assert files, "no output"
arr = np.load(files[0])
assert arr.shape == (4096, 3), arr.shape
assert np.isfinite(arr).all()
print("shape ok:", arr.shape)
PY
  pass_msg "PDANS line A one-sample 4096x3"
  echo ""
}

validate_tf_method() {
  local method="$1"
  local line="$2"
  local expected_pts="$3"
  echo "## ${method^^} compile + one-sample line ${line}"
  source scripts/activate_upsampling_env.sh "${method}"
  bash scripts/print_gpu_cuda_diagnostics.sh
  bash scripts/recompile_tf_ops.sh "${method}"

  local root="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated"
  if [[ "${method}" == "punet" ]]; then
    cd "${root}/PU-Net/code"
    python -c "from tf_ops.sampling.tf_sampling import farthest_point_sample; print('import ok')"
  else
    cd "${root}/PU-GCN"
    python -c "from tf_ops.grouping.tf_grouping import knn_point_2; print('import ok')"
  fi
  cd "${PROJECT_ROOT}"

  python scripts/run_upsampling_x4_smoke.py \
    --method "${method}" --line "${line}" \
    --smoke-tag envfix --max-per-split 1 --overwrite
  python scripts/audit_upsampling_x4_smoke.py \
    --method "${method}" --line "${line}" --smoke-tag envfix

  python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path("${PROJECT_ROOT}") / "scripts"))
import numpy as np
from upsampling_x4_common import smoke_variant_config
v = smoke_variant_config("${method}", "${line}", smoke_tag="envfix")
files = list((v["output_root"] / "train").rglob("*.npy"))
assert files, "no output"
arr = np.load(files[0])
assert arr.shape == (${expected_pts}, 3), arr.shape
assert np.isfinite(arr).all()
print("shape ok:", arr.shape)
PY
  pass_msg "${method} line ${line} one-sample ${expected_pts}x3"
  echo ""
}

validate_pdans_cuda
validate_tf_method punet A 4096
validate_tf_method punet B 2048
validate_tf_method pugcn A 4096
validate_tf_method pugcn B 2048

echo "## Summary"
if [[ ${fail} -eq 0 ]]; then
  echo "ALL CHECKS PASS"
  exit 0
fi
echo "SOME CHECKS FAILED (see above)"
exit 1
