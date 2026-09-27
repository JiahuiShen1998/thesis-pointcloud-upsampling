#!/usr/bin/env bash
# GPU validation gate for PDANS ×4 — writes reports/pdans_x4/pdans_x4_gpu_validate_report.md
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
PDANS_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS"
REPORT="${PROJECT_ROOT}/reports/pdans_x4/pdans_x4_gpu_validate_report.md"
SMOKE_TAG="validate"
LOG_DIR="${PROJECT_ROOT}/logs/pdans_x4"
mkdir -p "${PROJECT_ROOT}/reports/pdans_x4" "${LOG_DIR}"

cd "${PROJECT_ROOT}"
exec > >(tee "${LOG_DIR}/validate_gpu_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1

{
  echo "# PDANS ×4 GPU Validate Report"
  echo ""
  echo "- Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Host: $(hostname)"
  echo "- SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
  echo ""
} > "${REPORT}"

fail=0
pass_msg() { echo "PASS: $*" | tee -a "${REPORT}"; }
fail_msg() { echo "FAIL: $*" | tee -a "${REPORT}"; fail=1; }

echo "## Environment" | tee -a "${REPORT}"
source scripts/activate_upsampling_env.sh pdans
nvidia-smi | tee -a "${REPORT}" || true
echo "" | tee -a "${REPORT}"
{
  echo "- python: $(which python)"
  echo "- python version: $(python --version 2>&1)"
  python - <<'PY'
import torch
try:
    import pytorch3d
    p3d = pytorch3d.__version__
except Exception as exc:
    p3d = f"unavailable ({exc})"
print(f"- torch: {torch.__version__}")
print(f"- torch.version.cuda: {torch.version.cuda}")
print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"- GPU: {torch.cuda.get_device_name(0)}")
print(f"- pytorch3d: {p3d}")
PY
} | tee -a "${REPORT}"
echo "" | tee -a "${REPORT}"

echo "## torch.cuda" | tee -a "${REPORT}"
if python -c "import torch; assert torch.cuda.is_available()"; then
  pass_msg "torch.cuda.is_available()"
else
  fail_msg "torch.cuda.is_available()"
fi

echo "## pytorch3d GPU knn_points" | tee -a "${REPORT}"
if python - <<'PY'
import torch
from pytorch3d.ops import knn_points
x = torch.randn(1, 16, 3, device="cuda")
y = torch.randn(1, 32, 3, device="cuda")
d, idx, nn = knn_points(x, y, K=1, return_nn=True)
print("knn shapes:", d.shape, idx.shape, nn.shape)
PY
then
  pass_msg "pytorch3d knn_points on GPU"
else
  fail_msg "pytorch3d knn_points on GPU"
fi

echo "## PDANS checkpoints" | tee -a "${REPORT}"
for ckpt in "${PDANS_ROOT}/checkpoints/PU1K_PDANS.pkl" "${PDANS_ROOT}/checkpoints/PUGAN_PDANS.pkl"; do
  if [[ -f "${ckpt}" ]]; then
    echo "PASS: checkpoint exists — ${ckpt}" | tee -a "${REPORT}"
  else
    fail_msg "checkpoint missing — ${ckpt}"
  fi
done

echo "## PDANS import + load_upsampler" | tee -a "${REPORT}"
if python - <<'PY'
import sys
from pathlib import Path
root = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
sys.path.insert(0, str(root / "scripts"))
from upsampling_x4_factory import load_upsampler
import torch
assert torch.cuda.is_available()
u = load_upsampler("pdans")
print("PDANS upsampler:", type(u).__name__)
PY
then
  pass_msg "PDANS import + load_upsampler"
else
  fail_msg "PDANS import + load_upsampler"
fi

if [[ ${fail} -ne 0 ]]; then
  {
    echo ""
    echo "## Conclusion: **FAIL**"
    echo "Blocked at import/checkpoint stage — do not submit smoke or full generation."
  } | tee -a "${REPORT}"
  exit 1
fi

for LINE in A B; do
  echo "## Single-sample smoke Line ${LINE}" | tee -a "${REPORT}"
  OUT_A="${PROJECT_ROOT}/datasets/modelnet40_original_up/pdans_x4_smoke_validate"
  OUT_B="${PROJECT_ROOT}/datasets/modelnet40_downsampled50_up/pdans_x4_smoke_validate"
  EXPECT=$([[ "${LINE}" == "A" ]] && echo 4096 || echo 2048)
  INP=$([[ "${LINE}" == "A" ]] && echo 1024 || echo 512)

  if python scripts/run_upsampling_x4_smoke.py \
    --method pdans --line "${LINE}" \
    --smoke-tag "${SMOKE_TAG}" --max-per-split 1 --overwrite; then
    pass_msg "smoke generation Line ${LINE}"
  else
    fail_msg "smoke generation Line ${LINE}"
    continue
  fi

  if python - <<PY
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("${PROJECT_ROOT}") / "scripts"))
from upsampling_x4_common import smoke_variant_config
v = smoke_variant_config("pdans", "${LINE}", smoke_tag="${SMOKE_TAG}")
files = sorted((v["output_root"] / "train").rglob("*.npy"))
assert files, "no output"
arr = np.load(files[0])
assert arr.shape == (${EXPECT}, 3), arr.shape
assert np.isfinite(arr).all()
print("output:", v['output_root'])
print("shape:", arr.shape, "finite:", True)
PY
  then
    pass_msg "Line ${LINE} ${INP}→${EXPECT} shape + finite"
  else
    fail_msg "Line ${LINE} output shape/NaN check"
  fi
  echo "" | tee -a "${REPORT}"
done

echo "## Output directories" | tee -a "${REPORT}"
echo "- Line A: \`datasets/modelnet40_original_up/pdans_x4_smoke_validate\`" | tee -a "${REPORT}"
echo "- Line B: \`datasets/modelnet40_downsampled50_up/pdans_x4_smoke_validate\`" | tee -a "${REPORT}"
echo "" | tee -a "${REPORT}"

if [[ ${fail} -eq 0 ]]; then
  {
    echo "## Conclusion: **PASS**"
    echo ""
    echo "PDANS GPU validate complete. Ready for smoke generation (80 samples/line)."
  } | tee -a "${REPORT}"
  exit 0
fi

{
  echo "## Conclusion: **FAIL**"
  echo "See sections above."
} | tee -a "${REPORT}"
exit 1
