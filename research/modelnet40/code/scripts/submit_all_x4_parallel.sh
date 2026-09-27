#!/usr/bin/env bash
# Submit PDANS / PU-Net / PU-GCN ×4 smoke+full pipelines in parallel.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
SUBMIT_LOG="reports/x4_parallel_generation/submit_${TS}.md"
mkdir -p reports/x4_parallel_generation

{
  echo "# ×4 Parallel Generation Submit"
  echo ""
  echo "- Submitted at: ${TS}"
  echo "- EAR jobs (pre-existing, not resubmitted): 1716174 (A), 1716175 (B)"
  echo ""
} > "${SUBMIT_LOG}"

for METHOD in pdans punet pugcn; do
  echo "=== Submitting ${METHOD} ==="
  OUT=$(bash scripts/submit_upsampling_x4_method.sh "${METHOD}" 2>&1 | tee -a "${SUBMIT_LOG}.tmp")
  echo "${OUT}"
  echo "" >> "${SUBMIT_LOG}"
  echo "### ${METHOD}" >> "${SUBMIT_LOG}"
  echo '```' >> "${SUBMIT_LOG}"
  echo "${OUT}" >> "${SUBMIT_LOG}"
  echo '```' >> "${SUBMIT_LOG}"
  echo "" >> "${SUBMIT_LOG}"
done

rm -f "${SUBMIT_LOG}.tmp"
python scripts/generate_x4_parallel_status.py --submit-log "${SUBMIT_LOG}"
echo "Done. Status: reports/modelnet40_x4_parallel_generation_status.md"
