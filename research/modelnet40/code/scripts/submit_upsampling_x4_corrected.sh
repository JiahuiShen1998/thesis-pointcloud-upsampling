#!/usr/bin/env bash
# Re-submit corrected PU-Net / PU-GCN / PDANS ×4 smoke + full (after METHOD bug fix).
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"

export SMOKE_TAG="rerun"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG="reports/x4_parallel_generation/submit_corrected_${TS}.md"

{
  echo "# Corrected ×4 Upsampling Submit"
  echo ""
  echo "- Submitted at: ${TS}"
  echo "- SMOKE_TAG: rerun"
  echo "- Smoke dirs: \`{method}_x4_smoke_rerun\`"
  echo ""
} > "${LOG}"

for METHOD in pdans punet pugcn; do
  echo "=== Dry-run validate ${METHOD} ==="
  validated="$("${PROJECT_ROOT}/scripts/validate_upsampling_method.sh" "${METHOD}")"
  echo "validated METHOD=${validated}"
  OUT=$(SMOKE_TAG="${SMOKE_TAG}" bash scripts/submit_upsampling_x4_method.sh "${METHOD}" 2>&1)
  echo "${OUT}"
  echo "" >> "${LOG}"
  echo "### ${METHOD}" >> "${LOG}"
  echo '```' >> "${LOG}"
  echo "${OUT}" >> "${LOG}"
  echo '```' >> "${LOG}"
done

echo "Corrected submit log: ${LOG}"
