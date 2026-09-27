#!/usr/bin/env bash
# Submit Line B new protocol upsampling for EAR, PU-Net, PDANS in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs/lineB_downsampled_x4_up reports/lineB_downsampled_x4_up

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REPORT="reports/lineB_downsampled_x4_up/submit_all_${TS}.md"

echo "# Line B New Protocol — Parallel Submit" > "${REPORT}"
echo "" >> "${REPORT}"
echo "- Submitted at: ${TS}" >> "${REPORT}"
echo "- Protocol: 256 → 1024 (modelnet40_downsampled_x4)" >> "${REPORT}"
echo "- PU-GCN: pending (skipped)" >> "${REPORT}"
echo "" >> "${REPORT}"

for METHOD in ear pu_net pdans; do
  OUT=$(bash scripts/submit_lineB_downsampled_x4_method.sh "${METHOD}")
  echo "${OUT}"
  echo "- **${METHOD}**: ${OUT}" >> "${REPORT}"
  echo "" >> "${REPORT}"
done

echo "Report: ${REPORT}"
