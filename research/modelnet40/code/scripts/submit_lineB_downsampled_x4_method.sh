#!/usr/bin/env bash
# Submit Line B new protocol upsampling (256→1024) for one method.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

METHOD="${1:?Usage: $0 ear|pu_net|pdans}"
if [[ "${METHOD}" == "pu_gcn" ]]; then
  echo "PU-GCN is pending for new Line B protocol."
  exit 1
fi

SMOKE_TAG="${SMOKE_TAG:-}"
mkdir -p logs/lineB_downsampled_x4_up reports/lineB_downsampled_x4_up
mkdir -p datasets/lineB_downsampled_x4_up/raw datasets/lineB_downsampled_x4_up/strict_N
mkdir -p datasets/lineB_downsampled_x4_up_smoke

export_vars="METHOD=${METHOD}"
if [[ -n "${SMOKE_TAG}" ]]; then
  export_vars="${export_vars},SMOKE_TAG=${SMOKE_TAG}"
fi

smoke_job=$(sbatch --parsable \
  --job-name="lineB_x4_${METHOD}_smoke${SMOKE_TAG:+_${SMOKE_TAG}}" \
  --export="${export_vars}" \
  jobs/run_lineB_downsampled_x4_smoke.sbatch)

full_job=$(sbatch --parsable \
  --job-name="lineB_x4_${METHOD}${SMOKE_TAG:+_${SMOKE_TAG}}" \
  --dependency=afterok:"${smoke_job}" \
  --export="${export_vars}" \
  jobs/run_lineB_downsampled_x4_full_array.sbatch)

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REPORT="reports/lineB_downsampled_x4_up/submit_${METHOD}_${TS}.md"
mkdir -p reports/lineB_downsampled_x4_up
cat > "${REPORT}" <<EOF
# Line B New Protocol Upsampling Submit — ${METHOD}

- Submitted at: ${TS}
- Input: \`datasets/modelnet40_downsampled_x4\` (256 pts)
- Output: \`datasets/lineB_downsampled_x4_up/raw/${METHOD}\` → \`strict_N/${METHOD}\` (1024 pts)
- Smoke job: **${smoke_job}**
- Full array job: **${full_job}** (depends on smoke)
- SMOKE_TAG: \`${SMOKE_TAG:-}\`
EOF

echo "Method ${METHOD}: smoke=${smoke_job} full=${full_job}"
echo "Report: ${REPORT}"
