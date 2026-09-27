#!/usr/bin/env bash
# Submit smoke + dependent full generation for one method (both lines).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

METHOD="$("${SCRIPT_DIR}/validate_upsampling_method.sh" "${1:?Usage: $0 {pdans|punet|pugcn}}")"
SMOKE_TAG="${SMOKE_TAG:-}"

mkdir -p "logs/${METHOD}_x4_generation" "reports/${METHOD}_x4_generation"
mkdir -p "datasets/modelnet40_original_up/${METHOD}_x4"
mkdir -p "datasets/modelnet40_downsampled50_up/${METHOD}_x4"
if [[ -n "${SMOKE_TAG}" ]]; then
  mkdir -p "datasets/modelnet40_original_up/${METHOD}_x4_smoke_${SMOKE_TAG}"
  mkdir -p "datasets/modelnet40_downsampled50_up/${METHOD}_x4_smoke_${SMOKE_TAG}"
else
  mkdir -p "datasets/modelnet40_original_up/${METHOD}_x4_smoke"
  mkdir -p "datasets/modelnet40_downsampled50_up/${METHOD}_x4_smoke"
fi

SMOKE_A="" SMOKE_B="" FULL_A="" FULL_B=""
JOB_ENV="reports/${METHOD}_x4_generation/job_ids.env"
mkdir -p "reports/${METHOD}_x4_generation"

submit_line() {
  local line="$1"
  local line_lc
  line_lc="$(echo "${line}" | tr '[:upper:]' '[:lower:]')"

  local smoke_job full_job export_vars
  export_vars="METHOD=${METHOD},LINE=${line}"
  if [[ -n "${SMOKE_TAG}" ]]; then
    export_vars="${export_vars},SMOKE_TAG=${SMOKE_TAG}"
  fi

  smoke_job=$(sbatch --parsable \
    --job-name="${METHOD}_x4_${line_lc}_smoke${SMOKE_TAG:+_${SMOKE_TAG}}" \
    --export="${export_vars}" \
    jobs/run_upsampling_x4_smoke.sbatch)

  full_job=$(sbatch --parsable \
    --job-name="${METHOD}_x4_${line_lc}${SMOKE_TAG:+_${SMOKE_TAG}}" \
    --dependency=afterok:"${smoke_job}" \
    --export="${export_vars}" \
    jobs/run_upsampling_x4_full_array.sbatch)

  if [[ "${line}" == "A" ]]; then
    SMOKE_A="${smoke_job}"
    FULL_A="${full_job}"
  else
    SMOKE_B="${smoke_job}"
    FULL_B="${full_job}"
  fi

  echo "Line ${line}: smoke=${smoke_job} full=${full_job} METHOD=${METHOD}"
}

echo "Submitting ${METHOD} ×4 pipeline (SMOKE_TAG=${SMOKE_TAG:-none})..."
submit_line A
submit_line B

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REPORT="reports/${METHOD}_x4_generation/submit_${TS}.md"
METHOD_UPPER="$(echo "${METHOD}" | tr '[:lower:]' '[:upper:]')"
SMOKE_DIR_SUFFIX="_smoke"
if [[ -n "${SMOKE_TAG}" ]]; then
  SMOKE_DIR_SUFFIX="_smoke_${SMOKE_TAG}"
fi
cat > "${REPORT}" <<EOF
# ${METHOD_UPPER} ×4 Generation Submit

- Submitted at: ${TS}
- METHOD (validated): \`${METHOD}\`
- SMOKE_TAG: \`${SMOKE_TAG:-}\`
- Line A smoke: **${SMOKE_A}** → full array: **${FULL_A}**
- Line B smoke: **${SMOKE_B}** → full array: **${FULL_B}**
- Smoke output:
  - datasets/modelnet40_original_up/${METHOD}_x4${SMOKE_DIR_SUFFIX}
  - datasets/modelnet40_downsampled50_up/${METHOD}_x4${SMOKE_DIR_SUFFIX}
- Full output:
  - datasets/modelnet40_original_up/${METHOD}_x4
  - datasets/modelnet40_downsampled50_up/${METHOD}_x4
- Export: explicit \`METHOD=${METHOD},LINE=...\` (no --export=ALL)
EOF

echo "Report: ${REPORT}"

cat > "${JOB_ENV}" <<EOF
METHOD=${METHOD}
SMOKE_TAG=${SMOKE_TAG}
SMOKE_A=${SMOKE_A}
SMOKE_B=${SMOKE_B}
FULL_A=${FULL_A}
FULL_B=${FULL_B}
SUBMITTED_AT=${TS}
EOF
