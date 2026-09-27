#!/usr/bin/env bash
# Submit EAR full Line B array job + dependent finalize job.
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
LOG_DIR="${PROJECT_ROOT}/logs/ear_full_lineB"
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_ROOT}/datasets/modelnet40_downsampled50_up/ear"
mkdir -p "${LOG_DIR}/chunk_audits"

cd "${PROJECT_ROOT}"

ARRAY_JOB_ID=$(sbatch --parsable jobs/run_ear_full_lineB_cpu_array.sbatch)
echo "Submitted array job: ${ARRAY_JOB_ID}"

FINALIZE_JOB_ID=$(ARRAY_JOB_ID="${ARRAY_JOB_ID}" sbatch --parsable \
  --dependency=afterok:"${ARRAY_JOB_ID}" \
  jobs/run_ear_full_lineB_finalize.sbatch)
echo "Submitted finalize job: ${FINALIZE_JOB_ID} (depends on ${ARRAY_JOB_ID})"

cat > "${LOG_DIR}/job_ids.env" <<EOF
ARRAY_JOB_ID=${ARRAY_JOB_ID}
FINALIZE_JOB_ID=${FINALIZE_JOB_ID}
SUBMITTED_AT=$(date -Is)
EOF

echo "Job IDs saved to ${LOG_DIR}/job_ids.env"
