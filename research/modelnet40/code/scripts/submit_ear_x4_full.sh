#!/usr/bin/env bash
# Submit EAR ×4 full generation array jobs (Line A + Line B).
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"

mkdir -p logs/ear_x4_generation reports/ear_x4_generation
mkdir -p datasets/modelnet40_original_up/ear_x4
mkdir -p datasets/modelnet40_downsampled50_up/ear_x4

JOB_A=$(sbatch --parsable jobs/run_ear_x4_lineA_cpu_array.sbatch)
JOB_B=$(sbatch --parsable jobs/run_ear_x4_lineB_cpu_array.sbatch)

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REPORT="reports/ear_x4_generation/submit_${TS}.md"
cat > "${REPORT}" <<EOF
# EAR ×4 Full Generation Submit

- Submitted at: ${TS}
- Line A job array id: **${JOB_A}** (job name: ear_x4_a)
- Line B job array id: **${JOB_B}** (job name: ear_x4_b)
- Output dirs:
  - datasets/modelnet40_original_up/ear_x4
  - datasets/modelnet40_downsampled50_up/ear_x4
- Logs: logs/ear_x4_generation/
- Post-completion audit: python scripts/audit_ear_x4_full.py
EOF

echo "Submitted ear_x4_a array: ${JOB_A}"
echo "Submitted ear_x4_b array: ${JOB_B}"
echo "Report: ${REPORT}"
