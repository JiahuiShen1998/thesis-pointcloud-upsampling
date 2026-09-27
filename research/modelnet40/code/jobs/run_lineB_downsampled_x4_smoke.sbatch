#!/bin/bash
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/lineB_downsampled_x4_up/%x_%A.err

# Line B new protocol smoke: 256→1024
# Submit: sbatch --job-name=lineB_x4_pdans_smoke --export=ALL,METHOD=pdans jobs/run_lineB_downsampled_x4_smoke.sbatch
set -euo pipefail

: "${METHOD:?METHOD must be ear|pu_net|pdans}"

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
METHOD_KEY="${METHOD%\}}"
SMOKE_TAG="${SMOKE_TAG:-}"

echo "=== Line B x4 smoke METHOD=${METHOD_KEY} SMOKE_TAG=${SMOKE_TAG} started: $(date) ==="
cd "${PROJECT_ROOT}"
mkdir -p logs/lineB_downsampled_x4_up datasets/lineB_downsampled_x4_up_smoke

if [[ "${METHOD_KEY}" == "ear" ]]; then
  module purge 2>/dev/null || true
  module load python/pytorch2.6py3.12
else
  ALIAS="$("${PROJECT_ROOT}/scripts/validate_upsampling_method.sh" "${METHOD_KEY/pu_net/punet}")"
  source scripts/activate_upsampling_env.sh "${ALIAS}"
  bash scripts/print_gpu_cuda_diagnostics.sh
  if [[ "${ALIAS}" == "punet" ]]; then
    ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-Net/code"
    cd "${ROOT}"
    if ! python -c "from tf_ops.sampling.tf_sampling import farthest_point_sample" 2>/dev/null; then
      cd "${PROJECT_ROOT}"
      bash scripts/recompile_tf_ops.sh punet
    fi
    cd "${PROJECT_ROOT}"
  fi
fi

SMOKE_ARGS=(--method "${METHOD_KEY}")
if [[ -n "${SMOKE_TAG}" ]]; then
  SMOKE_ARGS+=(--smoke-tag "${SMOKE_TAG}")
fi

python scripts/run_lineB_downsampled_x4_smoke.py "${SMOKE_ARGS[@]}"
SMOKE_RC=$?

AUDIT_ARGS=(--method "${METHOD_KEY}")
if [[ -n "${SMOKE_TAG}" ]]; then
  AUDIT_ARGS+=(--smoke-tag "${SMOKE_TAG}")
fi
python scripts/audit_lineB_downsampled_x4_smoke.py "${AUDIT_ARGS[@]}"
AUDIT_RC=$?

if [[ ${SMOKE_RC} -ne 0 || ${AUDIT_RC} -ne 0 ]]; then
  echo "SMOKE/AUDIT failed: smoke=${SMOKE_RC} audit=${AUDIT_RC}"
  exit 1
fi

echo "=== Line B x4 smoke PASS: $(date) ==="
