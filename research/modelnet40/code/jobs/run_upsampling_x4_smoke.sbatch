#!/bin/bash
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/%x_%A.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/%x_%A.err

# Parameterized ×4 smoke job.
# Submit example:
#   sbatch --job-name=punet_x4_a_smoke --export=ALL,METHOD=punet,LINE=A jobs/run_upsampling_x4_smoke.sbatch
set -euo pipefail

: "${METHOD:?METHOD must be set (pdans|punet|pugcn)}"
: "${LINE:?LINE must be set (A|B)}"
SMOKE_TAG="${SMOKE_TAG:-}"

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
# Re-validate METHOD inside job (strip stray brace).
METHOD="$("${PROJECT_ROOT}/scripts/validate_upsampling_method.sh" "${METHOD}")"
echo "Validated METHOD=${METHOD} LINE=${LINE} SMOKE_TAG=${SMOKE_TAG}"
LOG_DIR="${PROJECT_ROOT}/logs/${METHOD}_x4_generation"
mkdir -p "${LOG_DIR}"

echo "=== ${METHOD} ×4 smoke line ${LINE} started: $(date) ==="
cd "${PROJECT_ROOT}"

source scripts/activate_upsampling_env.sh "${METHOD}"

echo "=== GPU/CUDA diagnostics (${METHOD}) ==="
bash scripts/print_gpu_cuda_diagnostics.sh
echo "=== end diagnostics ==="

if [[ "${METHOD}" == "punet" || "${METHOD}" == "pugcn" ]]; then
  ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated"
  if [[ "${METHOD}" == "punet" ]]; then
  cd "${ROOT}/PU-Net/code"
  else
  cd "${ROOT}/PU-GCN"
  fi
  if ! python -c "from tf_ops.sampling.tf_sampling import farthest_point_sample" 2>/dev/null \
     && ! python -c "from tf_ops.grouping.tf_grouping import knn_point_2" 2>/dev/null; then
    echo "Recompiling TF ops for ${METHOD}..."
    cd "${PROJECT_ROOT}"
    bash scripts/recompile_tf_ops.sh "${METHOD}"
  fi
  cd "${PROJECT_ROOT}"
fi

SMOKE_ARGS=(--method "${METHOD}" --line "${LINE}")
if [[ -n "${SMOKE_TAG}" ]]; then
  SMOKE_ARGS+=(--smoke-tag "${SMOKE_TAG}")
fi

python scripts/run_upsampling_x4_smoke.py "${SMOKE_ARGS[@]}"
SMOKE_RC=$?

AUDIT_ARGS=(--method "${METHOD}" --line "${LINE}")
if [[ -n "${SMOKE_TAG}" ]]; then
  AUDIT_ARGS+=(--smoke-tag "${SMOKE_TAG}")
fi
python scripts/audit_upsampling_x4_smoke.py "${AUDIT_ARGS[@]}"
AUDIT_RC=$?

if [[ ${SMOKE_RC} -ne 0 || ${AUDIT_RC} -ne 0 ]]; then
  echo "SMOKE/AUDIT failed: smoke=${SMOKE_RC} audit=${AUDIT_RC}"
  exit 1
fi

echo "=== ${METHOD} ×4 smoke line ${LINE} PASS: $(date) ==="
