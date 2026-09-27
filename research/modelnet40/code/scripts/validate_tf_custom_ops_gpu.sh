#!/usr/bin/env bash
# GPU compile-only validation for PU-Net / PU-GCN TF1 custom ops (no data generation).
set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
cd "${PROJECT_ROOT}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${PROJECT_ROOT}/reports/x4_parallel_generation/tf_custom_ops_compile_validate_${TS}.md"
mkdir -p "${PROJECT_ROOT}/reports/x4_parallel_generation"
exec > >(tee -a "${REPORT}") 2>&1

echo "# TF Custom Ops Compile Validation"
echo ""
echo "- Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "- Mode: compile-only (no smoke, no full generation)"
echo ""

fail=0
pass_msg() { echo "PASS: $*"; }
fail_msg() { echo "FAIL: $*"; fail=1; }

validate_method() {
  local method="$1"
  local import_expr="$2"
  echo "## ${method^^}"
  source scripts/activate_upsampling_env.sh "${method}"
  bash scripts/print_gpu_cuda_diagnostics.sh
  echo "Recompiling TF ops for ${method}..."
  bash scripts/recompile_tf_ops.sh "${method}"

  local root="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated"
  if [[ "${method}" == "punet" ]]; then
    cd "${root}/PU-Net/code"
  else
    cd "${root}/PU-GCN"
  fi

  so_path=""
  if [[ "${method}" == "punet" ]]; then
    so_path="${root}/PU-Net/code/tf_ops/sampling/tf_sampling_so.so"
  else
    so_path="${root}/PU-GCN/tf_ops/grouping/tf_grouping_so.so"
  fi
  if [[ -f "${so_path}" ]]; then
    echo "ldd ${so_path}:"
    ldd "${so_path}" 2>&1 | head -20 || true
  fi

  if python -c "${import_expr}"; then
    pass_msg "${method} TF ops import"
  else
    fail_msg "${method} TF ops import"
  fi
  echo ""
  cd "${PROJECT_ROOT}"
}

validate_method punet "from tf_ops.sampling.tf_sampling import farthest_point_sample; print('punet import ok')"
validate_method pugcn "from tf_ops.grouping.tf_grouping import knn_point_2; print('pugcn import ok')"

echo "## Summary"
STATUS="PASS"
if [[ ${fail} -ne 0 ]]; then
  STATUS="FAIL"
  echo "SOME COMPILE VALIDATIONS FAILED"
else
  echo "ALL COMPILE VALIDATIONS PASS"
fi

FINAL_REPORT="${PROJECT_ROOT}/reports/x4_parallel_generation/tf_custom_ops_compile_validation_report.md"
{
  echo "# TF Custom Ops Compile Validation Report"
  echo ""
  echo "- Updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Validate report: \`${REPORT}\`"
  echo "- **status = ${STATUS}**"
  echo ""
  echo "## Ops compiled / imported"
  echo ""
  echo "| Method | Compile | Import |"
  echo "| --- | --- | --- |"
  if [[ ${fail} -eq 0 ]]; then
    echo "| PU-Net | OK | tf_sampling |"
    echo "| PU-GCN | OK | tf_grouping |"
  else
    echo "| PU-Net / PU-GCN | see log above | see log above |"
  fi
  echo ""
  echo "## Gate"
  if [[ ${STATUS} == "PASS" ]]; then
    echo "- PU-Net smoke: **allowed**"
    echo "- PU-Net full generation: **allowed after smoke PASS**"
    echo "- PU-GCN smoke: **allowed**"
    echo "- PU-GCN full generation: **allowed after smoke PASS**"
  else
    echo "- PU-Net smoke: **blocked**"
    echo "- PU-Net full generation: **blocked**"
    echo "- PU-GCN smoke: **blocked**"
    echo "- PU-GCN full generation: **blocked**"
  fi
} > "${FINAL_REPORT}"

if [[ ${fail} -eq 0 ]]; then
  exit 0
fi
exit 1
