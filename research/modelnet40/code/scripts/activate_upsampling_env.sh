#!/usr/bin/env bash
# Activate conda env and paths for main-protocol upsampling methods.
set -euo pipefail

METHOD="${1:-}"
if [[ -z "${METHOD}" ]]; then
  echo "Usage: $0 {pdans|punet|pugcn}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup_cuda_toolkit_env.sh
source "${SCRIPT_DIR}/setup_cuda_toolkit_env.sh"

eval "$(conda shell.bash hook)"
export CONDA_PKGS_DIRS="${HOME}/.conda/pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"

case "${METHOD}" in
  pdans)
    conda activate "${HOME}/.conda/envs/pdans_x4"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.6}"
    export PYTHONPATH="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS:/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS/pointnet2_ops_lib:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  ;;
  punet|pugcn)
    conda activate "${HOME}/.conda/envs/tf15_upsampling"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    # shellcheck source=setup_tf_framework_link.sh
    source "${SCRIPT_DIR}/setup_tf_framework_link.sh"
    # shellcheck source=setup_tf_cuda10_runtime_compat.sh
    source "${SCRIPT_DIR}/setup_tf_cuda10_runtime_compat.sh"
  ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 1
  ;;
esac

echo "Activated ${METHOD} upsampling env (CUDA_HOME=${CUDA_HOME})"
