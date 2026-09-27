#!/usr/bin/env bash
# Load cluster CUDA toolkit module and export CUDA_HOME from nvcc location.
set -euo pipefail

module purge 2>/dev/null || true
module load gcc/11.5.0 cuda/11.8.0 python/3.12-conda

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found after 'module load cuda/11.8.0'" >&2
  exit 1
fi

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
