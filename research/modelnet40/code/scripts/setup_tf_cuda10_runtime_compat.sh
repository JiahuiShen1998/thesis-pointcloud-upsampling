#!/usr/bin/env bash
# TF 1.15 expects libcudart.so.10.0 at runtime; cluster provides CUDA 11.8.
# Provide compat symlinks + prefer conda cudatoolkit when present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPAT_DIR="${PROJECT_ROOT}/.cuda_runtime_compat"
mkdir -p "${COMPAT_DIR}"

# Prefer conda-shipped cudart (if user installed cudatoolkit=10.0).
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  for cand in \
    "${CONDA_PREFIX}/lib/libcudart.so.10.0" \
    "${CONDA_PREFIX}/lib/libcudart.so.10.2.89" \
    "${CONDA_PREFIX}/lib/libcudart.so"; do
    if [[ -e "${cand}" ]]; then
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
      echo "TF cuda compat: using conda cudart at ${CONDA_PREFIX}/lib"
      return 0 2>/dev/null || exit 0
    fi
  done
fi

# Symlink libcudart.so.10.0 -> CUDA 11.8 runtime (TF 1.15 dlopen name).
CUDART_SRC=""
for cand in \
  "${CUDA_HOME:-}/lib64/libcudart.so.11.0" \
  "${CUDA_HOME:-}/lib64/libcudart.so.11.8.89" \
  "${CUDA_HOME:-}/lib64/libcudart.so"; do
  if [[ -e "${cand}" ]]; then
    CUDART_SRC="${cand}"
    break
  fi
done

if [[ -n "${CUDART_SRC}" ]]; then
  ln -sf "${CUDART_SRC}" "${COMPAT_DIR}/libcudart.so.10.0"
  ln -sf "${CUDART_SRC}" "${COMPAT_DIR}/libcudart.so.10.2.89" 2>/dev/null || true
  export LD_LIBRARY_PATH="${COMPAT_DIR}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "TF cuda compat: symlink libcudart.so.10.0 -> ${CUDART_SRC}"
else
  echo "WARN: could not locate libcudart for TF 1.15 compat" >&2
fi
