#!/usr/bin/env bash
# Ensure libtensorflow_framework.so linker name exists and export TF lib paths.
set -euo pipefail

TF_LIB_DIR="$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')"
if [[ -f "${TF_LIB_DIR}/libtensorflow_framework.so.1" && ! -e "${TF_LIB_DIR}/libtensorflow_framework.so" ]]; then
  ln -sf libtensorflow_framework.so.1 "${TF_LIB_DIR}/libtensorflow_framework.so"
fi

export TF_LIB_DIR
export LD_LIBRARY_PATH="${TF_LIB_DIR}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
