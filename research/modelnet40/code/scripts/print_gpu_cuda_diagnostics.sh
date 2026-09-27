#!/usr/bin/env bash
# Print CUDA / Python diagnostics for upsampling smoke jobs (run on GPU node).
set -euo pipefail

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "TF_LIB_DIR=${TF_LIB_DIR:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
which python || true
python -V || true
which nvcc || true
nvcc --version 2>/dev/null | head -n 4 || true
nvidia-smi || true

if python -c "import torch" 2>/dev/null; then
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
else
  echo "torch: not installed (ok for TF-only methods punet/pugcn)"
fi

if python -c "import tensorflow" 2>/dev/null; then
  python - <<'PY'
import glob
import os
import tensorflow as tf
print("tf:", tf.__version__)
root = tf.sysconfig.get_lib()
print("tf sysconfig lib:", root)
print("framework libs:", glob.glob(root + "/libtensorflow_framework*"))
PY
fi
