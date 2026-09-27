#!/bin/bash
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=tf_ops_validate
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/tf_custom_ops_validate_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/tf_custom_ops_validate_%j.err

# GPU compile-only validation for PU-Net / PU-GCN TF custom ops.
# Does NOT run smoke or full generation.
#
# Submit:
#   sbatch.tinygpu jobs/validate_tf_custom_ops_gpu.sbatch

set -euo pipefail
PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "=== TF ops validate preflight ==="
# shellcheck source=scripts/setup_cuda_toolkit_env.sh
source scripts/setup_cuda_toolkit_env.sh
which nvcc || true
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

eval "$(conda shell.bash hook)"
export CONDA_PKGS_DIRS="${HOME}/.conda/pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"
conda activate "${HOME}/.conda/envs/tf15_upsampling"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# shellcheck source=scripts/setup_tf_framework_link.sh
source scripts/setup_tf_framework_link.sh
# shellcheck source=scripts/setup_tf_cuda10_runtime_compat.sh
source scripts/setup_tf_cuda10_runtime_compat.sh

which python
python -V
python -c "import tensorflow as tf; print('tf:', tf.__version__, tf.__file__)"
echo "=== end preflight ==="

bash scripts/validate_tf_custom_ops_gpu.sh
