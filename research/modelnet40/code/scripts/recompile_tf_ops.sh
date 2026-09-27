#!/usr/bin/env bash
# Recompile TensorFlow custom ops for PU-Net / PU-GCN (run once on GPU node).
set -euo pipefail

METHOD="${1:-}"
if [[ "${METHOD}" != "punet" && "${METHOD}" != "pugcn" ]]; then
  echo "Usage: $0 {punet|pugcn}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=activate_upsampling_env.sh
source "${SCRIPT_DIR}/activate_upsampling_env.sh" "${METHOD}"

ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated"
if [[ "${METHOD}" == "punet" ]]; then
  cd "${ROOT}/PU-Net/code/tf_ops"
  bash compile.sh
elif [[ "${METHOD}" == "pugcn" ]]; then
  cd "${ROOT}/PU-GCN/tf_ops"
  bash compile.sh linux
fi

echo "TF ops compile finished for ${METHOD}"
