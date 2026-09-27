#!/bin/bash
#SBATCH --job-name=pn2_smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_smoke_%j.log
#SBATCH --error=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_smoke_%j.err

set -euo pipefail

PROJECT_ROOT="/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"

echo "=== PointNet++ smoke test: $(date) ==="
echo "Host: $(hostname)"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
pip install --user -q tqdm

python --version
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"

cd "${PROJECT_ROOT}"

echo "--- Setup check ---"
python scripts/check_pointnet2_setup.py

echo "--- Mini train (2 train batches, 2 eval batches) ---"
python scripts/train_pointnet2.py \
  --variant original_baseline_smoke \
  --data-root datasets/modelnet40_original \
  --output-dir outputs/original_baseline_smoke \
  --log-file logs/original_baseline_smoke.log \
  --epoch 1 \
  --batch-size 8 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --num-workers 2

echo "=== Smoke test finished: $(date) ==="
