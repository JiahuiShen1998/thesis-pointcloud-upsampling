#!/usr/bin/env bash
set -euo pipefail

# Official PU-Net is TensorFlow 1.x and Python 2.7 oriented.
# This script records the exact setup sequence recommended by the upstream repo.

PUNET_ROOT="${PUNET_ROOT:-/home/ra87racy/projects/upsampling/PU-Net}"
PUNET_ENV_NAME="${PUNET_ENV_NAME:-punet_tf13}"

echo "PU-Net root: ${PUNET_ROOT}"
echo "Target conda env: ${PUNET_ENV_NAME}"

cat <<'EOF'
Recommended official setup sequence:

1. Create a legacy Python 2.7 environment.
2. Install TensorFlow 1.3 in that environment.
3. Install the helper packages required by the PU-Net codebase.
4. Compile the TF ops under code/tf_ops.
5. Download the pretrained model into model/.
6. Run:
     cd code
     python main.py --phase test --log_dir ../model/generator2_new6

Notes:
- The upstream repository says the code was tested with TF1.3 and Python 2.7.
- If your TensorFlow include/library paths differ, the TF op compile scripts must be updated.
- The PU-Net wrapper in PointRCNN can be driven from the Python 3 environment, but the official PU-Net code should run in the legacy TensorFlow environment.
EOF

echo
echo "Example conda commands:"
cat <<EOF
conda create -y -n ${PUNET_ENV_NAME} python=2.7
source activate ${PUNET_ENV_NAME}
pip install tensorflow==1.3.0 numpy h5py scipy matplotlib tqdm
cd ${PUNET_ROOT}/code/tf_ops
bash compile.sh
EOF

echo
echo "After the env exists, use this wrapper from the PointRCNN repo:"
cat <<EOF
python scripts/punet_kitti_adapter.py \\
  --kitti_root /home/ra87racy/projects/baseline_detectors/PointRCNN/data/KITTI/object/training \\
  --split_file /home/ra87racy/projects/baseline_detectors/PointRCNN/data/KITTI/ImageSets/val.txt \\
  --punet_root ${PUNET_ROOT} \\
  --punet_python /path/to/${PUNET_ENV_NAME}/bin/python \\
  --punet_log_dir ${PUNET_ROOT}/model/generator2_new6 \\
  --output_subdir velodyne_punet_x2 \\
  --up_ratio 2
EOF
