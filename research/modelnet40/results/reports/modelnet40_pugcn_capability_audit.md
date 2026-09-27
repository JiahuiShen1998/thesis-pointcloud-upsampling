# PU-GCN Capability Audit

- Generated at: 2026-07-05 18:52:51 UTC
- PROJECT_ROOT: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

## Summary

- **status:** `pending_environment`

| field | value |
| --- | --- |
| method | PU-GCN |
| code_path | /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN |
| checkpoint_path | /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN/pretrained/pu1k-pugcn |
| environment | conda:tf15_upsampling (/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/bin/python) |
| supports_x4 | yes |
| supports_lineA_1024_to_4096 | yes (up_ratio=4, num_point=1024 -> ~4096) |
| supports_lineB_256_to_1024 | yes (up_ratio=4, num_point=256 -> ~1024) |
| input_format | .npy (N,3) float32 xyz; converted to .xyz for PU-GCN test phase |
| output_format | .xyz from PU-GCN result dir -> .npy raw + strict normalized |
| existing_outputs | lineA_raw=0, lineA_strict=0, lineB_raw=0, lineB_strict=0 |
| status | pending_environment |
| note | tensorflow import failed: Command '['/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/bin/python', '-c', 'import tensorflow as tf; print(tf.__version__)']' timed out after 60 seconds; legacy outputs exist (lineA=0, lineB=0) — NOT valid for two-line protocol; GPU/CUDA runtime validation requires GPU node (activate_upsampling_env.sh pugcn) |

## Environment checks

- TensorFlow import: **fail** (Command '['/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/bin/python', '-c', 'import tensorflow as tf; print(tf.__v)
- GPU: **not_on_login_node** (nvidia-smi not found (expected on GPU node))
- Custom op .so: **present**

## Input datasets

- Line A input: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original` (1024 pts)
- Line B input: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4` (256 pts)

## Target output paths (two-line protocol)

- Line A raw: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineA_original_up/raw/pu_gcn`
- Line A strict: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineA_original_up/strict_4N/pu_gcn` (expected 4096 pts)
- Line B raw: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineB_downsampled_x4_up/raw/pu_gcn`
- Line B strict: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineB_downsampled_x4_up/strict_N/pu_gcn` (expected 1024 pts)

## Not in scope this run

- PU-EdgeFormer: **pending_checkpoint** (checkpoint not transferred)
- TULIP: **pending / not included in this run**

## Blocked

Do not submit full generation until resolved.