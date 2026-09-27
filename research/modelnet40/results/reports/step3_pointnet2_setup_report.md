# Step 3 — PointNet++ Classifier Setup Report

- Generated at: 2026-06-20
- Status: **READY** (smoke test passed on GPU)

## Repository

| Item | Value |
|------|-------|
| Repo | [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) |
| Local path | `external/Pointnet_Pointnet2_pytorch/` |
| Commit | `eb64fe0b4c24055559cea26299cb485dcb43d8dd` |
| Model | `pointnet2_cls_ssg` |
| CUDA extension | **Not required** (pure PyTorch implementation) |

## Environment

| Item | Value |
|------|-------|
| HPC module | `python/pytorch2.6py3.12` |
| PyTorch | 2.6.0 |
| CUDA | 12.6 (verified on `tg06a`) |
| Extra pip | `tqdm` (`pip install --user tqdm`) |

## Custom Integration

We do **not** use the upstream `modelnet40_normal_resampled` txt format. Instead:

| Component | Path |
|-----------|------|
| DataLoader | `scripts/modelnet_npy_dataloader.py` |
| Train wrapper | `scripts/train_pointnet2.py` |
| Eval wrapper | `scripts/eval_pointnet2.py` |
| Setup smoke test | `scripts/check_pointnet2_setup.py` |
| Shell wrappers | `scripts/train_pointnet2.sh`, `scripts/eval_pointnet2.sh` |
| Unified config | `configs/pointnet2_default.yaml` |

### Data flow

```
datasets/modelnet40_original/
  train/{class}/*.npy   (1024, 3)
  test/{class}/*.npy
  metadata/train_manifest.csv
  metadata/class_to_idx.json
        ↓
ModelNetNPYDataset
        ↓
pointnet2_cls_ssg
        ↓
outputs/{variant}/checkpoints/best_model.pth
reports/{variant}_result.csv
reports/{variant}_result.md
```

## Unified Training Hyperparameters (all variants)

| Parameter | Value |
|-----------|-------|
| model | pointnet2_cls_ssg |
| num_point | 1024 |
| num_category | 40 |
| batch_size | 24 |
| epoch | 200 |
| learning_rate | 0.001 |
| decay_rate | 1e-4 |
| optimizer | Adam |
| scheduler | StepLR(step=20, gamma=0.7) |
| seed | 42 |
| use_normals | false |
| augmentation | random dropout / scale / shift (upstream provider) |

## Smoke Test (Job 1709664)

| Check | Result |
|-------|--------|
| Import PointNet++ | PASS |
| Load `.npy` batch `(8, 1024, 3)` | PASS |
| GPU forward pass | PASS |
| Mini train 2 batches + eval 2 batches | PASS |
| Checkpoint saved | PASS |

Log: `logs/pointnet2_smoke_1709664.log`

## How to Run Step 4 (Original Baseline)

```bash
# Submit full 200-epoch training
sbatch.tinygpu /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/train_original_baseline_gpu.sbatch

# Monitor
squeue.tinygpu -u $USER
tail -f /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/original_baseline.log
```

Expected outputs after Step 4:

- `outputs/original_baseline/checkpoints/best_model.pth`
- `logs/original_baseline.log`
- `reports/original_baseline_result.csv`
- `reports/original_baseline_result.md`

## Next Step

Proceed to **Step 4**: run full Original baseline training on `datasets/modelnet40_original`.
