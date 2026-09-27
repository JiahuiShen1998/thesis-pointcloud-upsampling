# Step 6 — Downsampled50 Baseline Preparation (NOT SUBMITTED)

- Status: **READY TO SUBMIT** (waiting for Step 4 completion)
- Generated at: 2026-06-20

## Experiment Definition

| Field | Value |
| --- | --- |
| experiment name | `downsampled50_baseline` |
| line | B |
| dataset | `datasets/modelnet40_downsampled50` |
| raw point count on disk | 512 |
| PointNet++ input points | 1024 |
| resample strategy | random with replacement, fixed md5-derived seed per sample |
| model | `pointnet2_cls_ssg` |
| epoch | 200 |
| batch_size | 24 |
| learning_rate | 0.001 |
| optimizer | Adam |
| seed | 42 |

## Why 512 → 1024?

Downsampled50 files remain `(512, 3)` on disk. The DataLoader resamples to `(1024, 3)` before training so architecture and `num_point` match Original baseline exactly.

## Paths

| Artifact | Path |
| --- | --- |
| sbatch | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/train_downsampled50_baseline_gpu.sbatch` |
| train wrapper | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/scripts/train_pointnet2.sh` |
| dry-run | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/scripts/dryrun_downsampled50_baseline.py` |
| output dir | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/outputs/downsampled50_baseline` |
| log | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/downsampled50_baseline.log` |
| result csv | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/downsampled50_baseline_result.csv` |
| result md | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/downsampled50_baseline_result.md` |

## Submit Command (after Step 4 finishes)

```bash
sbatch.tinygpu /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/train_downsampled50_baseline_gpu.sbatch
```

## Do NOT submit while Job 1709666 is running
