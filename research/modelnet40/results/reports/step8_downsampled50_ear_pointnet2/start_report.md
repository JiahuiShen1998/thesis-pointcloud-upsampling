# Step 8 — Downsampled50 + EAR → PointNet++ Start Report

- Generated at: 2026-06-22
- Status: **SUBMITTED**
- Job ID: **1711437**
- Job name: `p2_ear_b`
- Cluster: tinygpu

## Experiment Goal

This is **not** a controlled point-count experiment.

- Downsampled50 baseline retains its native **512** points (on disk).
- Downsampled50 + EAR retains EAR upsampled **1024** points (on disk).
- We do **not** crop EAR output back to 512.
- We do **not** pad downsampled50 baseline to 1024 for this comparison.

**Comparison target:** downstream PointNet++ classification improvement after EAR upsampling on Line B (downsampled50 input).

## Dataset

| Field | Value |
| --- | --- |
| Experiment name | `downsampled50_ear_pointnet2` |
| Symlink path | `datasets/modelnet40_downsampled50_ear` → `modelnet40_downsampled50_up/ear` |
| Is symlink | Yes |
| Train samples | 9843 |
| Test samples | 2468 |
| Native points | 1024 (all samples) |
| NaN/Inf | none |

## PointNet++ Configuration

| Field | Value |
| --- | --- |
| `num_point` | 1024 (matches native EAR output) |
| `allow_resample` | false (no silent crop/pad) |
| Model | `pointnet2_cls_ssg` |
| Epochs | 200 |
| Batch size | 24 |
| LR | 0.001 |
| Optimizer | Adam |
| Seed | 42 |

## Paths

| Artifact | Path |
| --- | --- |
| Config | `configs/downsampled50_ear_pointnet2.yaml` |
| Train script | `scripts/train_pointnet2.py` |
| Train wrapper | `scripts/train_pointnet2.sh` |
| Sbatch | `jobs/train_downsampled50_ear_pointnet2_tinygpu.sbatch` |
| Output | `outputs/step8_downsampled50_ear_pointnet2/` |
| Log (wrapper) | `logs/step8_downsampled50_ear_pointnet2/train.log` |
| Slurm stdout | `logs/step8_downsampled50_ear_pointnet2/train_1711437.log` |
| Slurm stderr | `logs/step8_downsampled50_ear_pointnet2/train_1711437.err` |
| Reports | `reports/step8_downsampled50_ear_pointnet2/` |
| Point audit | `reports/step8_point_count_audit.md` |
| DataLoader audit | `reports/step8_downsampled50_ear_pointnet2/dataloader_audit.md` |

## Dry-Run Result

**PASSED**

```
dataset path: datasets/modelnet40_downsampled50_ear (symlink)
train count: 9843
test count: 2468
batch shape: (4, 1024, 3)
point count per sample: 1024
label shape: (4,)
num classes: 40
allow_resample: False
finite points: True
```

## Monitor Training

```bash
# Queue status
squeue.tinygpu -u $USER

# Live log (after job starts)
tail -f /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/step8_downsampled50_ear_pointnet2/train_1711437.log

# Training log from Python script
tail -f /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/step8_downsampled50_ear_pointnet2/train.log
```

## Comparison Reference (from point audit)

| Dataset | Native pts | Step 8 `num_point` |
| --- | ---: | ---: |
| original_baseline | 1024 | 1024 |
| downsampled50_baseline | 512 | 512 (native; Step 6 used 1024 with resample) |
| downsampled50_ear | 1024 | 1024 |

After training completes, compare `downsampled50_ear_pointnet2` test accuracy against `downsampled50_baseline` to measure classification gain from EAR upsampling.
