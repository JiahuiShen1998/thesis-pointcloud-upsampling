# Step 8 — Point Count Audit

- Generated at: 2026-06-22 11:00:47 UTC

## Purpose

Record **native on-disk point counts** for each dataset variant.
This experiment does **not** enforce equal point counts across variants.

| dataset | train | test | min | max | mean | sample shape | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| original_baseline | 9843 | 2468 | 1024 | 1024 | 1024.0 | (1024, 3):12311 | no (0) |
| downsampled50_baseline | 9843 | 2468 | 512 | 512 | 512.0 | (512, 3):12311 | no (0) |
| downsampled50_ear | 9843 | 2468 | 1024 | 1024 | 1024.0 | (1024, 3):12311 | no (0) |

## Per-Dataset Detail

### original_baseline

- Path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original`
- Train: min=1024, max=1024, mean=1024.0
- Test: min=1024, max=1024, mean=1024.0

### downsampled50_baseline

- Path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled50`
- Train: min=512, max=512, mean=512.0
- Test: min=512, max=512, mean=512.0

### downsampled50_ear

- Path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled50_up/ear`
- Train: min=1024, max=1024, mean=1024.0
- Test: min=1024, max=1024, mean=1024.0

## Experiment Point-Count Policy (Step 8)

- `original_baseline`: native 1024 points → `num_points=1024`
- `downsampled50_baseline`: native 512 points → `num_points=512` (no upsample to match EAR)
- `downsampled50_ear`: native 1024 points (EAR upsampled) → `num_points=1024`
- Do **not** crop EAR output back to 512.
- Do **not** pad downsampled50 baseline to 1024 for this comparison.
