# Step 8 — DataLoader Resample / Crop / Padding Audit

- Generated at: 2026-06-22

## `ModelNetNPYDataset` (`scripts/modelnet_npy_dataloader.py`)

| Behavior | Condition | Effect |
| --- | --- | --- |
| **Pass-through** | `points.shape[0] == num_points` | No change; returns native on-disk count |
| **Random resample** | `allow_resample=True` and count ≠ num_points | `resample_points()`: downsample without replacement if N > target; upsample with replacement if N < target |
| **Hard fail** | `allow_resample=False` and count ≠ num_points | Raises `ValueError` — no silent crop/pad |

## This Experiment (`downsampled50_ear_pointnet2`)

| Setting | Value |
| --- | --- |
| `num_points` | 1024 |
| `allow_resample` | **false** |
| On-disk EAR count | 1024 (all samples) |
| Loader output | `(B, 1024, 3)` — matches native, no resampling |

EAR output is **not** cropped back to 512.

## Training-Time Augmentation (`provider` in PointNet++ repo)

These operate on already-loaded point clouds and **do not change point count**:

- `random_point_dropout` — zeroes random points (keeps tensor shape)
- `random_scale_point_cloud` — per-axis scale
- `shift_point_cloud` — translation jitter

## Comparison Variants (reference)

| Variant | Native pts | `num_points` | `allow_resample` | Loader behavior |
| --- | ---: | ---: | --- | --- |
| original_baseline | 1024 | 1024 | true | pass-through |
| downsampled50_baseline (Step 6) | 512 | 1024 | true | upsample 512→1024 with fixed seed |
| **downsampled50_ear_pointnet2 (Step 8)** | **1024** | **1024** | **false** | **pass-through** |

Note: Step 6 downsampled50_baseline used 512→1024 resampling for architecture parity with original.
Step 8 EAR experiment uses native 1024 and does not force point-count matching with baseline.
