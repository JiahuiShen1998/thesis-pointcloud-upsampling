# downsampled50_ear_x4_pointnet2 — Training Start Report

- Submitted at: 2026-06-29
- Dry-run: **PASS**

## Experiment

| Field | Value |
| --- | --- |
| experiment name | `downsampled50_ear_x4_pointnet2` |
| line | B (Downsampled50 + EAR ×4) |
| protocol | 512 → 2048 (×4 upsampling) |
| config | `configs/downsampled50_ear_x4_pointnet2.yaml` |

## Dataset

| Field | Value |
| --- | --- |
| dataset path | `datasets/modelnet40_downsampled50_up/ear_x4` |
| train count | **9843** |
| test count | **2468** |
| num_point | **2048** |
| allow_resample | **false** |

## Dry-run batch shape

| Format | Shape |
| --- | --- |
| DataLoader (B × N × 3) | `(4, 2048, 3)` |
| Model input (B × 3 × N) | `(4, 3, 2048)` |
| label shape | `(4,)` |
| num_classes | **40** |
| NaN/Inf | **none** |

## Slurm

| Field | Value |
| --- | --- |
| sbatch | `jobs/train_downsampled50_ear_x4_pointnet2_tinygpu.sbatch` |
| job name | `p2_ear4_b` |
| job id | **1722409** |
| log path | `logs/downsampled50_ear_x4_pointnet2/train_1722409.log` |
| stderr | `logs/downsampled50_ear_x4_pointnet2/train_1722409.err` |

## Outputs

| Field | Value |
| --- | --- |
| output dir | `outputs/downsampled50_ear_x4_pointnet2/` |
| checkpoint pattern | `outputs/downsampled50_ear_x4_pointnet2/checkpoints/best_model.pth` |
| reports dir | `reports/downsampled50_ear_x4_pointnet2/` |

## Baseline comparison target

Compare against **downsampled50_native512_pointnet2** (native 512 pts):

- best test acc = **91.26%** (epoch 197)
- final test acc = 91.14% (epoch 200)
- checkpoint: `outputs/downsampled50_native512_pointnet2/checkpoints/best_model.pth`

## Prerequisites verified

- EAR ×4 full generation audit: **PASS** (`reports/ear_x4_full_generation_audit.md`)
- Upsampling ratio audit: **PASS** (2048×3, ratio 4.0)
