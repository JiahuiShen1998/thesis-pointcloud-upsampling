# original_ear_x4_pointnet2 — Training Start Report

- Submitted at: 2026-06-29
- Dry-run: **PASS**

## Experiment

| Field | Value |
| --- | --- |
| experiment name | `original_ear_x4_pointnet2` |
| line | A (Original + EAR ×4) |
| protocol | 1024 → 4096 (×4 upsampling) |
| config | `configs/original_ear_x4_pointnet2.yaml` |

## Dataset

| Field | Value |
| --- | --- |
| dataset path | `datasets/modelnet40_original_up/ear_x4` |
| train count | **9843** |
| test count | **2468** |
| num_point | **4096** |
| allow_resample | **false** |

## Dry-run batch shape

| Format | Shape |
| --- | --- |
| DataLoader (B × N × 3) | `(4, 4096, 3)` |
| Model input (B × 3 × N) | `(4, 3, 4096)` |
| label shape | `(4,)` |
| num_classes | **40** |
| NaN/Inf | **none** |

## Slurm

| Field | Value |
| --- | --- |
| sbatch | `jobs/train_original_ear_x4_pointnet2_tinygpu.sbatch` |
| job name | `p2_ear4_a` |
| job id | **1722408** |
| log path | `logs/original_ear_x4_pointnet2/train_1722408.log` |
| stderr | `logs/original_ear_x4_pointnet2/train_1722408.err` |

## Outputs

| Field | Value |
| --- | --- |
| output dir | `outputs/original_ear_x4_pointnet2/` |
| checkpoint pattern | `outputs/original_ear_x4_pointnet2/checkpoints/best_model.pth` |
| reports dir | `reports/original_ear_x4_pointnet2/` |

## Baseline comparison target

Compare against **original_baseline** (native 1024 pts):

- best test acc = **91.95%** (epoch 85)
- checkpoint: `outputs/original_baseline/checkpoints/best_model.pth`

## Prerequisites verified

- EAR ×4 full generation audit: **PASS** (`reports/ear_x4_full_generation_audit.md`)
- Upsampling ratio audit: **PASS** (4096×3, ratio 4.0)
