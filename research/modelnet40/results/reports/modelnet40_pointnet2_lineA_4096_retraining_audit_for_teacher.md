# Line A 4096 PointNet++ Retraining Audit (for Teacher)

- Generated: `2026-07-23 15:53:35 UTC`
- Project: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`
- Scope: Existing Line A Original+Upsampling 4096 PointNet++ full training only
- Safety: `DETECTOR_EVAL_STARTED=NO`, `KITTI_AP_EVAL_STARTED=NO`, `GEOMETRY_METRICS_STARTED=NO`
- Clean rerun started: **NO** (existing results already satisfy requirements)

## Verdict (YES/NO)

| Question | Answer |
| --- | --- |
| Existing Line A 4096 PointNet++ was trained with 4096-point input | **YES** |
| Existing Line A 4096 PointNet++ was trained from scratch | **YES** |
| Existing results can be described as 4096-point retrained PointNet++ | **YES** |
| Clean rerun required (`pointnet2_lineA_4096_retrain_clean`) | **NO** |

## Evidence summary

1. Each branch YAML sets `num_point: 4096` and `expected_point_count: 4096`.
2. Each full-training sbatch passes `--num-point 4096` and does **not** pass any pretrained/resume checkpoint.
3. `scripts/train_pointnet2.py` has **no** `torch.load` / `resume` / `pretrained` / `load_state_dict` path; the classifier is created with `get_model(...)` and trained from random initialization.
4. Every `train.log` records first batch shape `(24, 3, 4096)` and first loss ≈ 3.68–3.75 (near `ln(40)≈3.69`), consistent with from-scratch CE on 40 classes.
5. Each method finished `Epoch 200/200`, wrote `metrics.json`, and saved an independent `checkpoints/best_model.pth`.
6. No NaN / OOM / CUDA traceback found in train logs or Slurm err/out for these jobs.
7. Methods were **independently** trained (separate jobs, separate output dirs); not a single 1024 model evaluated on 4096 inputs.

## Per-method checklist

| Method | config `num_point` | batch shape | from scratch | final epoch | metrics.json | first loss finite | NaN/OOM/CUDA | independent | meets |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- |
| EAR | 4096 | `(24, 3, 4096)` | YES | 200 | YES | YES (3.682938) | none | YES | **YES** |
| PDANS | 4096 | `(24, 3, 4096)` | YES | 200 | YES | YES (3.737715) | none | YES | **YES** |
| PU-Net | 4096 | `(24, 3, 4096)` | YES | 200 | YES | YES (3.716941) | none | YES | **YES** |
| PU-GCN | 4096 | `(24, 3, 4096)` | YES | 200 | YES | YES (3.685437) | none | YES | **YES** |
| PU-EdgeFormer | 4096 | `(24, 3, 4096)` | YES | 200 | YES | YES (3.753759) | none | YES | **YES** |

## Accuracy snapshot (existing 4096-retrained results)

| Method | Best overall | Final overall | Best epoch | Job ID |
| --- | ---: | ---: | ---: | ---: |
| EAR | 91.48% | 90.85% | 61 | 1733492 |
| PDANS | 91.62% | 90.87% | 84 | 1733493 |
| PU-Net | 90.95% | 90.55% | 189 | 1733494 |
| PU-GCN | 91.63% | 91.00% | 81 | 1733495 |
| PU-EdgeFormer | 90.54% | 90.13% | 83 | 1749442 |

## Classifier setting label for final tables

Line A upsampled branches should be labeled:

> **PointNet++ retrained with 4096-point input**

Do **not** label them as “PointNet++ trained only for 1024 points”.

## Paths

- Audit CSV: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_lineA_4096_retraining_audit_for_teacher.csv`
- Teacher explanation: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_4096_adapted_explanation_for_teacher.md`

