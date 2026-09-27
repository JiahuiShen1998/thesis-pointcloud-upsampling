# Step 4 — Original Baseline Audit

- Generated at: 2026-06-20 17:15:14 CEST
- Experiment: `original_baseline`
- Training status: **IN_PROGRESS**

## Experiment Config

| Field | Value |
| --- | --- |
| dataset | `datasets/modelnet40_original` |
| model | `pointnet2_cls_ssg` |
| input points | 1024 |
| train samples | 9843 |
| test samples | 2468 |
| batch size | 24 |
| epochs | 200 |
| optimizer | Adam |
| learning rate | 0.001 |
| seed | 42 |

## Progress

- Completed epochs (logged): **3 / 200**
- Checkpoint exists: **True**
- Result CSV exists: **False**
- Result MD exists: **False**

## Best epoch (from training log)

- best epoch: **3**
- best overall accuracy: **74.51%**
- best class accuracy: **66.81%**

## Latest logged epoch

- epoch: **3**
- train accuracy: **69.85%**
- test overall accuracy: **74.51%**
- test class accuracy: **66.81%**

## Paths

- checkpoint: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/outputs/original_baseline/checkpoints/best_model.pth`
- training log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/original_baseline.log`
- slurm log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs` (original_baseline_*.log)
- result csv: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/original_baseline_result.csv`
- result md: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/original_baseline_result.md`

## Issues

- No OOM / NaN / traceback detected so far.

## Next step

Wait for training to finish, then re-run:

```bash
python scripts/generate_step4_audit.py
```
