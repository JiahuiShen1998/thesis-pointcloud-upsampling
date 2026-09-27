# ModelNet40 PointNet++ PU-EdgeFormer Full Training Audit

- Generated: `2026-07-17 04:10:04 UTC`
- Job IDs: Line B=`1749441`, Line A=`1749442`
- Overall: **PASS**

## Safety

- POINTNET_FULL_TRAINING_STARTED=YES
- DETECTOR_EVAL_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO

## Summary

| Line | job | state | exit | runtime | node | batch shape | best overall | status |
|---|---|---|---|---|---|---|---:|---|
| B | 1749441 | COMPLETED | 0:0 | 05:19:01 | tg060 | `(24, 3, 1024)` | 89.32% | **PASS** |
| A | 1749442 | COMPLETED | 0:0 | 08:34:08 | tg065 | `(24, 3, 4096)` | 90.54% | **PASS** |

## Line B detail

- Config: `configs/pointnet2_x4_two_line/lineB_downsampled_x4_up_pu_edgeformer_1024.yaml`
- Input: `pointnet2_inputs/lineB_downsampled_x4_up/pu_edgeformer`
- Expected points: 1024; actual points: 1024
- Batch shape: `(24, 3, 1024)`
- First loss: 3.753627
- Final epoch: 200; best epoch: 194
- Best overall / class: 0.8932038834951457 / 0.8521376262626262
- Final test overall / class: 0.8857605177993527 / 0.8403320707070707
- metrics.json: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/pu_edgeformer/metrics.json`
- Log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_full/lineB_pu_edgeformer_1024/slurm_1749441.out`
- no NaN / OOM / CUDA error: True / True / True
- Status: **PASS**

## Line A detail

- Config: `configs/pointnet2_x4_two_line/lineA_original_up_pu_edgeformer_4096.yaml`
- Input: `pointnet2_inputs/lineA_original_up/pu_edgeformer`
- Expected points: 4096; actual points: 4096
- Batch shape: `(24, 3, 4096)`
- First loss: 3.753759
- Final epoch: 200; best epoch: 83
- Best overall / class: 0.90542071197411 / 0.8634522005772005
- Final test overall / class: 0.901294498381877 / 0.8640939754689756
- metrics.json: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/lineA_original_up/pu_edgeformer/metrics.json`
- Log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_full/lineA_pu_edgeformer_4096/slurm_1749442.out`
- no NaN / OOM / CUDA error: True / True / True
- Status: **PASS**
