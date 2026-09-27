# ModelNet40 PointNet++ Smoke Plan

- Generated at: 2026-07-07 07:29:07 UTC
- Scope: prepare smoke jobs only; no full training launched here.
- Each smoke job runs 1 epoch with 2 train batches and 2 eval batches.
- Each smoke job uses branch-specific `--num-point` and `--no-allow-resample`.

## Smoke jobs

| branch | points | input | sbatch | smoke output dir |
| --- | ---: | --- | --- | --- |
| lineB_downsampled_x4_baseline | 256 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_baseline` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_downsampled_x4_baseline_256.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineB_downsampled_x4_baseline` |
| lineB_downsampled_x4_up_ear | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/ear` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_ear_1024.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineB_downsampled_x4_up/ear` |
| lineB_downsampled_x4_up_pdans | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pdans` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_pdans_1024.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineB_downsampled_x4_up/pdans` |
| lineB_downsampled_x4_up_pu_net | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pu_net` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_punet_1024.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineB_downsampled_x4_up/pu_net` |
| lineB_downsampled_x4_up_pu_gcn | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pu_gcn` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_pugcn_1024.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineB_downsampled_x4_up/pu_gcn` |
| lineA_original_baseline | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_baseline` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_original_baseline_1024.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_baseline` |
| lineA_original_up_ear | 4096 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/ear` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_ear_4096.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_up/ear` |
| lineA_original_up_pdans | 4096 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pdans` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_pdans_4096.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_up/pdans` |
| lineA_original_up_pu_net | 4096 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pu_net` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_punet_4096.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_up/pu_net` |
| lineA_original_up_pu_gcn | 4096 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pu_gcn` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_pugcn_4096.sbatch` | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_up/pu_gcn` |

## What each smoke verifies

- Branch name and input path are echoed at job start.
- Actual batch shape and actual point count are logged by `scripts/train_pointnet2.py` on the first batch.
- Forward/backward pass and loss computation run successfully.
- Evaluation loop runs successfully.
- Outputs land under `pointnet2_results/x4_two_line_smoke/` and do not overwrite full-training results.

## Submission status

- Not submitted in this preparation pass.
- Safe next step: submit the five Line B smoke jobs first, inspect logs, then submit the five Line A smoke jobs.
