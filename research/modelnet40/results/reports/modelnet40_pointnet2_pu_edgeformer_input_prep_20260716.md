# PointNet++ PU-EdgeFormer Input/Config Prep

- Generated: `2026-07-16 19:26:19 UTC`
- Scope: input symlink + config + smoke sbatch only
- POINTNET_FULL_TRAINING_STARTED=NO
- DETECTOR_EVAL_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO
- POINTNET_SMOKE_STARTED=pending_submit

- Line B config: `configs/pointnet2_x4_two_line/lineB_downsampled_x4_up_pu_edgeformer_1024.yaml`
- Line A config: `configs/pointnet2_x4_two_line/lineA_original_up_pu_edgeformer_4096.yaml`
- Line B smoke job: `jobs/pointnet2_smoke/smoke_lineB_pu_edgeformer_1024.sbatch`
- Line A smoke job: `jobs/pointnet2_smoke/smoke_lineA_pu_edgeformer_4096.sbatch`

| line | input | train | test | pts | manifests | status |
|---|---|---:|---:|---:|---|---|
| B | `pointnet2_inputs/lineB_downsampled_x4_up/pu_edgeformer` | 9843 | 2468 | 1024 | NONE (OK) | PASS |
| A | `pointnet2_inputs/lineA_original_up/pu_edgeformer` | 9843 | 2468 | 4096 | NONE (OK) | PASS |
