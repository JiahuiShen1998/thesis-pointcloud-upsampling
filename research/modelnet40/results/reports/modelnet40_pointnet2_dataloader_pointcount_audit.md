# ModelNet40 PointNet++ Dataloader Point-Count Audit

- Generated at: 2026-07-07 07:29:07 UTC

## Findings

- `scripts/train_pointnet2.py` defaults to `--num-point 1024` unless overridden at runtime.
- `scripts/modelnet_npy_dataloader.py` will resample any sample whose on-disk point count differs from `num_points` when `allow_resample=true`.
- Therefore the dataloader can silently change point counts unless each branch is launched with both the correct `--num-point` and `--no-allow-resample`.
- Legacy wrapper `scripts/train_pointnet2.sh` still contains old protocol branches (`downsampled50`, `2048`-point Line B upsampling roots) and must not be used for this x4 two-line final round.

## Runtime policy

- All 10 generated branch configs under `configs/pointnet2_x4_two_line/` set `allow_resample: false`.
- All generated smoke/full `sbatch` files call `scripts/train_pointnet2.py` directly instead of the legacy wrapper.
- Result: no branch may silently crop/pad/upsample/downsample inside the dataloader.

## Branch runtime num_points

| line | branch | method | source path | runtime num_points | allow_resample | expected on-disk points | verdict |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| B | lineB_downsampled_x4_baseline | baseline | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_baseline` | 256 | false | 256 | PASS (Baseline remains native 256-point input.) |
| B | lineB_downsampled_x4_up_ear | EAR | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/ear` | 1024 | false | 1024 | PASS (Upsampling branch preserves strict protocol output count.) |
| B | lineB_downsampled_x4_up_pdans | PDANS | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pdans` | 1024 | false | 1024 | PASS (Upsampling branch preserves strict protocol output count.) |
| B | lineB_downsampled_x4_up_pu_net | PU-Net | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pu_net` | 1024 | false | 1024 | PASS (Upsampling branch preserves strict protocol output count.) |
| B | lineB_downsampled_x4_up_pu_gcn | PU-GCN | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pu_gcn` | 1024 | false | 1024 | PASS (Upsampling branch preserves strict protocol output count.) |
| A | lineA_original_baseline | baseline | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_baseline` | 1024 | false | 1024 | PASS (Baseline remains native 1024-point input.) |
| A | lineA_original_up_ear | EAR | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/ear` | 4096 | false | 4096 | PASS (Upsampling branch preserves strict protocol output count.) |
| A | lineA_original_up_pdans | PDANS | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pdans` | 4096 | false | 4096 | PASS (Upsampling branch preserves strict protocol output count.) |
| A | lineA_original_up_pu_net | PU-Net | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pu_net` | 4096 | false | 4096 | PASS (Upsampling branch preserves strict protocol output count.) |
| A | lineA_original_up_pu_gcn | PU-GCN | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pu_gcn` | 4096 | false | 4096 | PASS (Upsampling branch preserves strict protocol output count.) |

## Key file references

- `scripts/train_pointnet2.py`
- `scripts/modelnet_npy_dataloader.py`
- `scripts/train_pointnet2.sh`
- `configs/pointnet2_default.yaml`
- `configs/pointnet2_x4_two_line/`

## Conclusion

The runtime point counts are now branch-specific and explicit: Line B baseline = 256, Line B upsampling branches = 1024, Line A baseline = 1024, Line A upsampling branches = 4096. No generated smoke/full job uses the default 1024 silently.
