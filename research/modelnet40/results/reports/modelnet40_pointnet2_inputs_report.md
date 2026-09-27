# PointNet++ Inputs Manifest

- Generated at: 2026-07-02 19:25:37 UTC

## Dataloader note

PointNet++ `ModelNetNPYDataset` uses `num_point` and `allow_resample`.
When `allow_resample=false`, samples must match `num_point` exactly on disk.
Set `num_point` per variant: N, 4N, N/4 as listed below.

| line | branch | method | points | status | path |
| --- | --- | --- | ---: | --- | --- |
| A | baseline | original | 1024 | exists | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_baseline` |
| A | upsampling | ear | 4096 | linked | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/ear` |
| A | upsampling | pu_net | 4096 | linked | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pu_net` |
| A | upsampling | pu_gcn | 4096 | missing_source | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pu_gcn` |
| A | upsampling | pdans | 4096 | linked | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pdans` |
| B | baseline | downsampled_x4 | 256 | exists | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_baseline` |
| B | upsampling | ear | 1024 | pending | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/ear` |
| B | upsampling | pu_net | 1024 | pending | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pu_net` |
| B | upsampling | pu_gcn | 1024 | pending | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pu_gcn` |
| B | upsampling | pdans | 1024 | pending | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pdans` |