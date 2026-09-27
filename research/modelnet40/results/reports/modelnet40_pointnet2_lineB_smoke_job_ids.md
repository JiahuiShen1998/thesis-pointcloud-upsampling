# ModelNet40 PointNet++ Line B Smoke Job IDs

- Submit time: **2026-07-07 12:25:06 UTC**
- Submit command: `sbatch.tinygpu`
- Cluster: tinygpu
- Note: duplicate accidental submission `1733164` (baseline) was cancelled immediately.

| branch name | sbatch path | job id | expected point count | expected input path |
| --- | --- | ---: | ---: | --- |
| lineB_downsampled_x4_baseline_256 | `jobs/pointnet2_smoke/smoke_lineB_downsampled_x4_baseline_256.sbatch` | 1733159 | 256 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_baseline` |
| lineB_ear_1024 | `jobs/pointnet2_smoke/smoke_lineB_ear_1024.sbatch` | 1733160 | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/ear` |
| lineB_pdans_1024 | `jobs/pointnet2_smoke/smoke_lineB_pdans_1024.sbatch` | 1733161 | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pdans` |
| lineB_punet_1024 | `jobs/pointnet2_smoke/smoke_lineB_punet_1024.sbatch` | 1733162 | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pu_net` |
| lineB_pugcn_1024 | `jobs/pointnet2_smoke/smoke_lineB_pugcn_1024.sbatch` | 1733163 | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineB_downsampled_x4_up/pu_gcn` |

## Monitoring

```bash
watch -n 60 '
echo "===== PointNet++ Line B smoke jobs ====="
squeue -u $USER | grep -Ei "pointnet|smoke|lineB|mn40" || true
'
```

```bash
sacct -j 1733159,1733160,1733161,1733162,1733163 --format=JobID%25,JobName%45,State,Elapsed,ExitCode
```
