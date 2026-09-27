# ModelNet40 PointNet++ Line A Smoke Job IDs

- Submit time: **2026-07-07 20:10:18 UTC**
- Submit command: `sbatch.tinygpu`
- Cluster: tinygpu
- Note: duplicate accidental submission `1733480` (baseline) was cancelled immediately.

| branch name | sbatch path | job id | expected point count | expected input path |
| --- | --- | ---: | ---: | --- |
| lineA_original_baseline_1024 | `jobs/pointnet2_smoke/smoke_lineA_original_baseline_1024.sbatch` | 1733475 | 1024 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_baseline` |
| lineA_ear_4096 | `jobs/pointnet2_smoke/smoke_lineA_ear_4096.sbatch` | 1733476 | 4096 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/ear` |
| lineA_pdans_4096 | `jobs/pointnet2_smoke/smoke_lineA_pdans_4096.sbatch` | 1733477 | 4096 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pdans` |
| lineA_punet_4096 | `jobs/pointnet2_smoke/smoke_lineA_punet_4096.sbatch` | 1733478 | 4096 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pu_net` |
| lineA_pugcn_4096 | `jobs/pointnet2_smoke/smoke_lineA_pugcn_4096.sbatch` | 1733479 | 4096 | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs/lineA_original_up/pu_gcn` |

## Monitoring

```bash
watch -n 60 '
echo "===== PointNet++ Line A smoke jobs ====="
squeue -u $USER | grep -Ei "pointnet|smoke|lineA|pn2smk" || true
'
```

```bash
sacct -j 1733475,1733476,1733477,1733478,1733479 --format=JobID%25,JobName%45,State,Elapsed,ExitCode
```
