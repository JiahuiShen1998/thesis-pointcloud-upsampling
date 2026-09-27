# ModelNet40 PointNet++ Line B Full Training Job IDs

- Submit time: **2026-07-07 13:04:28 UTC**
- Submit command: `sbatch.tinygpu`
- Prerequisite: `LINE_B_SMOKE_ALL_PASS` (5/5)
- Training: 200 epochs, seed=42, `--no-allow-resample`

| branch | method | sbatch path | job id | expected pts | input path | output path | note |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| lineB_downsampled_x4_baseline_256 | baseline | `jobs/pointnet2_full/run_lineB_downsampled_x4_baseline_256.sbatch` | **1733191** | 256 | `pointnet2_inputs/lineB_downsampled_x4_baseline` | `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_baseline` | formal full training |
| lineB_ear_1024 | EAR | `jobs/pointnet2_full/run_lineB_ear_1024.sbatch` | **1733192** | 1024 | `pointnet2_inputs/lineB_downsampled_x4_up/ear` | `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/ear` | formal full training |
| lineB_pdans_1024 | PDANS | `jobs/pointnet2_full/run_lineB_pdans_1024.sbatch` | **1733193** | 1024 | `pointnet2_inputs/lineB_downsampled_x4_up/pdans` | `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/pdans` | formal full training |
| lineB_punet_1024 | PU-Net | `jobs/pointnet2_full/run_lineB_punet_1024.sbatch` | **1733194** | 1024 | `pointnet2_inputs/lineB_downsampled_x4_up/pu_net` | `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/pu_net` | formal full training |
| lineB_pugcn_1024 | PU-GCN | `jobs/pointnet2_full/run_lineB_pugcn_1024.sbatch` | **1733195** | 1024 | `pointnet2_inputs/lineB_downsampled_x4_up/pu_gcn` | `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/pu_gcn` | formal full training |

## Monitoring

```bash
watch -n 60 '
echo "===== PointNet++ Line B full training jobs ====="
squeue.tinygpu -u $USER 2>/dev/null | grep -Ei "pointnet|lineB|mn40|full" || squeue -u $USER | grep -Ei "pointnet|lineB|mn40|full" || true
'
```

```bash
sacct -j 1733191,1733192,1733193,1733194,1733195 --format=JobID%25,JobName%45,State,Elapsed,ExitCode
```
