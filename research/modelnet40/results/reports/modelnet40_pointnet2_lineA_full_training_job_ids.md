# ModelNet40 PointNet++ Line A Full Training Job IDs

- Submit time: **2026-07-07 20:27:03 UTC**
- Submit command: `sbatch.tinygpu`
- Cluster: tinygpu
- Prerequisite: **LINE_A_SMOKE_ALL_PASS** (`reports/modelnet40_pointnet2_lineA_smoke_final_status.md`)

| branch | method | sbatch path | job id | expected point count | input path | output path |
| --- | --- | --- | ---: | ---: | --- | --- |
| lineA_original_baseline_1024 | baseline | `jobs/pointnet2_full/run_lineA_original_baseline_1024.sbatch` | **1733491** | 1024 | `pointnet2_inputs/lineA_original_baseline` | `pointnet2_results/x4_two_line_final/lineA_original_baseline` |
| lineA_ear_4096 | EAR | `jobs/pointnet2_full/run_lineA_ear_4096.sbatch` | **1733492** | 4096 | `pointnet2_inputs/lineA_original_up/ear` | `pointnet2_results/x4_two_line_final/lineA_original_up/ear` |
| lineA_pdans_4096 | PDANS | `jobs/pointnet2_full/run_lineA_pdans_4096.sbatch` | **1733493** | 4096 | `pointnet2_inputs/lineA_original_up/pdans` | `pointnet2_results/x4_two_line_final/lineA_original_up/pdans` |
| lineA_punet_4096 | PU-Net | `jobs/pointnet2_full/run_lineA_punet_4096.sbatch` | **1733494** | 4096 | `pointnet2_inputs/lineA_original_up/pu_net` | `pointnet2_results/x4_two_line_final/lineA_original_up/pu_net` |
| lineA_pugcn_4096 | PU-GCN | `jobs/pointnet2_full/run_lineA_pugcn_4096.sbatch` | **1733495** | 4096 | `pointnet2_inputs/lineA_original_up/pu_gcn` | `pointnet2_results/x4_two_line_final/lineA_original_up/pu_gcn` |

## Note

- Full training: 200 epochs, batch size 24 (baseline 1024 pts; upsampling 4096 pts).
- Do not mix smoke metrics with these final results.
- Line B full training already complete — do not re-submit.

## Monitoring

```bash
watch -n 60 '
echo "===== PointNet++ Line A full training jobs ====="
squeue.tinygpu -u $USER 2>/dev/null | grep -Ei "pointnet|lineA|mn40|full" || squeue -u $USER | grep -Ei "pointnet|lineA|mn40|full" || true
'
```

```bash
sacct -j 1733491,1733492,1733493,1733494,1733495 --format=JobID%25,JobName%45,State,Elapsed,ExitCode
```
