# ModelNet40 PointNet++ Line A Smoke Retry Job IDs

- Retry submit time: **2026-07-07 20:19:17 UTC**
- Submit command: `sbatch.tinygpu`
- Cluster: tinygpu
- Prerequisite: Line A v2 metadata fix applied (`reports/modelnet40_pointnet2_lineA_metadata_fix_audit.md`)

| branch | sbatch path | old failed job id | new retry job id | expected point count |
| --- | --- | ---: | ---: | ---: |
| lineA_ear_4096 | `jobs/pointnet2_smoke/smoke_lineA_ear_4096.sbatch` | 1733476 | **1733487** | 4096 |
| lineA_pdans_4096 | `jobs/pointnet2_smoke/smoke_lineA_pdans_4096.sbatch` | 1733477 | **1733488** | 4096 |
| lineA_punet_4096 | `jobs/pointnet2_smoke/smoke_lineA_punet_4096.sbatch` | 1733478 | **1733489** | 4096 |
| lineA_pugcn_4096 | `jobs/pointnet2_smoke/smoke_lineA_pugcn_4096.sbatch` | 1733479 | **1733490** | 4096 |

## Monitoring

```bash
watch -n 60 '
echo "===== PointNet++ Line A smoke retry jobs ====="
squeue.tinygpu -u $USER 2>/dev/null | grep -Ei "pointnet|lineA|smoke|mn40" || squeue -u $USER | grep -Ei "pointnet|lineA|smoke|mn40" || true
'
```

```bash
sacct -j 1733487,1733488,1733489,1733490 --format=JobID%25,JobName%45,State,Elapsed,ExitCode
```
