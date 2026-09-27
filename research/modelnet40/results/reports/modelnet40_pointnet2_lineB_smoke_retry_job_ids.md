# ModelNet40 PointNet++ Line B Smoke Retry Job IDs

## Retry round 1 (metadata full-dir symlink — FAILED)

- Submit time: 2026-07-07 12:37:47 UTC
- Failure: manifests pointed to 256-pt baseline paths

| branch | old job id | retry job id | result |
| --- | ---: | ---: | --- |
| lineB_ear_1024 | 1733160 | 1733167 | FAILED (manifest path mismatch) |
| lineB_pdans_1024 | 1733161 | 1733168 | FAILED |
| lineB_punet_1024 | 1733162 | 1733169 | FAILED |
| lineB_pugcn_1024 | 1733163 | 1733170 | FAILED |

## Retry round 2 (metadata v2: label maps only — current)

- Submit time: **2026-07-07 12:39:27 UTC**
- Fix: `class_to_idx.json` + `idx_to_class.json` symlinks only; no manifests → dataloader directory scan
- Submit command: `sbatch.tinygpu`

| branch | sbatch path | old job id | retry job id | expected point count |
| --- | --- | ---: | ---: | ---: |
| lineB_ear_1024 | `jobs/pointnet2_smoke/smoke_lineB_ear_1024.sbatch` | 1733160 | **1733171** | 1024 |
| lineB_pdans_1024 | `jobs/pointnet2_smoke/smoke_lineB_pdans_1024.sbatch` | 1733161 | **1733172** | 1024 |
| lineB_punet_1024 | `jobs/pointnet2_smoke/smoke_lineB_punet_1024.sbatch` | 1733162 | **1733173** | 1024 |
| lineB_pugcn_1024 | `jobs/pointnet2_smoke/smoke_lineB_pugcn_1024.sbatch` | 1733163 | **1733174** | 1024 |

## Monitoring

```bash
watch -n 60 '
echo "===== PointNet++ Line B smoke retry jobs ====="
squeue -u $USER | grep -Ei "pointnet|smoke|lineB|mn40" || true
'
```

```bash
sacct -j 1733171,1733172,1733173,1733174 --format=JobID%25,JobName%45,State,Elapsed,ExitCode
```
