# ModelNet40 PointNet++ Line B Smoke Audit

- Date: 2026-07-07
- Scope: Line B only (5 branches)
- Overall status: **BLOCKED — jobs not submitted**

## Summary

| Branch | Expected pts | Job state | Exit code | Status |
|--------|--------------|-----------|-----------|--------|
| lineB_downsampled_x4_baseline_256 | 256 | NOT_SUBMITTED | — | BLOCKED |
| lineB_ear_1024 | 1024 | NOT_SUBMITTED | — | BLOCKED |
| lineB_pdans_1024 | 1024 | NOT_SUBMITTED | — | BLOCKED |
| lineB_punet_1024 | 1024 | NOT_SUBMITTED | — | BLOCKED |
| lineB_pugcn_1024 | 1024 | NOT_SUBMITTED | — | BLOCKED |

- PASS count: **0 / 5**
- FAIL count: **0 / 5** (not run)
- BLOCKED count: **5 / 5**

## Root cause

This audit was attempted from agent host `lms41-24`:

1. Canonical HPC path `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling` is **not mounted**.
2. Local staging root `/home/ra87racy/projects/modelnet40_pointnet2_upsampling` does **not** contain `jobs/pointnet2_smoke/*.sbatch` or PointNet++ training prep artifacts described in the protocol checklist.
3. `sbatch.tinygpu`, `sbatch`, and `squeue` are **unavailable** on this host.

No smoke logs, metrics.json, or training outputs were produced in this attempt.

## Monitoring command (run on HPC after submission)

```bash
watch -n 60 '
echo "===== PointNet++ Line B smoke jobs ====="
squeue -u $USER | grep -Ei "pointnet|smoke|lineB|mn40" || true
'
```

After jobs finish:

```bash
sacct -j <JOBID1>,<JOBID2>,<JOBID3>,<JOBID4>,<JOBID5> --format=JobID%25,JobName%45,State,Elapsed,ExitCode
```

## CSV

- `reports/modelnet40_pointnet2_lineB_smoke_audit.csv`

## Next step

Submit the 5 Line B smoke jobs from the HPC login node, then re-run smoke log audit and update this report.
