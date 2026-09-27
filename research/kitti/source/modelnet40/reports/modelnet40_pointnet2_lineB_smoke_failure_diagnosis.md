# ModelNet40 PointNet++ Line B Smoke — Failure Diagnosis

- Date: 2026-07-07
- Type: **submission blocked** (not a training/runtime failure)

## Failed branches

All 5 Line B smoke branches are **BLOCKED** before submission:

1. lineB_downsampled_x4_baseline_256
2. lineB_ear_1024
3. lineB_pdans_1024
4. lineB_punet_1024
5. lineB_pugcn_1024

## Job metadata

| Field | Value |
|-------|-------|
| job id | NOT_SUBMITTED |
| job state | NOT_SUBMITTED |
| exit code | N/A |
| log path | N/A (no jobs launched) |

## Exact error / blocker message

```
cd: /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling: No such file or directory
which sbatch.tinygpu -> not found
squeue -> unavailable
jobs/pointnet2_smoke/smoke_lineB_*.sbatch -> not present on local staging root
```

## Likely cause

**Environment / host mismatch**, not a PointNet++ training bug:

- Smoke submission requires the HPC login node with SLURM and the canonical woody `PROJECT_ROOT`.
- Current agent host only has a partial local staging copy (PU-GCN upsampling scripts only).

## Is it a dataloader point count issue?

**No** — training never started; no dataloader was invoked.

## Is it GPU OOM?

**No** — no GPU job was launched.

## Is it a config path issue?

**Possibly on HPC** — cannot verify from this host because `jobs/pointnet2_smoke/` and branch configs are absent locally. On HPC, confirm configs exist under the canonical project before submit.

## Is it an environment/module issue?

**Yes** — primary blocker is missing HPC filesystem mount and missing SLURM client on agent host.

## Recommended fix

1. SSH to HPC login node.
2. `cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`
3. Verify smoke sbatch files exist:
   ```bash
   ls jobs/pointnet2_smoke/smoke_lineB_*.sbatch
   ```
4. Submit 5 Line B smoke jobs with `sbatch.tinygpu` (or `sbatch`).
5. Record job IDs in `reports/modelnet40_pointnet2_lineB_smoke_job_ids.md`.
6. After completion, audit logs and regenerate:
   - `reports/modelnet40_pointnet2_lineB_smoke_audit.csv`
   - `reports/modelnet40_pointnet2_lineB_smoke_audit.md`

## Full training

**Do not submit Line B full training** until all 5 smoke branches PASS on HPC.
