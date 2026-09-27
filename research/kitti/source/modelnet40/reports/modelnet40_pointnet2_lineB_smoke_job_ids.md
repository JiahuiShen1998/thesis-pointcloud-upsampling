# ModelNet40 PointNet++ Line B Smoke — Job IDs

- Submit time: **2026-07-07 13:06 UTC+2**
- Submit host: `lms41-24` (agent host)
- Status: **NOT_SUBMITTED — HPC path / SLURM unavailable on this host**

## Blocker

| Check | Result |
|-------|--------|
| Canonical `PROJECT_ROOT` `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling` | **Not mounted** |
| Local staging `/home/ra87racy/projects/modelnet40_pointnet2_upsampling` | Exists, but **no** `jobs/pointnet2_smoke/` sbatch files |
| `sbatch.tinygpu` | **Not found** |
| `sbatch` / `squeue` | **Not available** |

Smoke jobs must be submitted from the HPC login node where the canonical project and SLURM are available.

## Intended submissions (pending HPC)

| Branch | sbatch path | job id | expected point count | expected input path |
|--------|-------------|--------|----------------------|---------------------|
| lineB_downsampled_x4_baseline_256 | `jobs/pointnet2_smoke/smoke_lineB_downsampled_x4_baseline_256.sbatch` | **NOT_SUBMITTED** | 256 | `pointnet2_inputs/lineB_downsampled_x4/baseline/` |
| lineB_ear_1024 | `jobs/pointnet2_smoke/smoke_lineB_ear_1024.sbatch` | **NOT_SUBMITTED** | 1024 | `pointnet2_inputs/lineB_downsampled_x4_up/ear/` |
| lineB_pdans_1024 | `jobs/pointnet2_smoke/smoke_lineB_pdans_1024.sbatch` | **NOT_SUBMITTED** | 1024 | `pointnet2_inputs/lineB_downsampled_x4_up/pdans/` |
| lineB_punet_1024 | `jobs/pointnet2_smoke/smoke_lineB_punet_1024.sbatch` | **NOT_SUBMITTED** | 1024 | `pointnet2_inputs/lineB_downsampled_x4_up/pu_net/` |
| lineB_pugcn_1024 | `jobs/pointnet2_smoke/smoke_lineB_pugcn_1024.sbatch` | **NOT_SUBMITTED** | 1024 | `pointnet2_inputs/lineB_downsampled_x4_up/pu_gcn/` |

## HPC submit commands (run on login node)

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineB_downsampled_x4_baseline_256.sbatch
sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineB_ear_1024.sbatch
sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineB_pdans_1024.sbatch
sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineB_punet_1024.sbatch
sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineB_pugcn_1024.sbatch
```

Fallback if `sbatch.tinygpu` unavailable:

```bash
sbatch jobs/pointnet2_smoke/smoke_lineB_downsampled_x4_baseline_256.sbatch
sbatch jobs/pointnet2_smoke/smoke_lineB_ear_1024.sbatch
sbatch jobs/pointnet2_smoke/smoke_lineB_pdans_1024.sbatch
sbatch jobs/pointnet2_smoke/smoke_lineB_punet_1024.sbatch
sbatch jobs/pointnet2_smoke/smoke_lineB_pugcn_1024.sbatch
```
