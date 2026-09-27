# EAR ×2 Job Reclassification

- Generated: 2026-06-25
- Project: `modelnet40_pointnet2_upsampling`

## Job 1711437 (`p2_ear_b`)

| Field | Value |
| --- | --- |
| Job ID | **1711437** |
| Job name | `p2_ear_b` |
| Experiment | `downsampled50_ear_pointnet2` |
| Protocol | Legacy EAR ×2 (512 → 1024) |
| Slurm state | **COMPLETED** (2026-06-22) |
| Action taken | **No scancel** — job already finished |
| Reclassification | **ablation_preliminary_x2** |

## Notes

- Training result preserved under the existing `downsampled50_ear_pointnet2` experiment outputs.
- This job is **not** part of the main R=×4 protocol.
- Main protocol EAR branch now targets:
  - Line A: 1024 → **4096**
  - Line B: 512 → **2048**
- Legacy upsampled data remains at `datasets/modelnet40_downsampled50_up/ear` (×2, 1024 pts).

## Queue check command

```bash
squeue.tinygpu -u $USER
sacct -j 1711437 --format=JobID,JobName,State,ExitCode,End
```
