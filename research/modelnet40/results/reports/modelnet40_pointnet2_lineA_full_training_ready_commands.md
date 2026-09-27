# ModelNet40 PointNet++ Line A Full Training Ready Commands

- Generated at: 2026-07-07 20:21:00 UTC
- Prerequisite: **5 / 5 Line A smoke PASS** (`LINE_A_SMOKE_ALL_PASS`)
- **Not submitted automatically** — run manually when ready.

## Commands (sbatch.tinygpu — preferred)

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

sbatch.tinygpu jobs/pointnet2_full/run_lineA_original_baseline_1024.sbatch
sbatch.tinygpu jobs/pointnet2_full/run_lineA_ear_4096.sbatch
sbatch.tinygpu jobs/pointnet2_full/run_lineA_pdans_4096.sbatch
sbatch.tinygpu jobs/pointnet2_full/run_lineA_punet_4096.sbatch
sbatch.tinygpu jobs/pointnet2_full/run_lineA_pugcn_4096.sbatch
```

## Fallback (plain sbatch)

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

sbatch jobs/pointnet2_full/run_lineA_original_baseline_1024.sbatch
sbatch jobs/pointnet2_full/run_lineA_ear_4096.sbatch
sbatch jobs/pointnet2_full/run_lineA_pdans_4096.sbatch
sbatch jobs/pointnet2_full/run_lineA_punet_4096.sbatch
sbatch jobs/pointnet2_full/run_lineA_pugcn_4096.sbatch
```

## Recommended order

1. Submit all 5 Line A full training jobs (above).
2. Monitor under `pointnet2_results/x4_two_line_final/`.
3. Line B full training is already complete — do not re-submit.
4. Do not mix smoke metrics with final classification results.

## Output directories

- `pointnet2_results/x4_two_line_final/lineA_original_baseline/`
- `pointnet2_results/x4_two_line_final/lineA_original_up/ear/`
- `pointnet2_results/x4_two_line_final/lineA_original_up/pdans/`
- `pointnet2_results/x4_two_line_final/lineA_original_up/pu_net/`
- `pointnet2_results/x4_two_line_final/lineA_original_up/pu_gcn/`

## Smoke job reference

| branch | smoke job id | point count |
| --- | ---: | ---: |
| lineA_original_baseline_1024 | 1733475 | 1024 |
| lineA_ear_4096 | 1733487 | 4096 |
| lineA_pdans_4096 | 1733488 | 4096 |
| lineA_punet_4096 | 1733489 | 4096 |
| lineA_pugcn_4096 | 1733490 | 4096 |
