# ModelNet40 PointNet++ Line B Full Training Ready Commands

- Generated at: 2026-07-07 12:41:00 UTC
- Prerequisite: **5 / 5 Line B smoke PASS** (`LINE_B_SMOKE_ALL_PASS`)
- **Not submitted automatically** — run manually when ready.

## Commands (sbatch.tinygpu — preferred)

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

sbatch.tinygpu jobs/pointnet2_full/run_lineB_downsampled_x4_baseline_256.sbatch
sbatch.tinygpu jobs/pointnet2_full/run_lineB_ear_1024.sbatch
sbatch.tinygpu jobs/pointnet2_full/run_lineB_pdans_1024.sbatch
sbatch.tinygpu jobs/pointnet2_full/run_lineB_punet_1024.sbatch
sbatch.tinygpu jobs/pointnet2_full/run_lineB_pugcn_1024.sbatch
```

## Fallback (plain sbatch)

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

sbatch jobs/pointnet2_full/run_lineB_downsampled_x4_baseline_256.sbatch
sbatch jobs/pointnet2_full/run_lineB_ear_1024.sbatch
sbatch jobs/pointnet2_full/run_lineB_pdans_1024.sbatch
sbatch jobs/pointnet2_full/run_lineB_punet_1024.sbatch
sbatch jobs/pointnet2_full/run_lineB_pugcn_1024.sbatch
```

## Recommended order

1. Submit Line B full training jobs (above).
2. Monitor under `pointnet2_results/x4_two_line_final/`.
3. After Line B full training starts or completes, submit Line A smoke jobs.
4. Do not mix smoke metrics with final classification results.

## Output directories

- `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_baseline/`
- `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/ear/`
- `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/pdans/`
- `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/pu_net/`
- `pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/pu_gcn/`
