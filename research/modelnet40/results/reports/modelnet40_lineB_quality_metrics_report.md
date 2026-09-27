# Line B Quality Metrics Report

- Generated at: 2026-07-02 19:35:40 UTC

## Metric definitions

- **CD**: Chamfer distance (forward + backward mean squared)
- **HD**: Hausdorff distance (max min-distance)
- **P2F**: Point-to-surface distance to original CAD .off mesh (unit-sphere normalized)
- **NUC**: NUC (Neighborhood Uniformity Coefficient): For each sample, select 128 query centers (fixed seed). At radii (0.02, 0.05, 0.1), count neighbors within each radius; compute coefficient of variation (std/mean) across centers. Lower NUC = more uniform local density. nuc_mean = mean of CV at 3 radii.

- Per-sample CSV: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_lineB_quality_metrics_per_sample.csv`
- Summary CSV: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_lineB_quality_metrics_summary.csv`
- Samples computed: 12311