# Geometry vs Original — Audit

- Generated: 2026-07-23 17:06:18 UTC
- Script: `scripts/compute_geometry_vs_original.py`
- Reference: corresponding Original 1024-point cloud only (no mesh / dense surface / P2F)
- Splits: test
- Samples per split used: 2468 (total sample IDs: 2468)
- Averaging: arithmetic mean over valid samples
- CD definition: mean NN L2(a→b) + mean NN L2(b→a)
- HD definition: max( max NN L2(a→b), max NN L2(b→a) )
- NUC: CV of neighbor counts at radii (0.02, 0.05, 0.1), 128 query centers, seed base 42
- Workers: 16
- Elapsed: 63.4s
- Valid result rows: 32084
- Failures: 0

## Input directories

- Original: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original`
- Downsampled ×4: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4`
- Line A strict 4N: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineA_original_up/strict_4N/{method}`
- Line B strict N: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineB_downsampled_x4_up/strict_N/{method}`

## Summary (CD / HD / NUC)

| Line | Method | Points | CD vs Original | HD vs Original | NUC | ΔNUC | N |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Original | 1024 | 0.000000 | 0.000000 | 1.080757 | 0.000000 | 2468 |
| A | EAR | 4096 | 0.029171 | 0.098926 | 0.814734 | -0.266023 | 2468 |
| A | PDANS | 4096 | 0.025907 | 0.083822 | 0.661557 | -0.419199 | 2468 |
| A | PU-Net | 4096 | 0.051957 | 0.088135 | 0.606415 | -0.474341 | 2468 |
| A | PU-GCN | 4096 | 0.034710 | 0.101875 | 0.522997 | -0.557760 | 2468 |
| A | PU-EdgeFormer | 4096 | 0.030343 | 0.045665 | 0.674666 | -0.406091 | 2468 |
| B | Original | 1024 | 0.000000 | 0.000000 | 1.080757 | 0.000000 | 2468 |
| B | Downsampled ×4 | 256 | 0.046396 | 0.198587 | 2.087969 | 1.007213 | 2468 |
| B | EAR | 1024 | 0.076743 | 0.193182 | 0.961451 | -0.119305 | 2468 |
| B | PDANS | 1024 | 0.062696 | 0.190541 | 1.065492 | -0.015265 | 2468 |
| B | PU-Net | 1024 | 0.064573 | 0.108319 | 1.084117 | 0.003361 | 2468 |
| B | PU-GCN | 1024 | 0.055802 | 0.151403 | 1.512887 | 0.432131 | 2468 |
| B | PU-EdgeFormer | 1024 | 0.061900 | 0.177572 | 1.121371 | 0.040615 | 2468 |

## Notes

- Original vs Original has CD=0 and HD=0 by definition.
- Line A compares unequal cardinalities (4096 vs 1024); metrics measure consistency with the discrete Original set, not continuous-surface accuracy.
- P2F is intentionally not computed.
- Old mesh-reference CD/HD values must not be reused.

## Outputs

- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_geometry_vs_original_lineA.csv`
- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_geometry_vs_original_lineB.csv`
- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_geometry_vs_original_summary.csv`
- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_geometry_vs_original_per_sample.csv`
- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_geometry_vs_original_audit.md`
