# ModelNet40 PointNet++ Final Classification Report (with PU-EdgeFormer)

- Generated: `2026-07-17 04:10:04 UTC`
- Delta units: **percentage points (pp)**
- Line A primary reference: Original baseline 1024
- Line B primary reference: Downsampled ×4 baseline 256
- Line B secondary gap: Original baseline 1024
- Geometry deltas remain vs Original baseline (separate from classification)
- See `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`

## Line A

- Line A upsampling classifier setting: **PointNet++ retrained with 4096-point input** (not a fixed 1024-only model)
- See `reports/modelnet40_pointnet2_4096_adapted_explanation_for_teacher.md`

| Method | Points | Classifier setting | Best Overall | Δ vs Original (pp) | Final Overall | Status |
|---|---:|---|---:|---:|---:|---|
| Original baseline 1024 | 1024 | PointNet++ trained with 1024-point input | 91.95% | +0.00 pp | 91.31% | PASS |
| Original + EAR 4096 | 4096 | PointNet++ retrained with 4096-point input | 91.48% | -0.47 pp | 90.85% | PASS |
| Original + PDANS 4096 | 4096 | PointNet++ retrained with 4096-point input | 91.62% | -0.33 pp | 90.87% | PASS |
| Original + PU-Net 4096 | 4096 | PointNet++ retrained with 4096-point input | 90.95% | -1.00 pp | 90.55% | PASS |
| Original + PU-GCN 4096 | 4096 | PointNet++ retrained with 4096-point input | 91.63% | -0.32 pp | 91.00% | PASS |
| Original + PU-EdgeFormer 4096 | 4096 | PointNet++ retrained with 4096-point input | 90.54% | -1.41 pp | 90.13% | PASS |

## Line B

| Method | Points | Best Overall | Δ vs Downsampled (pp, primary) | gap vs Original (pp, secondary) | Final Overall | Status |
|---|---:|---:|---:|---:|---:|---|
| Downsampled x4 baseline 256 | 256 | 90.85% | +0.00 pp | -1.10 pp | 90.46% | PASS |
| Downsampled x4 + EAR 1024 | 1024 | 88.81% | -2.04 pp | -3.14 pp | 88.02% | PASS |
| Downsampled x4 + PDANS 1024 | 1024 | 90.34% | -0.51 pp | -1.61 pp | 89.46% | PASS |
| Downsampled x4 + PU-Net 1024 | 1024 | 91.27% | +0.42 pp | -0.68 pp | 90.65% | PASS |
| Downsampled x4 + PU-GCN 1024 | 1024 | 90.06% | -0.79 pp | -1.89 pp | 89.63% | PASS |
| Downsampled x4 + PU-EdgeFormer 1024 | 1024 | 89.32% | -1.53 pp | -2.63 pp | 88.58% | PASS |

