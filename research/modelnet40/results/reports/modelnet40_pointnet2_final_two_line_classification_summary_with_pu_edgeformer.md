# Two-Line Classification Summary (with PU-EdgeFormer)

- Generated: `2026-07-17 04:10:04 UTC`
- Units for delta/gap columns: pp
- Line A upsampling classifier setting: **PointNet++ retrained with 4096-point input**

| Line | Method | Points | Classifier setting | Best Overall | Primary Δ (pp) | Secondary gap (pp) |
|---|---|---:|---|---:|---:|---:|
| A | Original baseline 1024 | 1024 | PointNet++ trained with 1024-point input | 91.95% | +0.00 pp | — |
| A | Original + EAR 4096 | 4096 | PointNet++ retrained with 4096-point input | 91.48% | -0.47 pp | — |
| A | Original + PDANS 4096 | 4096 | PointNet++ retrained with 4096-point input | 91.62% | -0.33 pp | — |
| A | Original + PU-Net 4096 | 4096 | PointNet++ retrained with 4096-point input | 90.95% | -1.00 pp | — |
| A | Original + PU-GCN 4096 | 4096 | PointNet++ retrained with 4096-point input | 91.63% | -0.32 pp | — |
| A | Original + PU-EdgeFormer 4096 | 4096 | PointNet++ retrained with 4096-point input | 90.54% | -1.41 pp | — |
| B | Downsampled x4 baseline 256 | 256 | PointNet++ trained with 256-point input | 90.85% | +0.00 pp | -1.10 pp |
| B | Downsampled x4 + EAR 1024 | 1024 | PointNet++ retrained with 1024-point input | 88.81% | -2.04 pp | -3.14 pp |
| B | Downsampled x4 + PDANS 1024 | 1024 | PointNet++ retrained with 1024-point input | 90.34% | -0.51 pp | -1.61 pp |
| B | Downsampled x4 + PU-Net 1024 | 1024 | PointNet++ retrained with 1024-point input | 91.27% | +0.42 pp | -0.68 pp |
| B | Downsampled x4 + PU-GCN 1024 | 1024 | PointNet++ retrained with 1024-point input | 90.06% | -0.79 pp | -1.89 pp |
| B | Downsampled x4 + PU-EdgeFormer 1024 | 1024 | PointNet++ retrained with 1024-point input | 89.32% | -1.53 pp | -2.63 pp |
