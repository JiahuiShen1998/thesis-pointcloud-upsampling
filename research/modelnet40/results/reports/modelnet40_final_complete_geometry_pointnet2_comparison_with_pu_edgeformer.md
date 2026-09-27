# Final Complete Geometry + PointNet++ Comparison (with PU-EdgeFormer)

- Generated: `2026-07-17 08:35:11 UTC`
- Geometry absolute values vs **dense mesh surface reference**.
- Geometry deltas vs **Original baseline 1024**.
- Line B classification primary Δ vs **Downsampled ×4 baseline 256**.
- Line B secondary gap vs **Original baseline 1024**.
- Line A classification Δ vs **Original baseline 1024**.

| Line | Method | pts | CD | ΔCD | Best OA | Primary Δ pp | Gap pp | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | Original baseline | 1024 | 0.049548 | +0.000000 | 91.95% | +0.00 pp | — | PASS |
| A | EAR | 4096 | 0.049197 | -0.000351 | 91.48% | -0.47 pp | — | PASS |
| A | PDANS | 4096 | 0.041018 | -0.008530 | 91.62% | -0.33 pp | — | PASS |
| A | PU-Net | 4096 | 0.044525 | -0.005023 | 90.95% | -1.00 pp | — | PASS |
| A | PU-GCN | 4096 | 0.036304 | -0.013243 | 91.63% | -0.32 pp | — | PASS |
| A | PU-EdgeFormer | 4096 | 0.040779 | -0.008769 | 90.54% | -1.41 pp | — | PASS |
| B | Downsampled baseline | 256 | 0.077123 | +0.027575 | 90.85% | +0.00 pp | -1.10 pp | PASS |
| B | EAR | 1024 | 0.081356 | +0.031809 | 88.81% | -2.04 pp | -3.14 pp | PASS |
| B | PDANS | 1024 | 0.063137 | +0.013590 | 90.34% | -0.51 pp | -1.61 pp | PASS |
| B | PU-Net | 1024 | 0.057115 | +0.007567 | 91.27% | +0.42 pp | -0.68 pp | PASS |
| B | PU-GCN | 1024 | 0.054827 | +0.005279 | 90.06% | -0.79 pp | -1.89 pp | PASS |
| B | PU-EdgeFormer | 1024 | 0.062994 | +0.013447 | 89.32% | -1.53 pp | -2.63 pp | PASS |
