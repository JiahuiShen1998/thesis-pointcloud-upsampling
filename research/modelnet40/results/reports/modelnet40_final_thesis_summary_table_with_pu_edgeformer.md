# Thesis Summary Table (with PU-EdgeFormer)

- Generated: `2026-07-17 08:35:11 UTC`

| Line | Method | Points | CD | ΔCD vs Original | NUC | Best OA | Primary Δ pp | Gap pp | Observation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | Original baseline | 1024 | 0.049548 | +0.000000 | 1.0619 | 91.95% | +0.00 pp | — | Line A reference; strongest classification |
| A | EAR | 4096 | 0.049197 | -0.000351 | 0.8065 | 91.48% | -0.47 pp | — | below Original baseline |
| A | PDANS | 4096 | 0.041018 | -0.008530 | 0.6468 | 91.62% | -0.33 pp | — | below Original baseline |
| A | PU-Net | 4096 | 0.044525 | -0.005023 | 0.5919 | 90.95% | -1.00 pp | — | below Original baseline |
| A | PU-GCN | 4096 | 0.036304 | -0.013243 | 0.5513 | 91.63% | -0.32 pp | — | below Original baseline |
| A | PU-EdgeFormer | 4096 | 0.040779 | -0.008769 | 0.6648 | 90.54% | -1.41 pp | — | below Original baseline |
| B | Downsampled baseline | 256 | 0.077123 | +0.027575 | 2.0493 | 90.85% | +0.00 pp | -1.10 pp | Line B degraded input reference |
| B | EAR | 1024 | 0.081356 | +0.031809 | 0.9578 | 88.81% | -2.04 pp | -3.14 pp | below Downsampled baseline |
| B | PDANS | 1024 | 0.063137 | +0.013590 | 1.0899 | 90.34% | -0.51 pp | -1.61 pp | below Downsampled baseline |
| B | PU-Net | 1024 | 0.057115 | +0.007567 | 1.0582 | 91.27% | +0.42 pp | -0.68 pp | only method above Downsampled baseline on classification |
| B | PU-GCN | 1024 | 0.054827 | +0.005279 | 1.4865 | 90.06% | -0.79 pp | -1.89 pp | smallest ΔCD vs Original baseline |
| B | PU-EdgeFormer | 1024 | 0.062994 | +0.013447 | 1.0979 | 89.32% | -1.53 pp | -2.63 pp | valid geometry/classification; classification below both baselines |
