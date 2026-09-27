# ModelNet40 PointNet++ Line B Classification Summary

- Generated at: 2026-07-07 19:48:31 UTC
- Formal full training (200 epochs, seed=42)

| method | pts | best overall | best class | Δ best overall vs baseline | Δ best class vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Downsampled x4 baseline | 256 | 90.85% | 87.37% | +0.00pp | +0.00pp |
| Downsampled x4 + EAR | 1024 | 88.81% | 84.50% | -2.04pp | -2.86pp |
| Downsampled x4 + PDANS | 1024 | 90.34% | 86.20% | -0.51pp | -1.17pp |
| Downsampled x4 + PU-Net | 1024 | 91.27% | 87.86% | +0.42pp | +0.49pp |
| Downsampled x4 + PU-GCN | 1024 | 90.06% | 86.41% | -0.79pp | -0.96pp |
