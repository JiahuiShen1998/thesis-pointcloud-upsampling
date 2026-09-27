# ModelNet40 Thesis Table: Geometry (Main)

- Generated at: 2026-07-12 17:58:43 UTC
- Reference: Original baseline (1024 pts)
- Lower is better for CD / HD / exact P2F / NUC
- Scope: Original baseline + Line B upsampling methods

| Method | Points | CD | Δ CD vs Original | HD | Δ HD vs Original | NUC | Δ NUC vs Original | exact P2F | Δ P2F vs Original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 1024 | 0.049548 | 0.000000 | 0.119720 | 0.000000 | 1.061943 | 0.000000 | 0.082469 | 0.000000 |
| Downsampled x4 + EAR | 1024 | 0.081356 | 0.031809 | 0.208793 | 0.089073 | 0.957795 | -0.104148 | 0.081154 | -0.001315 |
| Downsampled x4 + PDANS | 1024 | 0.063137 | 0.013590 | 0.205927 | 0.086207 | 1.089924 | 0.027981 | 0.079586 | -0.002883 |
| Downsampled x4 + PU-Net | 1024 | 0.057115 | 0.007567 | 0.137674 | 0.017954 | 1.058212 | -0.003731 | 0.080067 | -0.002403 |
| Downsampled x4 + PU-GCN | 1024 | 0.054827 | 0.005279 | 0.165212 | 0.045491 | 1.486469 | 0.424526 | 0.083779 | 0.001309 |
