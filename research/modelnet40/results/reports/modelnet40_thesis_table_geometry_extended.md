# ModelNet40 Thesis Table: Geometry (Extended)

- Generated at: 2026-07-12 17:58:43 UTC
- Reference: Original baseline (1024 pts)
- Lower is better for CD / HD / exact P2F / NUC
- Scope: All protocol branches including Line A upsampling and downsampled baseline

| Method | Points | CD | Δ CD vs Original | HD | Δ HD vs Original | NUC | Δ NUC vs Original | exact P2F | Δ P2F vs Original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 1024 | 0.049548 | 0.000000 | 0.119720 | 0.000000 | 1.061943 | 0.000000 | 0.082469 | 0.000000 |
| Downsampled x4 + EAR | 1024 | 0.081356 | 0.031809 | 0.208793 | 0.089073 | 0.957795 | -0.104148 | 0.081154 | -0.001315 |
| Downsampled x4 + PDANS | 1024 | 0.063137 | 0.013590 | 0.205927 | 0.086207 | 1.089924 | 0.027981 | 0.079586 | -0.002883 |
| Downsampled x4 + PU-Net | 1024 | 0.057115 | 0.007567 | 0.137674 | 0.017954 | 1.058212 | -0.003731 | 0.080067 | -0.002403 |
| Downsampled x4 + PU-GCN | 1024 | 0.054827 | 0.005279 | 0.165212 | 0.045491 | 1.486469 | 0.424526 | 0.083779 | 0.001309 |
| Downsampled x4 baseline | 256 | 0.077123 | 0.027575 | 0.214251 | 0.094531 | 2.049289 | 0.987346 | 0.082490 | 0.000021 |
| Original + EAR | 4096 | 0.049197 | -0.000351 | 0.117408 | -0.002312 | 0.806502 | -0.255441 | 0.082370 | -0.000100 |
| Original + PDANS | 4096 | 0.041018 | -0.008530 | 0.115437 | -0.004284 | 0.646813 | -0.415130 | 0.081159 | -0.001310 |
| Original + PU-Net | 4096 | 0.044525 | -0.005023 | 0.114043 | -0.005677 | 0.591942 | -0.470001 | 0.080043 | -0.002427 |
| Original + PU-GCN | 4096 | 0.036304 | -0.013243 | 0.090933 | -0.028787 | 0.551304 | -0.510639 | 0.081426 | -0.001043 |
