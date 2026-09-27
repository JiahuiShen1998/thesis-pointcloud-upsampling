# ModelNet40 Thesis Table: Combined Classification (Line A + Line B)

- Generated at: 2026-07-12 17:58:43 UTC

## Line A

- Generated at: 2026-07-12 17:58:43 UTC
- Source: existing classification summaries (no retraining)

| Method | Points | Final Overall Accuracy | Final Class Accuracy | Best Overall Accuracy | Best Class Accuracy | Δ Best Overall vs line baseline | Δ Final Overall vs line baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline 1024 | 1024 | 91.31% | 87.20% | 91.95% | 88.00% | +0.00pp | +0.00pp |
| Original + EAR 4096 | 4096 | 90.85% | 87.30% | 91.48% | 88.19% | -0.47pp | -0.46pp |
| Original + PDANS 4096 | 4096 | 90.87% | 87.93% | 91.62% | 88.44% | -0.33pp | -0.44pp |
| Original + PU-Net 4096 | 4096 | 90.55% | 87.04% | 90.95% | 87.46% | -1.00pp | -0.76pp |
| Original + PU-GCN 4096 | 4096 | 91.00% | 87.26% | 91.63% | 87.74% | -0.32pp | -0.32pp |

## Line B

- Generated at: 2026-07-12 17:58:43 UTC
- Source: existing classification summaries (no retraining)

| Method | Points | Final Overall Accuracy | Final Class Accuracy | Best Overall Accuracy | Best Class Accuracy | Δ Best Overall vs line baseline | Δ Final Overall vs line baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Downsampled x4 baseline 256 | 256 | 90.46% | 87.52% | 90.85% | 87.37% | +0.00pp | +0.00pp |
| Downsampled x4 + EAR 1024 | 1024 | 88.02% | 84.13% | 88.81% | 84.50% | -2.04pp | -2.44pp |
| Downsampled x4 + PDANS 1024 | 1024 | 89.46% | 85.86% | 90.34% | 86.20% | -0.51pp | -1.00pp |
| Downsampled x4 + PU-Net 1024 | 1024 | 90.65% | 87.39% | 91.27% | 87.86% | +0.42pp | +0.19pp |
| Downsampled x4 + PU-GCN 1024 | 1024 | 89.63% | 85.25% | 90.06% | 86.41% | -0.79pp | -0.83pp |
