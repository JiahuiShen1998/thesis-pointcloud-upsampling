# ModelNet40 Geometry vs Classification Final Summary

- Generated at: 2026-07-12 17:58:43 UTC
- Merged from geometry thesis table and two-line classification summaries

| Line | Method | Points | CD | HD | NUC | exact P2F | Best Overall Accuracy | Δ Best Overall vs line baseline | Final Overall Accuracy | Δ Final Overall vs line baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Original baseline 1024 | 1024 | 0.049548 | 0.119720 | 1.061943 | 0.082469 | 91.95% | +0.00pp | 91.31% | +0.00pp |
| A | Original + EAR 4096 | 4096 | 0.049197 | 0.117408 | 0.806502 | 0.082370 | 91.48% | -0.47pp | 90.85% | -0.46pp |
| A | Original + PDANS 4096 | 4096 | 0.041018 | 0.115437 | 0.646813 | 0.081159 | 91.62% | -0.33pp | 90.87% | -0.44pp |
| A | Original + PU-Net 4096 | 4096 | 0.044525 | 0.114043 | 0.591942 | 0.080043 | 90.95% | -1.00pp | 90.55% | -0.76pp |
| A | Original + PU-GCN 4096 | 4096 | 0.036304 | 0.090933 | 0.551304 | 0.081426 | 91.63% | -0.32pp | 91.00% | -0.32pp |
| B | Downsampled x4 baseline 256 | 256 | 0.077123 | 0.214251 | 2.049289 | 0.082490 | 90.85% | +0.00pp | 90.46% | +0.00pp |
| B | Downsampled x4 + EAR 1024 | 1024 | 0.081356 | 0.208793 | 0.957795 | 0.081154 | 88.81% | -2.04pp | 88.02% | -2.44pp |
| B | Downsampled x4 + PDANS 1024 | 1024 | 0.063137 | 0.205927 | 1.089924 | 0.079586 | 90.34% | -0.51pp | 89.46% | -1.00pp |
| B | Downsampled x4 + PU-Net 1024 | 1024 | 0.057115 | 0.137674 | 1.058212 | 0.080067 | 91.27% | +0.42pp | 90.65% | +0.19pp |
| B | Downsampled x4 + PU-GCN 1024 | 1024 | 0.054827 | 0.165212 | 1.486469 | 0.083779 | 90.06% | -0.79pp | 89.63% | -0.83pp |
