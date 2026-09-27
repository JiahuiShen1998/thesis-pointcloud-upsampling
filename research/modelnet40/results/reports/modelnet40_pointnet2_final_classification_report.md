# ModelNet40 PointNet++ Final Classification Report

- Generated at: 2026-07-13 14:32:51 UTC
- Source: existing classification summaries (no retraining)
- Delta units: percentage points (pp)

## 1. Line A Classification

- Line A upsampling classifier setting: **PointNet++ retrained with 4096-point input**
- Clarification for teacher: `reports/modelnet40_pointnet2_4096_adapted_explanation_for_teacher.md`

| Method | Points | Best Overall | Best Class | Δ Best (pp) | Final Overall | Δ Final (pp) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original baseline 1024 | 1024 | 91.95% | 88.00% | +0.00 pp | 91.31% | +0.00 pp | PASS |
| Original + EAR 4096 | 4096 | 91.48% | 88.19% | -0.47 pp | 90.85% | -0.46 pp | PASS |
| Original + PDANS 4096 | 4096 | 91.62% | 88.44% | -0.33 pp | 90.87% | -0.44 pp | PASS |
| Original + PU-Net 4096 | 4096 | 90.95% | 87.46% | -1.00 pp | 90.55% | -0.76 pp | PASS |
| Original + PU-GCN 4096 | 4096 | 91.63% | 87.74% | -0.32 pp | 91.00% | -0.32 pp | PASS |

### Delta vs Original baseline 1024

- EAR: Δ best overall = -0.47 pp
- PDANS: Δ best overall = -0.33 pp
- PU-Net: Δ best overall = -1.00 pp
- PU-GCN: Δ best overall = -0.32 pp

### Ranking (best overall accuracy)

1. Original baseline 1024 — 91.95% (+0.00 pp vs baseline)
2. Original + PU-GCN 4096 — 91.63% (-0.32 pp vs baseline)
3. Original + PDANS 4096 — 91.62% (-0.33 pp vs baseline)
4. Original + EAR 4096 — 91.48% (-0.47 pp vs baseline)
5. Original + PU-Net 4096 — 90.95% (-1.00 pp vs baseline)

## 2. Line B Classification

| Method | Points | Best Overall | Best Class | Δ Best (pp) | Final Overall | Δ Final (pp) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Downsampled x4 baseline 256 | 256 | 90.85% | 87.37% | +0.00 pp | 90.46% | +0.00 pp | PASS |
| Downsampled x4 + EAR 1024 | 1024 | 88.81% | 84.50% | -2.04 pp | 88.02% | -2.44 pp | PASS |
| Downsampled x4 + PDANS 1024 | 1024 | 90.34% | 86.20% | -0.51 pp | 89.46% | -1.00 pp | PASS |
| Downsampled x4 + PU-Net 1024 | 1024 | 91.27% | 87.86% | +0.42 pp | 90.65% | +0.19 pp | PASS |
| Downsampled x4 + PU-GCN 1024 | 1024 | 90.06% | 86.41% | -0.79 pp | 89.63% | -0.83 pp | PASS |

### Delta vs Downsampled x4 baseline 256

- EAR: Δ best overall = -2.04 pp
- PDANS: Δ best overall = -0.51 pp
- PU-Net: Δ best overall = +0.42 pp
- PU-GCN: Δ best overall = -0.79 pp

### Ranking (best overall accuracy)

1. Downsampled x4 + PU-Net 1024 — 91.27% (+0.42 pp vs baseline)
2. Downsampled x4 baseline 256 — 90.85% (+0.00 pp vs baseline)
3. Downsampled x4 + PDANS 1024 — 90.34% (-0.51 pp vs baseline)
4. Downsampled x4 + PU-GCN 1024 — 90.06% (-0.79 pp vs baseline)
5. Downsampled x4 + EAR 1024 — 88.81% (-2.04 pp vs baseline)

## 3. Combined Two-Line Classification

| Line | Method | Points | Best Overall | Δ Best (pp) | Baseline Reference |
| --- | --- | ---: | ---: | ---: | --- |
| A | Original baseline 1024 | 1024 | 91.95% | +0.00 pp | Original baseline 1024 |
| A | Original + EAR 4096 | 4096 | 91.48% | -0.47 pp | Original baseline 1024 |
| A | Original + PDANS 4096 | 4096 | 91.62% | -0.33 pp | Original baseline 1024 |
| A | Original + PU-Net 4096 | 4096 | 90.95% | -1.00 pp | Original baseline 1024 |
| A | Original + PU-GCN 4096 | 4096 | 91.63% | -0.32 pp | Original baseline 1024 |
| B | Downsampled x4 baseline 256 | 256 | 90.85% | +0.00 pp | Downsampled x4 baseline 256 |
| B | Downsampled x4 + EAR 1024 | 1024 | 88.81% | -2.04 pp | Downsampled x4 baseline 256 |
| B | Downsampled x4 + PDANS 1024 | 1024 | 90.34% | -0.51 pp | Downsampled x4 baseline 256 |
| B | Downsampled x4 + PU-Net 1024 | 1024 | 91.27% | +0.42 pp | Downsampled x4 baseline 256 |
| B | Downsampled x4 + PU-GCN 1024 | 1024 | 90.06% | -0.79 pp | Downsampled x4 baseline 256 |

## 4. Delta Accuracy Summary

**Line A:** All upsampling methods (EAR, PDANS, PU-Net, PU-GCN) show negative Δ best overall vs Original baseline 1024.

**Line B:** Only PU-Net shows positive Δ best overall (+0.42 pp) vs Downsampled x4 baseline 256. EAR, PDANS, and PU-GCN remain below baseline.

## 5. Ranking Summary

- **Line A best:** Original baseline 1024 — 91.95%
- **Line B best (upsampling):** Downsampled x4 + PU-Net 1024 — 91.27%
- **Line B baseline:** Downsampled x4 baseline 256 — 90.85%

