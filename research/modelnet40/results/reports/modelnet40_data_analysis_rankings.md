# ModelNet40 Data Analysis Rankings

- Generated at: 2026-07-12 17:58:43 UTC
- Derived from finalized geometry and classification summaries (no retraining)

## Line A Classification

- **Highest best overall accuracy:** Original baseline 1024 (91.95%)
- **Original baseline best overall:** 91.95%; **final overall:** 91.31%

### Original + Upsampling vs Original baseline (best overall)

- Above baseline (best overall): none
- Below baseline: Original + EAR 4096 (91.48%, -0.47pp), Original + PDANS 4096 (91.62%, -0.33pp), Original + PU-Net 4096 (90.95%, -1.00pp), Original + PU-GCN 4096 (91.63%, -0.32pp)

### Original + Upsampling vs Original baseline (final overall)

- Above baseline: none
- Below baseline: Original + EAR 4096 (90.85%), Original + PDANS 4096 (90.87%), Original + PU-Net 4096 (90.55%), Original + PU-GCN 4096 (91.00%)

### 4096-point upsampling benefit (Line A)

- No upsampling method exceeds the Original 1024 baseline on **best overall** accuracy.
- On **final overall** accuracy, all four upsampling methods are also below baseline.
- Increasing point count to 4096 does **not** yield a stable classification gain in this protocol.

## Line B Classification

- **Highest best overall accuracy:** Downsampled x4 + PU-Net 1024 (91.27%)
- **Downsampled x4 baseline best overall:** 90.85%

### Downsampled + Upsampling vs Downsampled baseline (best overall)

- Above baseline: Downsampled x4 + PU-Net 1024 (91.27%, +0.42pp)
- Below baseline: Downsampled x4 + EAR 1024 (88.81%, -2.04pp), Downsampled x4 + PDANS 1024 (90.34%, -0.51pp), Downsampled x4 + PU-GCN 1024 (90.06%, -0.79pp)

- **PU-Net only above baseline?** Yes

### Per-method delta vs Downsampled x4 baseline (best overall)

- Downsampled x4 + EAR 1024: -2.04pp
- Downsampled x4 + PDANS 1024: -0.51pp
- Downsampled x4 + PU-Net 1024: +0.42pp
- Downsampled x4 + PU-GCN 1024: -0.79pp

## Geometry (vs Original baseline)

### Line B upsampling methods (thesis focus)

- **Closest CD:** Downsampled x4 + PU-GCN (Δ=0.005279)
- **Closest HD:** Downsampled x4 + PU-Net (Δ=0.017954)
- **Closest NUC:** Downsampled x4 + PU-Net (Δ=-0.003731)
- **Closest exact_P2F:** Downsampled x4 + PU-GCN (Δ=0.001309)

### All protocol branches

- **Closest CD:** Original + EAR (Δ=-0.000351)
- **Closest HD:** Original + EAR (Δ=-0.002312)
- **Closest NUC:** Downsampled x4 + PU-Net (Δ=-0.003731)
- **Closest exact_P2F:** Downsampled x4 baseline (Δ=0.000021)

Line B upsampling CD ranking (smallest Δ CD vs Original):
- PU-GCN (0.005279) < PU-Net (0.007567) < PDANS (0.013590) < EAR (0.031809)

## Geometry vs Classification

- **Best Line B CD (upsampling):** Downsampled x4 + PU-GCN 1024 (CD=0.054827)
- **Best Line B classification (best overall):** Downsampled x4 + PU-Net 1024 (91.27%)
- **Geometric CD leader equals classification leader?** No

### Geometry improves but classification does not improve (Line B upsampling)

- Downsampled x4 + EAR 1024: better than downsampled baseline on at least one geometry metric, but best overall ≤ baseline (88.81%)
- Downsampled x4 + PDANS 1024: better than downsampled baseline on at least one geometry metric, but best overall ≤ baseline (90.34%)
- Downsampled x4 + PU-GCN 1024: better than downsampled baseline on at least one geometry metric, but best overall ≤ baseline (90.06%)

### Classification improves but geometry vs Original is mixed

- Downsampled x4 + PU-Net 1024: best overall above baseline, but only 2/4 geometry metrics beat Original baseline
