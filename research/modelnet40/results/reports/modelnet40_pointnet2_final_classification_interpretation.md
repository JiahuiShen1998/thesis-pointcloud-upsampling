# ModelNet40 PointNet++ Final Classification Interpretation

- Generated at: 2026-07-13 14:32:51 UTC
- Source: finalized two-line PointNet++ classification results (no retraining)

## 1. Line A

- **Original baseline 1024** achieves the highest best overall accuracy: **91.95%**.
- All **Original + Upsampling 4096** methods remain **below** this baseline on best overall accuracy:
  - EAR: 91.48% (−0.47 pp)
  - PDANS: 91.62% (−0.33 pp)
  - PU-Net: 90.95% (−1.00 pp)
  - PU-GCN: 91.63% (−0.32 pp)
- **Interpretation:** Increasing the input point count to 4096 via upsampling does **not** yield a stable classification benefit in Line A. The native 1024-point Original baseline remains strongest.

## 2. Line B

- **Downsampled x4 baseline 256** best overall accuracy: **90.85%**.
- **PU-Net** (Downsampled x4 + PU-Net 1024) best overall accuracy: **91.27%**.
- PU-Net is the **only** upsampling method slightly above the downsampled baseline, with a gain of **+0.42 percentage points**.
- EAR (−2.04 pp), PDANS (−0.51 pp), and PU-GCN (−0.79 pp) do **not** exceed the downsampled baseline on best overall accuracy.
- **Interpretation:** Recovering point count after ×4 downsampling can marginally help classification, but only PU-Net achieves this in our experiment.

## 3. Overall Conclusions

1. **Point cloud upsampling does not universally improve downstream classification.** Line A shows no benefit from 4096-point upsampling; Line B shows benefit only for PU-Net.
2. **The effect depends on the upsampling method and the downstream task.** Geometry-leading methods (e.g., PU-GCN on CD) do not necessarily yield the best classifier.
3. **PU-Net is the most promising method in Line B**, as the sole upsampling variant exceeding the downsampled baseline on best overall accuracy.
4. **Geometric quality and classification accuracy are not fully aligned.** Best CD (PU-GCN) ≠ best classification (PU-Net).

## 4. Thesis Usage

- Use **delta methods-only figures** in the main text to show relative improvement without baseline bars.
- Use **absolute accuracy figures** when showing baseline as a real evaluated training branch.
- Report absolute values with baseline rows in tables; interpret deltas in percentage points (pp).
