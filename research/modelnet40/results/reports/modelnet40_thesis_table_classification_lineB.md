# ModelNet40 Thesis Table: Line B Classification

- Generated at: 2026-07-16 19:19:22 UTC
- Source: existing classification summaries (no retraining)
- Primary delta: vs **Downsampled ×4 baseline 256** (`delta_accuracy_vs_downsampled_baseline`).
- Secondary gap: vs **Original baseline 1024** (`gap_accuracy_vs_original_baseline`).
- Geometry deltas remain vs Original baseline; do not mix with classification primary.
- See `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`.

| Method | Points | Final Overall Accuracy | Final Class Accuracy | Best Overall Accuracy | Best Class Accuracy | Δ Best Overall vs Downsampled baseline (primary) | Δ Final Overall vs Downsampled baseline (primary) | gap Best Overall vs Original baseline (secondary) | gap Final Overall vs Original baseline (secondary) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Downsampled x4 baseline 256 | 256 | 90.46% | 87.52% | 90.85% | 87.37% | +0.00pp | +0.00pp | -1.10pp | -0.85pp |
| Downsampled x4 + EAR 1024 | 1024 | 88.02% | 84.13% | 88.81% | 84.50% | -2.04pp | -2.44pp | -3.14pp | -3.29pp |
| Downsampled x4 + PDANS 1024 | 1024 | 89.46% | 85.86% | 90.34% | 86.20% | -0.51pp | -1.00pp | -1.61pp | -1.85pp |
| Downsampled x4 + PU-Net 1024 | 1024 | 90.65% | 87.39% | 91.27% | 87.86% | +0.42pp | +0.19pp | -0.68pp | -0.66pp |
| Downsampled x4 + PU-GCN 1024 | 1024 | 89.63% | 85.25% | 90.06% | 86.41% | -0.79pp | -0.83pp | -1.89pp | -1.68pp |
