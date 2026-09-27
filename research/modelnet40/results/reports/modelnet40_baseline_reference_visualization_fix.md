# ModelNet40 Baseline and Reference Visualization Guide

- Generated at: 2026-07-13 14:32:51 UTC

## 1. Why delta figures no longer show baseline bars

In **delta (methods-only) figures**, the baseline is a **reference for comparison**, not an upsampling method. Including the baseline as a bar at y = 0 is visually redundant and can imply the baseline is one of the compared upsampling techniques. Instead:

- The baseline defines the **y = 0 reference line** (classification) or **zero deviation** (geometry).
- Only upsampling methods (EAR, PDANS, PU-Net, PU-GCN) appear as bars or scatter points.
- The caption states which baseline was used.

## 2. Why absolute classification figures can show baseline

In **absolute accuracy figures**, the Original baseline (Line A) and Downsampled x4 baseline (Line B) are **real training branches** with independently trained PointNet++ models. They are legitimate experimental conditions, not computed references. Showing them as bars alongside upsampling methods is appropriate.

A horizontal dashed line marks the baseline value for easy visual comparison.

## 3. Mesh reference vs experimental baseline

| Concept | Role | Used in |
| --- | --- | --- |
| **Dense mesh surface reference** (GT reference) | Ground truth for geometry metrics (CD, HD, exact P2F, NUC) | Geometry evaluation |
| **Original baseline 1024** | Line A classification baseline; geometry zero-reference for delta plots | Classification Line A; geometry deltas |
| **Downsampled x4 baseline 256** | Line B classification baseline | Classification Line B |

Do **not** call the mesh reference a "baseline" in figure captions. Use "dense mesh surface reference" or "mesh reference."

## 4. Absolute metric tables vs delta figures

- **Tables** report absolute metric values with a baseline row (delta = 0). Readers need exact numbers for thesis tables.
- **Delta figures** emphasize **relative change** and focus on method comparison. Baseline is implicit at zero.

## 5. Why baseline rows are kept in the final classification report

The report CSV/MD includes baseline rows with delta = 0 because:

- Tables must show the full experimental matrix.
- Readers can verify delta calculations from absolute values.
- Ranking and summary sections reference the baseline explicitly.

## 6. Recommended thesis figure usage

| Location | Recommended figures |
| --- | --- |
| **Main text** | Delta methods-only accuracy and geometry figures; final combined geometry-vs-classification scatter |
| **Tables** | Absolute values with baseline row (`modelnet40_pointnet2_final_classification_report`) |
| **Supplementary** | Absolute geometry bar plots; point cloud qualitative comparisons; interactive HTML viewers |
