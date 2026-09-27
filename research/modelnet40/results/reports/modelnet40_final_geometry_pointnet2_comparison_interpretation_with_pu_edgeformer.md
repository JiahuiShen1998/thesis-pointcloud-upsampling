# Geometry vs PointNet++ Final Interpretation (with PU-EdgeFormer)

- Generated: `2026-07-17 08:35:11 UTC`
- Comparison logic: `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`

## 1. Line A

- **Original baseline 1024** remains the strongest PointNet++ branch (**91.95%** best overall accuracy).
- None of the **4096-point upsampling** branches exceeds the Original baseline on best overall accuracy:
  - EAR: 91.48% (−0.47 pp)
  - PDANS: 91.62% (−0.33 pp)
  - PU-Net: 90.95% (−1.00 pp)
  - PU-GCN: 91.63% (−0.32 pp)
  - PU-EdgeFormer: 90.54% (−1.41 pp)
- **Interpretation:** Increasing point count via upsampling does not yield a stable classification benefit in Line A.

## 2. Line B

- **Geometry** is evaluated against the **Original baseline 1024** (recovery question).
- **PU-GCN** has the smallest **ΔCD** vs Original baseline among upsampling methods.
- **Classification primary comparison** uses the **Downsampled ×4 baseline 256**.
- **PU-Net** achieves the best PointNet++ classification among upsampling methods and is the **only** method above the Downsampled baseline (**+0.42 pp**).
- PU-Net still remains **below** the Original baseline (**−0.68 pp** gap).
- **PU-EdgeFormer** reports valid geometry and classification results, but classification decreases vs both references:
  - primary Δ vs Downsampled baseline: **−1.53 pp**
  - secondary gap vs Original baseline: **−2.63 pp**

## 3. Geometry vs classification

- Geometry recovery and downstream classification are **not perfectly aligned**.
- PU-EdgeFormer reinforces this mismatch: moderate geometry deltas with weaker classification recovery.

## 4. Point cloud qualitative visualization

- **Static figures** (`final_visualization_package/pointcloud_static/`) are suitable for thesis document insertion.
- **Interactive HTML** (`pointcloud_examples_interactive_v2/`) is indexed for supplementary material, presentation, and qualitative inspection.
- Legacy static comparisons do not include PU-EdgeFormer panels; interactive v2 grids should be used for full-method qualitative review.

## 5. Safety / scope

- No retraining, no geometry recomputation, no dataset modification in this visualization pass.
- DETECTOR_EVAL_STARTED=NO; KITTI_AP_EVAL_STARTED=NO
