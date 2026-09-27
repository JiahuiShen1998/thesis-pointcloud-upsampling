# ModelNet40 Visualization Summary

- Generated at: 2026-07-13 14:32:51 UTC
- Project: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

## 1. PointNet++ final absolute accuracy figures

- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/figures/modelnet40/pointnet2_final/accuracy_absolute` — 3 PNG figures
- Line A, Line B, and two-line grouped absolute best overall accuracy
- Baseline shown as real training branch with horizontal reference line

## 2. PointNet++ final delta accuracy figures (methods only)

- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/figures/modelnet40/pointnet2_final/accuracy_delta_methods_only` — 3 PNG figures
- Line A, Line B, and two-line grouped Δ best overall (pp)
- Baseline not plotted as bar; y=0 reference only

## 3. Geometry delta figures (methods only)

- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/figures/modelnet40/geometry_delta_methods_only` — 4 PNG figures
- Line B Δ CD, HD, NUC, exact P2F vs Original baseline
- Original baseline as zero reference; metrics vs dense mesh surface reference

## 4. Final combined geometry-vs-classification figures

- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/figures/modelnet40/final_combined` — 4 PNG figures
- Line B scatter: geometry delta vs classification delta
- Highlights misalignment between geometry and classification leaders

## 5. Point cloud qualitative figures

- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/figures/modelnet40/pointcloud_examples` — 10 PNG comparison figures
- Classes: airplane, chair, table, car, sofa

## 6. Visualization conventions

- For **absolute accuracy figures**, baseline is shown because it is a real evaluated branch.
- For **delta figures**, baseline is not plotted as a method bar and is used only as the zero reference.
- **Geometry absolute values** are computed against the dense mesh surface reference.
- **Delta geometry figures** compare methods relative to the Original baseline.

## 7. Key observations

- **Line A:** Original baseline 1024 (91.95%) remains strongest; 4096-point upsampling does not improve classification.
- **Line B:** PU-Net (91.27%) is the only method slightly above the downsampled ×4 baseline (90.85%, +0.42 pp).
- **Geometry:** PU-GCN and PU-Net have the smallest CD delta vs Original; PU-GCN NUC is clearly elevated.
- **Alignment:** Geometry best ≠ classification best.

## 8. Failures

- None reported.

## 9. Total figures saved this run

- 16 figure artifacts (PNG/PDF)


## 10. Interactive point cloud viewers (v2)

- `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/figures/modelnet40/pointcloud_examples_interactive_v2` — 52 HTML viewers
- Plotly 3D scatter with rotate / zoom / pan
- Line A and Line B multi-panel grids (black + depth color modes)
- Dropdown switcher pages for presentation
- Single-method detail pages for airplane and chair
- **Interactive HTML viewers** are intended for qualitative inspection and presentation
- **Static figures** in `pointcloud_examples/` remain for thesis document insertion
- Interactive 3D viewers provide clearer structural comparison than orthographic static PNGs
- Audit report: `reports/modelnet40_interactive_pointcloud_v2_audit.md`

## PU-EdgeFormer geometry update

- Directory: `figures/modelnet40/geometry_delta_methods_only_with_pu_edgeformer/`
- PU-EdgeFormer geometry metrics were added after authenticity verification.
- PU-EdgeFormer PointNet++ classification is still pending.
- Delta geometry figures include PU-EdgeFormer but do not include baseline as a method bar.
- Absolute metrics use dense mesh surface / mesh / GT reference.
- **Line B geometry Δ** is vs **Original baseline 1024** (recovery of geometric quality).
- **Line B classification primary Δ** (future) is vs **Downsampled ×4 baseline 256**; secondary `gap_accuracy_vs_original_baseline` is separate.
- Comparison logic: `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`.

## PU-EdgeFormer PointNet++ classification update

- PU-EdgeFormer PointNet++ classification has been added.
- Line B classification uses Downsampled baseline as the primary reference.
- Line B also reports gap to Original baseline.
- Geometry delta remains relative to Original baseline.
- Figures: `figures/modelnet40/pointnet2_final_with_pu_edgeformer/`
- Reports: `reports/modelnet40_pointnet2_final_*_with_pu_edgeformer.*`
- Audit: `reports/modelnet40_pointnet2_pu_edgeformer_full_training_audit_20260716.md`
- DETECTOR_EVAL_STARTED=NO; KITTI_AP_EVAL_STARTED=NO

## Final visualization package (complete thesis-ready)

- Root: `figures/modelnet40/final_visualization_package/`
- Master index: `reports/modelnet40_complete_final_visualization_index.md`
- Audit: `reports/modelnet40_complete_final_visualization_audit.md`

### Logic (must not mix)

1. **Main geometry focus:** Line B delta vs Original baseline 1024 (methods-only bars; y=0 reference).
2. **Classification figures:** shown separately for Line A and Line B.
3. **Line B classification:** primary delta vs Downsampled ×4 baseline 256; secondary gap vs Original baseline 1024.
4. **Point cloud qualitative:** static PNG/PDF + interactive HTML both indexed.
5. Absolute geometry uses **dense mesh surface reference** (not called “baseline”).

### Key outputs

- Complete table: `reports/modelnet40_final_complete_geometry_pointnet2_comparison_with_pu_edgeformer.{csv,md}`
- Thesis summary: `reports/modelnet40_final_thesis_summary_table_with_pu_edgeformer.{csv,md}`
- Interpretation: `reports/modelnet40_final_geometry_pointnet2_comparison_interpretation_with_pu_edgeformer.md`
- Interactive index: `reports/modelnet40_interactive_pointcloud_final_index.md`

