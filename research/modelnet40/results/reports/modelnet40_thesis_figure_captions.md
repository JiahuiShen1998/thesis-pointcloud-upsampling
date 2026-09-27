# ModelNet40 Thesis Figure Captions

- Generated at: 2026-07-13 14:32:51 UTC
- Style: concise, thesis-ready; classification deltas in percentage points (pp)

## PointNet++ absolute accuracy figures

**Figure: Line A absolute best overall accuracy** (`pointnet2_final/accuracy_absolute/lineA_best_overall_accuracy_absolute`)

Best overall test accuracy of PointNet++ on ModelNet40 under Line A (Original 1024 baseline vs Original + Upsampling at 4096 points). The Original baseline is shown as a real training branch and used as the Line A reference (dashed line at 91.95%). None of the upsampling methods exceed this reference.

**Figure: Line B absolute best overall accuracy** (`pointnet2_final/accuracy_absolute/lineB_best_overall_accuracy_absolute`)

Best overall test accuracy under Line B (Downsampled ×4 baseline at 256 points vs upsampling to 1024 points). The dashed line marks the downsampled baseline (90.85%). PU-Net is the only method slightly above the downsampled baseline.

**Figure: Two-line absolute accuracy** (`pointnet2_final/accuracy_absolute/two_line_best_overall_accuracy_absolute`)

Grouped comparison of best overall accuracy across Line A and Line B. Line A baseline = Original baseline 1024; Line B baseline = Downsampled x4 baseline 256.

## PointNet++ delta accuracy figures (methods only)

**Figure: Line A delta best overall** (`pointnet2_final/accuracy_delta_methods_only/lineA_delta_best_overall_vs_original_baseline_methods_only`)

Change in best overall accuracy relative to the Original 1024 baseline (percentage points). Only upsampling methods are plotted; the Original baseline is the y=0 reference, not a bar. All methods show negative deltas.

**Figure: Line B delta best overall** (`pointnet2_final/accuracy_delta_methods_only/lineB_delta_best_overall_vs_downsampled_baseline_methods_only`)

Change in best overall accuracy relative to the Downsampled ×4 baseline (percentage points). Only upsampling methods are plotted. PU-Net shows +0.42 pp; other methods are negative.

**Figure: Two-line delta best overall** (`pointnet2_final/accuracy_delta_methods_only/two_line_delta_best_overall_methods_only`)

Grouped comparison of Δ best overall for Line A (vs Original baseline) and Line B (vs downsampled baseline). Baselines are used only as zero references in this delta plot.

## Geometry delta figures (methods only)

**Figure: Line B geometry deltas** (`geometry_delta_methods_only/lineB_delta_*_vs_original_baseline_methods_only`)

Geometric quality deltas for Line B upsampling methods relative to the Original 1024 baseline. The Original baseline is used as the zero reference and is not plotted as a method bar. Absolute metrics were computed with respect to the dense mesh surface reference. PU-GCN and PU-Net have the smallest CD deltas; PU-GCN NUC is clearly elevated.

## Geometry vs classification (final combined)

**Figure: Line B geometry delta vs classification delta** (`final_combined/lineB_delta_*_vs_delta_accuracy`)

Scatter plots comparing **geometry Δ vs Original baseline 1024** (x-axis) with **classification primary Δ vs Downsampled ×4 baseline 256** (y-axis, pp). These two references are intentionally different and must not be mixed. Only Line B upsampling methods are shown. PU-GCN has the smallest CD delta but not the highest accuracy delta; PU-Net has the highest accuracy delta. Geometric quality and classification are not perfectly aligned.

## Qualitative point cloud examples

**Figure: Point cloud comparisons** (`pointcloud_examples/lineA_*`, `pointcloud_examples/lineB_*`)

Qualitative comparison of test-set point clouds under Line A and Line B protocols. Identical viewpoint and axis limits across methods.


## Interactive point cloud viewers (v2)

**Figure: Line B interactive grid** (`pointcloud_examples_interactive_v2/lineB/lineB_<class>_<id>_interactive_grid_depth.html`)

Interactive 3D comparison of test-set point clouds under Line B: Original 1024, Downsampled ×4 256, and four upsampling methods at 1024 points. All subplots share identical axis limits and initial camera for fair visual comparison. Black and depth-colored variants are available.

**Figure: Line A interactive grid** (`pointcloud_examples_interactive_v2/lineA/lineA_<class>_<id>_interactive_grid_depth.html`)

Interactive 3D comparison under Line A: Original 1024 vs four upsampling methods at 4096 points.

**Figure: Dropdown viewer** (`pointcloud_examples_interactive_v2/dropdown/`)

Single-scene presentation viewer with dropdown method selection; camera remains stable when switching methods.

**Figure: Single-method detail** (`pointcloud_examples_interactive_v2/single_method/`)

Full-size interactive view for individual methods (airplane and chair). Intended for detailed qualitative inspection during presentation or review.

Interactive HTML viewers are for qualitative inspection and presentation; static PNG/PDF figures remain for thesis document insertion.

## Geometry delta methods-only with PU-EdgeFormer

Line B geometry Δ is vs **Original baseline 1024** (recovery of geometric quality). The Original baseline is the zero reference only (not a method bar). Absolute metrics use the dense mesh surface / mesh / GT reference. Classification primary Δ remains vs Downsampled ×4 baseline 256; secondary `gap_accuracy_vs_original_baseline` is separate when classification exists.

Methods on the x-axis: EAR, PDANS, PU-Net, PU-GCN, PU-EdgeFormer.
PU-EdgeFormer geometry metrics were added after authenticity verification.
PU-EdgeFormer PointNet++ classification is still pending.

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

## Final visualization package captions

**Figure: Line B geometry delta (final package)** (`final_visualization_package/geometry_lineB_focus/lineB_delta_*_vs_original_baseline_methods_only_final`)

Line B geometric quality deltas for upsampling methods relative to the Original 1024 baseline. The Original baseline is the zero reference only (not a method bar). Absolute metrics use the dense mesh surface reference.

**Figure: PointNet++ Line B primary delta (final package)** (`final_visualization_package/pointnet2_lineB/lineB_delta_best_overall_vs_downsampled_baseline_methods_only_final`)

Change in best overall accuracy relative to the Downsampled ×4 baseline (pp). Methods only; baseline is y=0.

**Figure: PointNet++ Line B secondary gap (final package)** (`final_visualization_package/pointnet2_lineB/lineB_gap_best_overall_vs_original_baseline_methods_only_final`)

Recovery gap to Original baseline 1024 (pp). Secondary column only; not the primary ranking metric.

**Figure: Geometry vs classification combined (final package)** (`final_visualization_package/combined_geometry_vs_classification/lineB_delta_*_vs_delta_accuracy_final`)

Scatter comparing geometry Δ (vs Original) with classification primary Δ (vs Downsampled baseline). Different references by design.

