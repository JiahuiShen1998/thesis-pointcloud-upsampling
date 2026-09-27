# ModelNet40 Thesis Figure Index

- Generated at: 2026-07-13 14:32:51 UTC
- Root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/figures/modelnet40`

## PointNet++ Final Figures (thesis-ready)

| Figure | Source data | Caption | Thesis section |
| --- | --- | --- | --- |
| `pointnet2_final/accuracy_absolute/lineA_best_overall_accuracy_absolute.png` / `.pdf` | `modelnet40_pointnet2_lineA_classification_summary.csv` | Line A absolute best overall accuracy; baseline shown as real branch with dashed reference. | Results — Line A classification |
| `pointnet2_final/accuracy_absolute/lineB_best_overall_accuracy_absolute.png` / `.pdf` | `modelnet40_pointnet2_lineB_classification_summary.csv` | Line B absolute best overall accuracy; PU-Net only method above downsampled baseline. | Results — Line B classification |
| `pointnet2_final/accuracy_absolute/two_line_best_overall_accuracy_absolute.png` / `.pdf` | `modelnet40_pointnet2_final_two_line_classification_summary.csv` | Grouped two-line absolute best overall accuracy. | Results — Overview |
| `pointnet2_final/accuracy_delta_methods_only/lineA_delta_best_overall_vs_original_baseline_methods_only.png` / `.pdf` | `modelnet40_pointnet2_lineA_classification_summary.csv` | Line A Δ best overall vs Original baseline (pp); methods only, y=0 reference. | Results — Line A classification |
| `pointnet2_final/accuracy_delta_methods_only/lineB_delta_best_overall_vs_downsampled_baseline_methods_only.png` / `.pdf` | `modelnet40_pointnet2_lineB_classification_summary.csv` | Line B Δ best overall vs downsampled baseline (pp); PU-Net +0.42 pp. | Results — Line B classification |
| `pointnet2_final/accuracy_delta_methods_only/two_line_delta_best_overall_methods_only.png` / `.pdf` | `modelnet40_pointnet2_final_two_line_classification_summary.csv` | Two-line grouped Δ best overall (methods only). | Results — Overview |
| `geometry_delta_methods_only/lineB_delta_cd_vs_original_baseline_methods_only.png` / `.pdf` | `modelnet40_quality_metrics_thesis_table.csv` | Line B Δ CD vs Original baseline (methods only). | Results — Geometry quality |
| `geometry_delta_methods_only/lineB_delta_hd_vs_original_baseline_methods_only.png` / `.pdf` | `modelnet40_quality_metrics_thesis_table.csv` | Line B Δ HD vs Original baseline (methods only). | Results — Geometry quality |
| `geometry_delta_methods_only/lineB_delta_nuc_vs_original_baseline_methods_only.png` / `.pdf` | `modelnet40_quality_metrics_thesis_table.csv` | Line B Δ NUC vs Original baseline (methods only). | Results — Geometry quality |
| `geometry_delta_methods_only/lineB_delta_exact_p2f_vs_original_baseline_methods_only.png` / `.pdf` | `modelnet40_quality_metrics_thesis_table.csv` | Line B Δ exact P2F vs Original baseline (methods only). | Results — Geometry quality |
| `final_combined/lineB_delta_cd_vs_delta_accuracy.png` / `.pdf` | `modelnet40_geometry_vs_classification_final_summary.csv` | Line B Δ CD vs Δ classification accuracy scatter. | Discussion — Geometry vs classification |
| `final_combined/lineB_delta_hd_vs_delta_accuracy.png` / `.pdf` | `modelnet40_geometry_vs_classification_final_summary.csv` | Line B Δ HD vs Δ classification accuracy scatter. | Discussion — Geometry vs classification |
| `final_combined/lineB_delta_nuc_vs_delta_accuracy.png` / `.pdf` | `modelnet40_geometry_vs_classification_final_summary.csv` | Line B Δ NUC vs Δ classification accuracy scatter. | Discussion — Geometry vs classification |
| `final_combined/lineB_delta_p2f_vs_delta_accuracy.png` / `.pdf` | `modelnet40_geometry_vs_classification_final_summary.csv` | Line B Δ exact P2F vs Δ classification accuracy scatter. | Discussion — Geometry vs classification |

## Visualization conventions

- **Absolute accuracy figures:** baseline is shown because it is a real evaluated branch.
- **Delta figures (methods-only):** baseline is not plotted as a method bar; used only as y=0 reference.
- **Geometry absolute values:** computed against the dense mesh surface reference.
- **Delta geometry figures:** compare methods relative to the Original baseline.

## Qualitative point cloud examples

See `pointcloud_examples/` for Line A and Line B qualitative comparisons (airplane, chair, table, car, sofa).

## Interactive point cloud viewers (v2)

- Generated at: 2026-07-13 17:30:38 UTC
- Root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/figures/modelnet40/pointcloud_examples_interactive_v2`
- Purpose: qualitative inspection and presentation (rotate / zoom / pan)
- Static PNG/PDF figures in `pointcloud_examples/` remain for thesis document insertion

| HTML pattern | Description | Thesis section |
| --- | --- | --- |
| `pointcloud_examples_interactive_v2/lineA/lineA_<class>_<id>_interactive_grid_black.html` | Line A 5-panel grid, single-color markers | Qualitative — Line A |
| `pointcloud_examples_interactive_v2/lineA/lineA_<class>_<id>_interactive_grid_depth.html` | Line A 5-panel grid, depth-colored markers | Qualitative — Line A |
| `pointcloud_examples_interactive_v2/lineB/lineB_<class>_<id>_interactive_grid_black.html` | Line B 6-panel grid, single-color markers | Qualitative — Line B |
| `pointcloud_examples_interactive_v2/lineB/lineB_<class>_<id>_interactive_grid_depth.html` | Line B 6-panel grid, depth-colored markers | Qualitative — Line B |
| `pointcloud_examples_interactive_v2/dropdown/lineA_<class>_<id>_dropdown.html` | Line A dropdown method switcher | Presentation |
| `pointcloud_examples_interactive_v2/dropdown/lineB_<class>_<id>_dropdown.html` | Line B dropdown method switcher | Presentation |
| `pointcloud_examples_interactive_v2/single_method/lineA_<class>_<id>_<method>.html` | Line A single-method detail view | Presentation / supplementary |
| `pointcloud_examples_interactive_v2/single_method/lineB_<class>_<id>_<method>.html` | Line B single-method detail view | Presentation / supplementary |

Classes covered: airplane, chair, table, car, sofa (test split).

## Geometry delta methods-only (with PU-EdgeFormer)

| Figure | Source | Caption | Section |
|---|---|---|---|
| `geometry_delta_methods_only_with_pu_edgeformer/lineB_delta_cd_vs_original_baseline_methods_only_with_pu_edgeformer` | `modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv` | Line B Δ CD vs Original baseline (methods only, includes PU-EdgeFormer). | Results — Geometry quality |
| `geometry_delta_methods_only_with_pu_edgeformer/lineB_delta_hd_vs_original_baseline_methods_only_with_pu_edgeformer` | `modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv` | Line B Δ HD vs Original baseline (methods only, includes PU-EdgeFormer). | Results — Geometry quality |
| `geometry_delta_methods_only_with_pu_edgeformer/lineB_delta_nuc_vs_original_baseline_methods_only_with_pu_edgeformer` | `modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv` | Line B Δ NUC vs Original baseline (methods only, includes PU-EdgeFormer). | Results — Geometry quality |
| `geometry_delta_methods_only_with_pu_edgeformer/lineB_delta_exact_p2f_vs_original_baseline_methods_only_with_pu_edgeformer` | `modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv` | Line B Δ exact P2F vs Original baseline (methods only, includes PU-EdgeFormer). | Results — Geometry quality |

- Note: PU-EdgeFormer geometry metrics were added after authenticity verification. PU-EdgeFormer PointNet++ classification is still pending. Delta figures include PU-EdgeFormer but do not include baseline as a method bar.
- Comparison logic: Line B geometry Δ vs Original baseline 1024; Line B classification primary Δ vs Downsampled ×4 baseline 256; secondary `gap_accuracy_vs_original_baseline` is separate. See `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`.

## PU-EdgeFormer PointNet++ classification update

- PU-EdgeFormer PointNet++ classification has been added.
- Line B classification uses Downsampled baseline as the primary reference.
- Line B also reports gap to Original baseline.
- Geometry delta remains relative to Original baseline.
- Figures: `figures/modelnet40/pointnet2_final_with_pu_edgeformer/`
- Reports: `reports/modelnet40_pointnet2_final_*_with_pu_edgeformer.*`
- Audit: `reports/modelnet40_pointnet2_pu_edgeformer_full_training_audit_20260716.md`
- DETECTOR_EVAL_STARTED=NO; KITTI_AP_EVAL_STARTED=NO

## PointNet++ final (with PU-EdgeFormer)

| Figure stem | Source | Description | Suggested thesis location |
|---|---|---|---|
| `pointnet2_final_with_pu_edgeformer/accuracy_absolute/lineA_best_overall_accuracy_absolute_with_pu_edgeformer` | final classification with EdgeFormer | Line A absolute best overall including PU-EdgeFormer. | Results — Line A classification |
| `pointnet2_final_with_pu_edgeformer/accuracy_absolute/lineB_best_overall_accuracy_absolute_with_pu_edgeformer` | final classification with EdgeFormer | Line B absolute best overall including PU-EdgeFormer. | Results — Line B classification |
| `pointnet2_final_with_pu_edgeformer/accuracy_delta_methods_only/lineA_delta_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer` | final classification with EdgeFormer | Line A Δ vs Original baseline (methods only). | Results — Line A classification |
| `pointnet2_final_with_pu_edgeformer/accuracy_delta_methods_only/lineB_delta_best_overall_vs_downsampled_baseline_methods_only_with_pu_edgeformer` | final classification with EdgeFormer | Line B primary Δ vs Downsampled baseline (methods only). | Results — Line B classification |
| `pointnet2_final_with_pu_edgeformer/accuracy_delta_methods_only/lineB_gap_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer` | final classification with EdgeFormer | Line B secondary gap vs Original baseline (methods only). | Results — Line B classification |
| `pointnet2_final_with_pu_edgeformer/final_combined/lineB_delta_*_vs_delta_accuracy_with_pu_edgeformer` | geometry + classification with EdgeFormer | Geometry Δ (vs Original) vs classification Δ (vs Downsampled). | Discussion — Geometry vs classification |

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

