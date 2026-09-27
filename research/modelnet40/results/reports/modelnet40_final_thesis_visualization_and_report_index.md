# ModelNet40 Final Thesis Visualization and Report Index

- Generated at: 2026-07-13 14:32:51 UTC
- Project root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

## 1. PointNet++ final classification report

- Markdown: `reports/modelnet40_pointnet2_final_classification_report.md`
- CSV: `reports/modelnet40_pointnet2_final_classification_report.csv`

## 2. PointNet++ final interpretation report

- `reports/modelnet40_pointnet2_final_classification_interpretation.md`

## 3. PointNet++ absolute accuracy figures

- Directory: `figures/modelnet40/pointnet2_final/accuracy_absolute/`
- `lineA_best_overall_accuracy_absolute.png` / `.pdf`
- `lineB_best_overall_accuracy_absolute.png` / `.pdf`
- `two_line_best_overall_accuracy_absolute.png` / `.pdf`

## 4. PointNet++ delta methods-only figures

- Directory: `figures/modelnet40/pointnet2_final/accuracy_delta_methods_only/`
- `lineA_delta_best_overall_vs_original_baseline_methods_only.png` / `.pdf`
- `lineB_delta_best_overall_vs_downsampled_baseline_methods_only.png` / `.pdf`
- `two_line_delta_best_overall_methods_only.png` / `.pdf`

## 5. Geometry delta methods-only figures

- Directory: `figures/modelnet40/geometry_delta_methods_only/`
- `lineB_delta_cd_vs_original_baseline_methods_only.png` / `.pdf`
- `lineB_delta_hd_vs_original_baseline_methods_only.png` / `.pdf`
- `lineB_delta_nuc_vs_original_baseline_methods_only.png` / `.pdf`
- `lineB_delta_exact_p2f_vs_original_baseline_methods_only.png` / `.pdf`

## 6. Geometry vs PointNet++ final combined figures

- Directory: `figures/modelnet40/final_combined/`
- `lineB_delta_cd_vs_delta_accuracy.png` / `.pdf`
- `lineB_delta_hd_vs_delta_accuracy.png` / `.pdf`
- `lineB_delta_nuc_vs_delta_accuracy.png` / `.pdf`
- `lineB_delta_p2f_vs_delta_accuracy.png` / `.pdf`

## 7. Existing point cloud qualitative figures

- Directory: `figures/modelnet40/pointcloud_examples/`
- Line A and Line B comparisons for airplane, chair, table, car, sofa

## 8. Captions

- `reports/modelnet40_thesis_figure_captions.md`

## 9. Figure index

- `figures/modelnet40/figure_index.md`

## 10. Visualization summary

- `reports/modelnet40_visualization_summary.md`

## 11. Baseline/reference visualization guide

- `reports/modelnet40_baseline_reference_visualization_fix.md`

## 12. Supporting classification tables

- `reports/modelnet40_thesis_table_classification_lineA.md`
- `reports/modelnet40_thesis_table_classification_lineB.md`
- `reports/modelnet40_thesis_table_classification_combined.md`
- `reports/modelnet40_pointnet2_final_two_line_classification_summary.csv`

## 13. Supporting geometry tables

- `reports/modelnet40_quality_metrics_thesis_table.csv`
- `reports/modelnet40_geometry_vs_classification_final_summary.csv`

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

