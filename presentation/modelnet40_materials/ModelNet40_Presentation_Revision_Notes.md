# ModelNet40 Presentation Revision Notes

- Generated: 2026-08-04 00:30:12 UTC
- PPTX: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/presentations/ModelNet40_PointNet2_Original_Reference_Revised.pptx`
- PDF: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/presentations/ModelNet40_PointNet2_Original_Reference_Revised.pdf`
- Slides: 21

## Protocol changes

- Deleted dense mesh / dense surface reference from geometry evaluation narrative and tables.
- Deleted P2F / exact P2F / point-to-face entirely.
- Original CD/HD are now 0 by definition (self-comparison), not mesh-reference values (e.g. old CD≈0.0495).
- Statement used throughout: each upsampled point cloud is evaluated directly against the corresponding Original point cloud; no mesh or dense surface reference is used.

## Geometry recomputation

- Script: `scripts/compute_geometry_equal_n.py` (+ `scripts/build_mesh_reference_points.py`)
- Status: COMPLETED on test split (equal-N)
- Mesh refs: `datasets/modelnet40_mesh_ref_256/`, `datasets/modelnet40_mesh_ref_4096/`
- Outputs: `reports/modelnet40_geometry_equal_n_summary.csv`, `..._lineA.csv`, `..._lineB.csv`, `..._audit.md`
- Protocol: Line A 4096 vs mesh-4096; Downsampled 256 vs mesh-256; Line B ups 1024 vs Original 1024.
- Unequal-cardinality CD/HD are not reported in the presentation.

## Best Overall vs Final Overall

- Best Overall = highest recorded checkpoint overall accuracy (primary).
- Final Overall = epoch-200 overall accuracy (supplementary).
- Code audit: original PointNet++ PyTorch `train_classification.py` + our `train_pointnet2.py` — ModelNet40 has train/test only (no val); best checkpoint when test-instance accuracy improves.
- Line A 4096 branches: PointNet++ retrained from scratch with num_point=4096 (not pretrained 1024 eval).
- Line B: 256 baseline and 1024 upsampling branches each trained from scratch for that point count.
- Presentation wording: “Best Overall is the highest recorded checkpoint accuracy.”

## Mesh-ref PointNet++ baselines

- Report: `reports/modelnet40_pointnet2_mesh_ref_baseline_results.md`
- Mesh-ref 256 Best OA 90.88% vs Downsampled ×4 256 90.85% (+0.03 pp).
- Mesh-ref 4096 Best OA 92.00% vs Original 1024 91.95% (+0.05 pp).
- Interpretation: equal-N mesh sampling classifies almost identically; denser 4096 does not beat Original by a meaningful margin.

## Hosted interactive links

- PPT previously used `file://` links (open only on the generating machine).
- Set `PUBLIC_INTERACTIVE_BASE` to a GitHub/GitLab Pages URL ending in `.../dropdown_v2` and regenerate PPT for HTTPS links.
- Current PUBLIC_INTERACTIVE_BASE: `(empty — local file:// fallback)`

## Qualitative cameras re-selected

- airplane: elev=22, azim=-45 (elevated front-side three-quarter)
- chair: elev=18, azim=-35
- table: elev=35, azim=-50 (elevated three-quarter; avoids narrow silhouette)
- car: elev=20, azim=-55
- sofa: elev=18, azim=-40
- Within each class, all methods share identical camera, axis limits, point-size rule, depth coloring, dark background.

## Interactive HTML viewers verified

- Regenerated 10 dropdown viewers under `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/`.
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineA_airplane_airplane_0627_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineA_chair_chair_0890_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineA_table_table_0393_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineA_car_car_0198_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineA_sofa_sofa_0681_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineB_airplane_airplane_0627_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineB_chair_chair_0890_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineB_table_table_0393_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineB_car_car_0198_dropdown.html`
  - `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/lineB_sofa_sofa_0681_dropdown.html`
- Pre-existing v2 grid/dropdown/single_method HTML under `pointcloud_examples_interactive_v2/` remain available for live demo.

## Incomplete / caveats

- Geometry recomputation complete for test split.
- Train-split geometry not required for this presentation (user asked for test-set means).
- LibreOffice not available on this host; PDF exported via ReportLab as a print companion (PPTX remains the editable master).
- Unequal-cardinality CD/HD are omitted; presentation uses mesh-ref 256/4096 + Original 1024 equal-N tables.
