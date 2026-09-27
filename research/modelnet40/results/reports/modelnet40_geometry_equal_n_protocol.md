# Mesh-sampled Equal-N Geometry Protocol

- Generated: 2026-08-03
- Scripts: `scripts/build_mesh_reference_points.py`, `scripts/compute_geometry_equal_n.py`

## Sampling

- Source: ModelNet40 `.off` via original manifests (`source_off`)
- Method: triangle **area-weighted** surface sampling + **unit-sphere** normalize (same as Original 1024 / `prepare_modelnet40.py`)
- Seed: `stable_seed(42, "mesh_ref", "{N}", "{split}/{class}/{shape_id}")`
- Outputs:
  - `datasets/modelnet40_mesh_ref_256/` (12311 shapes)
  - `datasets/modelnet40_mesh_ref_4096/` (12311 shapes)
- Build audits: `reports/modelnet40_mesh_ref_{256,4096}_build_audit.md` (failed=0)

## Equal-cardinality comparisons (test N=2468)

| Evaluated | Points | Reference |
|-----------|-------:|-----------|
| Line A upsamplers | 4096 | Mesh-ref 4096 |
| Line B Downsampled ×4 | 256 | Mesh-ref 256 |
| Line B upsamplers | 1024 | Original 1024 |

Self rows (Mesh-ref / Original): CD=0, HD=0.

## Outputs

- `reports/modelnet40_geometry_equal_n_summary.csv`
- `reports/modelnet40_geometry_equal_n_lineA.csv`
- `reports/modelnet40_geometry_equal_n_lineB.csv`
- `reports/modelnet40_geometry_equal_n_audit.md`
- `reports/modelnet40_geometry_equal_n_per_sample.csv`
- `reports/modelnet40_geometry_equal_n_meta.json`

## Out of scope

- PointNet++ retrain on mesh-256 / mesh-4096 (classification tables unchanged)
- Unequal-cardinality CD/HD ranking (4096 vs 1024, 256 vs 1024)
