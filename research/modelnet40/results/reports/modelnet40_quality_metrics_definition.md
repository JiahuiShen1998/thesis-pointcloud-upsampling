# ModelNet40 Geometric Quality Metrics Definition

- Generated at: 2026-07-06 22:04:21 UTC

## GT reference

All CD/HD comparisons use a **dense ground-truth point cloud** sampled from the original ModelNet40 `.off` mesh surface:

- Path: `quality_metrics/gt_reference_points_10000/{split}/{class}/{shape_id}.npy`
- Point count: **10000** per shape
- Sampling: triangle-area weighted surface sampling, unit-sphere normalization (same convention as `modelnet40_original`)
- Seed: global=42, per-sample deterministic via `stable_seed`

**Why not use original 1024 as GT?** If target = original 1024, then Original baseline CD/HD vs itself would be **0**, which is not informative. Dense mesh-sampled GT measures deviation from the underlying continuous surface.

## CD (Chamfer Distance) — lower is better

- `cd_forward` = mean nearest-neighbor **L2** distance from source → GT
- `cd_backward` = mean nearest-neighbor **L2** distance from GT → source
- `cd_total` = `cd_forward + cd_backward`
- Uses Euclidean (L2) distance, **not** squared L2.

## HD (Hausdorff Distance) — lower is better

- `hd_forward` = max nearest-neighbor L2 distance from source → GT
- `hd_backward` = max nearest-neighbor L2 distance from GT → source
- `hd_total` = max(`hd_forward`, `hd_backward`)

## P2F (Point-to-Surface Distance) — lower is better

Measures distance from each source point to the original `.off` mesh surface.

- `p2f_mean`, `p2f_median`, `p2f_95`, `p2f_max`
- Current status: **exact**
- Preferred: Open3D `RaycastingScene` exact unsigned distance
- Fallback: approximate P2F using 50000 mesh surface sample points + nearest-neighbor distance

## NUC (Neighborhood Uniformity Coefficient) — lower is more uniform

- Query centers: **128** (fixed seed per sample)
- Radii: **(0.02, 0.05, 0.1)**
- At each center, count neighbors within radius; compute coefficient of variation `std/mean` across centers
- `nuc_r1`, `nuc_r2`, `nuc_r3` at the three radii; `nuc_mean` = mean of the three

## Main vs supplementary groups

**Main groups** (primary paper comparison: Original baseline vs Downsampled×4 + Upsampling):

1. `original_baseline_vs_gt` — original 1024 pts
2. `downsampled_x4_up_{ear,pdans,pu_net,pu_gcn}_vs_gt` — Line B upsampled 1024 pts

**Supplementary groups**:

- `downsampled_x4_baseline_vs_gt` — downsampled 256 pts (degradation from downsampling)
- `original_up_{method}_vs_gt` — Line A upsampled 4096 pts
