# Mesh-ref PointNet++ baseline training status

- Status: **DONE** (2026-08-04 00:49 CEST)
- Jobs:
  - `1768767` — `mesh_ref_baseline_256` — finished; best overall **90.88%** (epoch 129)
  - `1768768` — `mesh_ref_baseline_4096` — finished; best overall **92.00%** (epoch 148)
- Protocol: PointNet++ `pointnet2_cls_ssg`, from scratch, 200 epochs, Adam lr=0.001, seed=42, `--no-allow-resample`
- Data: area-weighted mesh samples (`datasets/modelnet40_mesh_ref_{256,4096}`)
- Results: `reports/modelnet40_pointnet2_mesh_ref_baseline_results.{csv,md}`

## Comparisons

| Mesh-ref | Best Overall | Compare to | Compare Best | Δ |
|---|---:|---|---:|---:|
| Mesh-ref 256 | 90.88% | Downsampled ×4 256 | 90.85% | +0.03 pp |
| Mesh-ref 4096 | 92.00% | Original 1024 | 91.95% | +0.05 pp |
