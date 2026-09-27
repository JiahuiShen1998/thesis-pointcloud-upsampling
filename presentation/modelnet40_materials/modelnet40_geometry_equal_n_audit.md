# Geometry Equal-N — Audit

- Generated: 2026-08-03 13:57:32 UTC
- Script: `scripts/compute_geometry_equal_n.py`
- Protocol: equal-cardinality CD/HD only
  - Line A: upsamplers 4096 vs `datasets/modelnet40_mesh_ref_4096`
  - Line B Downsampled: 256 vs `datasets/modelnet40_mesh_ref_256`
  - Line B upsamplers: 1024 vs `datasets/modelnet40_original`
- Split: test (N=2468)
- CD: mean NN L2(a→b) + mean NN L2(b→a)
- HD: max(max NN a→b, max NN b→a)
- Workers elapsed: 128.1s
- Failures: 0

## Summary

| Line | Method | Points | Reference | CD | HD | NUC | ΔNUC | N |
|---|---|---:|---|---:|---:|---:|---:|---:|
| A | Mesh-ref 4096 | 4096 | Mesh-ref 4096 | 0.000000 | 0.000000 | 0.641852 | 0.000000 | 2468 |
| A | EAR | 4096 | Mesh-ref 4096 | 0.054722 | 0.115396 | 0.815631 | 0.173778 | 2468 |
| A | PDANS | 4096 | Mesh-ref 4096 | 0.046821 | 0.113510 | 0.661176 | 0.019323 | 2468 |
| A | PU-Net | 4096 | Mesh-ref 4096 | 0.049719 | 0.111177 | 0.605211 | -0.036641 | 2468 |
| A | PU-GCN | 4096 | Mesh-ref 4096 | 0.042306 | 0.093219 | 0.520398 | -0.121454 | 2468 |
| A | PU-EdgeFormer | 4096 | Mesh-ref 4096 | 0.046666 | 0.110932 | 0.675435 | 0.033583 | 2468 |
| B | Mesh-ref 256 | 256 | Mesh-ref 256 | 0.000000 | 0.000000 | 2.126494 | 0.000000 | 2468 |
| B | Downsampled ×4 | 256 | Mesh-ref 256 | 0.133319 | 0.204914 | 2.093341 | -0.033153 | 2468 |
| B | Original | 1024 | Original 1024 | 0.000000 | 0.000000 | 1.083553 | 0.000000 | 2468 |
| B | EAR | 1024 | Original 1024 | 0.076743 | 0.193182 | 0.965292 | -0.118261 | 2468 |
| B | PDANS | 1024 | Original 1024 | 0.062696 | 0.190541 | 1.052826 | -0.030727 | 2468 |
| B | PU-Net | 1024 | Original 1024 | 0.064573 | 0.108319 | 1.080102 | -0.003452 | 2468 |
| B | PU-GCN | 1024 | Original 1024 | 0.055802 | 0.151403 | 1.523996 | 0.440442 | 2468 |
| B | PU-EdgeFormer | 1024 | Original 1024 | 0.061900 | 0.177572 | 1.122505 | 0.038951 | 2468 |

## Notes

- Unequal-cardinality CD/HD (4096 vs 1024, 256 vs 1024) are intentionally not reported here.
- Mesh-ref rows have CD=0 / HD=0 by self-comparison.
