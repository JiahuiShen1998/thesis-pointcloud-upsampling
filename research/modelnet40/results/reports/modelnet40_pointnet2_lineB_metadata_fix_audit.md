# ModelNet40 PointNet++ Line B Metadata Fix Audit

- Generated at: 2026-07-07 12:40:00 UTC
- Metadata source: `datasets/modelnet40_downsampled_x4/metadata` (40 classes, train/test manifests)
- Fix method: symlink only — **no `.npy` files modified**

## Fix applied

**v1 (retry round 1 — failed):** symlink entire `metadata/` dir → manifests loaded 256-pt baseline paths.

**v2 (retry round 2 — PASS):** per-method `metadata/` directory with only:

```
class_to_idx.json -> modelnet40_downsampled_x4/metadata/class_to_idx.json
idx_to_class.json -> modelnet40_downsampled_x4/metadata/idx_to_class.json
```

No `train_manifest.csv` / `test_manifest.csv` → `ModelNetNPYDataset` falls back to directory scan under `strict_N/<method>/train|test`, loading 1024-pt upsampled `.npy` files.

## Audit summary

| branch | method | class_to_idx | class_count | manifests | symlink | status |
| --- | --- | --- | ---: | --- | --- | --- |
| lineB_ear_1024 | EAR | yes | 40 | train+test | yes | PASS |
| lineB_pdans_1024 | PDANS | yes | 40 | train+test | yes | PASS |
| lineB_punet_1024 | PU-Net | yes | 40 | train+test | yes | PASS |
| lineB_pugcn_1024 | PU-GCN | yes | 40 | train+test | yes | PASS |

All four branches: **PASS** — ready for smoke retry.
