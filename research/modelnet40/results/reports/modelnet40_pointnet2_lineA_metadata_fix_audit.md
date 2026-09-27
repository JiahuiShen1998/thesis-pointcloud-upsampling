# ModelNet40 PointNet++ Line A Metadata Fix Audit

- Generated at: 2026-07-07 20:20:00 UTC
- Metadata source: `datasets/modelnet40_original/metadata` (40 classes)
- Fix method: symlink only — **no `.npy` files modified**

## Fix applied (v2)

Per-method `metadata/` directory with only:

```
class_to_idx.json -> modelnet40_original/metadata/class_to_idx.json
idx_to_class.json -> modelnet40_original/metadata/idx_to_class.json
```

No `train_manifest.csv` / `test_manifest.csv` → `ModelNetNPYDataset` falls back to directory scan under `strict_4N/<method>/train|test`, loading 4096-pt upsampled `.npy` files.

## Audit summary

| branch | method | class_to_idx | class_count | manifests | symlink | sample pts | status |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| lineA_ear_4096 | EAR | yes | 40 | absent | yes | 4096 | **PASS** |
| lineA_pdans_4096 | PDANS | yes | 40 | absent | yes | 4096 | **PASS** |
| lineA_punet_4096 | PU-Net | yes | 40 | absent | yes | 4096 | **PASS** |
| lineA_pugcn_4096 | PU-GCN | yes | 40 | absent | yes | 4096 | **PASS** |

All four branches: **PASS** — ready for smoke retry.
