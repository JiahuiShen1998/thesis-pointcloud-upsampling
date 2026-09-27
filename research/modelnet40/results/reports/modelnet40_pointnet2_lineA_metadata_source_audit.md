# ModelNet40 PointNet++ Line A Metadata Source Audit

- Generated at: 2026-07-07 20:20:00 UTC
- Purpose: select label map source for Line A upsampling v2 metadata fix

## Candidate sources

| path | class_to_idx | idx_to_class | class count | selected |
| --- | --- | --- | ---: | --- |
| `datasets/modelnet40_original/metadata/` | yes | yes | 40 | **yes** |
| `pointnet2_inputs/lineA_original_baseline/metadata/` | yes (same file) | yes (same file) | 40 | equivalent |
| `datasets/modelnet40_downsampled_x4/metadata/` | yes (same file) | yes (same file) | 40 | fallback only |

## Selected source

**Primary:** `datasets/modelnet40_original/metadata/`

- `class_to_idx.json` — 40 classes (airplane … xbox)
- `idx_to_class.json` — 40 entries, consistent reverse mapping
- Same inode as Line A baseline input (`pointnet2_inputs/lineA_original_baseline` → `modelnet40_original`)
- Line B downsampled ×4 metadata uses identical label map (627 / 707 bytes)

## Validation

```bash
python3 -c "
import json
from pathlib import Path
c2i = json.loads(Path('datasets/modelnet40_original/metadata/class_to_idx.json').read_text())
i2c = json.loads(Path('datasets/modelnet40_original/metadata/idx_to_class.json').read_text())
assert len(c2i) == 40 and len(i2c) == 40
"
```

Result: **PASS** — 40 classes, mapping valid.

## Manifest policy (v2)

- **Do not** symlink `train_manifest.csv` or `test_manifest.csv` to upsampling branches.
- Manifests in `modelnet40_original` point to 1024-pt baseline `.npy` paths.
- Dataloader must scan `strict_4N/<method>/train` and `test` directories directly (4096-pt upsampled clouds).

## Note on pointnet2_inputs

`pointnet2_inputs/lineA_original_up/{ear,pdans,pu_net,pu_gcn}` are symlinks to `datasets/lineA_original_up/strict_4N/<method>/`. Fixing `strict_4N/<method>/metadata/` is sufficient for smoke jobs.
