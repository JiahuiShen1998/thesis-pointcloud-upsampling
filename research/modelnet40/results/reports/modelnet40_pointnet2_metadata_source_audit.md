# ModelNet40 PointNet++ Metadata Source Audit

- Generated at: 2026-07-07 12:40:00 UTC

## Selected metadata source

**Primary source:** `datasets/modelnet40_downsampled_x4/metadata/`

Resolved path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4/metadata`

## Files present

| file | exists | note |
| --- | --- | --- |
| `class_to_idx.json` | yes | 40 classes |
| `idx_to_class.json` | yes | reverse mapping |
| `train_manifest.csv` | yes | 9843 samples — used by `ModelNetNPYDataset` |
| `test_manifest.csv` | yes | 2468 samples — used by `ModelNetNPYDataset` |
| `modelnet40_shape_names.txt` | no | not required by current dataloader |
| `train_files.txt` | no | not required; manifests used instead |
| `test_files.txt` | no | not required; manifests used instead |

## Baseline via pointnet2_inputs

`pointnet2_inputs/lineB_downsampled_x4_baseline` → symlink to `datasets/modelnet40_downsampled_x4`

Metadata accessible at `pointnet2_inputs/lineB_downsampled_x4_baseline/metadata/class_to_idx.json` — **same source**.

## Alignment

- Class count: **40**
- Train samples in manifest: **9843**
- Test samples in manifest: **2468**
- Total: **12311** — matches ModelNet40 x4 protocol and upsampling strict_N outputs

## Conclusion

`datasets/modelnet40_downsampled_x4/metadata` is a valid metadata source for Line B upsampling branches. Symlink reuse is safe; no `.npy` files modified.
