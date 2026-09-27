# PDANS ×4 PointNet++ Dry-Run Report

- Generated: 2026-06-30
- Script: `scripts/dryrun_ear_x4_pointnet2.py` (generic dataloader dry-run)

## original_pdans_x4_pointnet2 — **PASS**

| Check | Result |
| --- | --- |
| Dataset | `datasets/modelnet40_original_up/pdans_x4` |
| Train count | 9843 |
| Test count | 2468 |
| Sample shape | (4096, 3) |
| Batch shape | (4, 4096, 3) |
| `num_point` | 4096 |
| `allow_resample` | false |
| NaN/Inf | none |

## downsampled50_pdans_x4_pointnet2 — **PASS**

| Check | Result |
| --- | --- |
| Dataset | `datasets/modelnet40_downsampled50_up/pdans_x4` |
| Train count | 9843 |
| Test count | 2468 |
| Sample shape | (2048, 3) |
| Batch shape | (4, 2048, 3) |
| `num_point` | 2048 |
| `allow_resample` | false |
| NaN/Inf | none |

## Conclusion

Both configs ready for Slurm training submission.
