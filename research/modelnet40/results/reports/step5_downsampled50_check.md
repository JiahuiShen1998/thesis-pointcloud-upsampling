# Step 5 — ModelNet40 Downsampled50 Check

- Generated at: 2026-06-20 19:06:01 CEST
- Original root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original`
- Downsampled root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled50`
- Seed: 42

## Counts

- Original train/test: 9843 / 2468
- Downsampled train/test: 9843 / 2468
- Classes: 40 (expected 40)

## Validation checks

| check | result |
| --- | --- |
| train_count_match_original | PASS |
| test_count_match_original | PASS |
| total_count_ok | PASS |
| class_count_ok | PASS |
| per_class_counts_match | PASS |
| shape_ok | PASS |
| nan_ok | PASS |
| inf_ok | PASS |
| subset_membership_ok | PASS |
| deterministic_repro_ok | PASS |

## Random subset verification

| file | subset_of_original | exact_repro |
| --- | --- | --- |
| `test/chair/chair_0933.npy` | yes | yes |
| `train/bookshelf/bookshelf_0405.npy` | yes | yes |
| `train/airplane/airplane_0410.npy` | yes | yes |
| `test/tv_stand/tv_stand_0346.npy` | yes | yes |
| `train/flower_pot/flower_pot_0137.npy` | yes | yes |
| `train/desk/desk_0152.npy` | yes | yes |
| `train/cup/cup_0014.npy` | yes | yes |
| `train/bottle/bottle_0295.npy` | yes | yes |
| `test/toilet/toilet_0440.npy` | yes | yes |
| `train/bookshelf/bookshelf_0260.npy` | yes | yes |
| `test/mantel/mantel_0307.npy` | yes | yes |
| `test/tv_stand/tv_stand_0332.npy` | yes | yes |
| `train/tv_stand/tv_stand_0025.npy` | yes | yes |
| `train/bookshelf/bookshelf_0005.npy` | yes | yes |
| `train/wardrobe/wardrobe_0022.npy` | yes | yes |
| `train/range_hood/range_hood_0038.npy` | yes | yes |
| `train/airplane/airplane_0521.npy` | yes | yes |
| `train/airplane/airplane_0489.npy` | yes | yes |
| `train/bookshelf/bookshelf_0116.npy` | yes | yes |
| `train/cone/cone_0106.npy` | yes | yes |

## Overall

**PASS** — ready for Step 6 after Step 4 completes.
