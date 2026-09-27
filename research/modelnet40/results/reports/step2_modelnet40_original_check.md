# Step 2 — ModelNet40 Original Dataset Check

- Generated at: 2026-06-20 16:59:40 CEST
- Data root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original`

## Counts

- Train `.npy` files: **9843** (expected 9843)
- Test `.npy` files: **2468** (expected 2468)
- Total: **12311** (expected 12311)
- Classes: **40** (expected 40)

## Validation checks

| check | result |
| --- | --- |
| train_count_ok | PASS |
| test_count_ok | PASS |
| total_count_ok | PASS |
| class_count_ok | PASS |
| shape_ok | PASS |
| nan_ok | PASS |
| inf_ok | PASS |
| unit_sphere_ok | PASS |

## Random sample inspection

| file | shape | max_radius | nan | inf |
| --- | --- | ---: | --- | --- |
| `test/chair/chair_0933.npy` | (1024, 3) | 1.000000 | no | no |
| `train/bookshelf/bookshelf_0405.npy` | (1024, 3) | 1.000000 | no | no |
| `train/airplane/airplane_0410.npy` | (1024, 3) | 1.000000 | no | no |
| `test/tv_stand/tv_stand_0346.npy` | (1024, 3) | 1.000000 | no | no |
| `train/flower_pot/flower_pot_0137.npy` | (1024, 3) | 1.000000 | no | no |
| `train/desk/desk_0152.npy` | (1024, 3) | 1.000000 | no | no |
| `train/cup/cup_0014.npy` | (1024, 3) | 1.000000 | no | no |
| `train/bottle/bottle_0295.npy` | (1024, 3) | 1.000000 | no | no |
| `test/toilet/toilet_0440.npy` | (1024, 3) | 1.000000 | no | no |
| `train/bookshelf/bookshelf_0260.npy` | (1024, 3) | 1.000000 | no | no |
| `test/mantel/mantel_0307.npy` | (1024, 3) | 1.000000 | no | no |
| `test/tv_stand/tv_stand_0332.npy` | (1024, 3) | 1.000000 | no | no |
| `train/tv_stand/tv_stand_0025.npy` | (1024, 3) | 1.000000 | no | no |
| `train/bookshelf/bookshelf_0005.npy` | (1024, 3) | 1.000000 | no | no |
| `train/wardrobe/wardrobe_0022.npy` | (1024, 3) | 1.000000 | no | no |
| `train/range_hood/range_hood_0038.npy` | (1024, 3) | 1.000000 | no | no |
| `train/airplane/airplane_0521.npy` | (1024, 3) | 1.000000 | no | no |
| `train/airplane/airplane_0489.npy` | (1024, 3) | 1.000000 | no | no |
| `train/bookshelf/bookshelf_0116.npy` | (1024, 3) | 1.000000 | no | no |
| `train/cone/cone_0106.npy` | (1024, 3) | 1.000000 | no | no |

## train per-class counts

| class | count |
| --- | ---: |
| airplane | 626 |
| bathtub | 106 |
| bed | 515 |
| bench | 173 |
| bookshelf | 572 |
| bottle | 335 |
| bowl | 64 |
| car | 197 |
| chair | 889 |
| cone | 167 |
| cup | 79 |
| curtain | 138 |
| desk | 200 |
| door | 109 |
| dresser | 200 |
| flower_pot | 149 |
| glass_box | 171 |
| guitar | 155 |
| keyboard | 145 |
| lamp | 124 |
| laptop | 149 |
| mantel | 284 |
| monitor | 465 |
| night_stand | 200 |
| person | 88 |
| piano | 231 |
| plant | 240 |
| radio | 104 |
| range_hood | 115 |
| sink | 128 |
| sofa | 680 |
| stairs | 124 |
| stool | 90 |
| table | 392 |
| tent | 163 |
| toilet | 344 |
| tv_stand | 267 |
| vase | 475 |
| wardrobe | 87 |
| xbox | 103 |

## test per-class counts

| class | count |
| --- | ---: |
| airplane | 100 |
| bathtub | 50 |
| bed | 100 |
| bench | 20 |
| bookshelf | 100 |
| bottle | 100 |
| bowl | 20 |
| car | 100 |
| chair | 100 |
| cone | 20 |
| cup | 20 |
| curtain | 20 |
| desk | 86 |
| door | 20 |
| dresser | 86 |
| flower_pot | 20 |
| glass_box | 100 |
| guitar | 100 |
| keyboard | 20 |
| lamp | 20 |
| laptop | 20 |
| mantel | 100 |
| monitor | 100 |
| night_stand | 86 |
| person | 20 |
| piano | 100 |
| plant | 100 |
| radio | 20 |
| range_hood | 100 |
| sink | 20 |
| sofa | 100 |
| stairs | 20 |
| stool | 20 |
| table | 100 |
| tent | 20 |
| toilet | 100 |
| tv_stand | 100 |
| vase | 100 |
| wardrobe | 20 |
| xbox | 20 |

## Overall

**PASS** — dataset is ready for PointNet++ Step 3.
