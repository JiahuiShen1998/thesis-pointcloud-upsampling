# Step 2 — ModelNet40 Prepare Report

- Generated at: 2026-06-20 16:59:30 CEST
- Raw ModelNet40 path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/data/raw/ModelNet40`
- Output path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original`
- Random seed: 42
- Sampling: 1024 points per shape (triangle area weighted when faces exist)
- Normalization: center + unit sphere
- Workers: 8
- Elapsed seconds: 153.7

## Summary

- Number of classes: **40**
- Train samples: **9843**
- Test samples: **2468**
- Total successful samples: **12311**
- Failed files: **0**

## Audit

- Checked `.npy` files: 12311
- Shape mismatches (expected `(1024, 3)`): 0
- Non-finite values: 0
- Max radius min/mean/max: 1.000000 / 1.000000 / 1.000000
- Unit sphere check: PASS

## Per-split class counts

### train

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

### test

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

## Next steps

1. Run `scripts/check_modelnet40_original.py` to validate the processed dataset.
2. Proceed to Step 3: clone and prepare PointNet++ classifier.
3. Use `datasets/modelnet40_original/` as Line A Original baseline input.
