# Step 5 — ModelNet40 Downsampled50 Report

- Generated at: 2026-06-20 19:05:53 CEST
- Input: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original`
- Output: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled50`
- Random seed: 42
- Original points per sample: 1024
- Downsampled points per sample: 512
- Elapsed seconds: 7.2

## Summary

- Train samples: **9843**
- Test samples: **2468**
- Total successful: **12311**
- Failed samples: **0**
- Class count: **40**

## Audit

- Shape `(512, 3)` verified: 12311/12311
- Non-finite outputs: 0
- Status: PASS

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

## Next step

Wait for Step 4 Original baseline to finish, then proceed to Step 6: Downsampled50 baseline training.
