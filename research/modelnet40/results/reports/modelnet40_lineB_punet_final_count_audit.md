# Line B PU-Net Final Count Audit

- Generated at: 2026-07-05 16:49:29 UTC
- Resume job **1731611**: **COMPLETED**

## Paths

- raw: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineB_downsampled_x4_up/raw/pu_net`
- strict_N: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineB_downsampled_x4_up/strict_N/pu_net`

## Counts

| split | raw | strict_N | expected |
| --- | ---: | ---: | ---: |
| train | 9843 | 9843 | 9843 |
| test | 2468 | 2468 | 2468 |
| total | 12311 | 12311 | 12311 |

## Quality checks

- every strict output shape = (1024, 3): **PASS** (0 bad)
- NaN: **PASS** (0)
- Inf: **PASS** (0)
- missing samples: **0**

## Status: **PASS**