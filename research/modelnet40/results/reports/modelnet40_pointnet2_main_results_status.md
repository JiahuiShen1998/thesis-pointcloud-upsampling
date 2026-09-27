# ModelNet40 PointNet++ — Main Results Status

- Updated: 2026-07-01 17:45 CEST
- CSV: `reports/modelnet40_pointnet2_main_results_status.csv`

## Completed

| Experiment | Line | Best acc | Final acc | `num_point` |
| --- | --- | ---: | ---: | ---: |
| original_baseline | A | 91.95% | 91.31% | 1024 |
| downsampled50_native512_pointnet2 | B | 91.26% | 91.14% | 512 |
| original_ear_x4_pointnet2 | A | 91.48% | 90.85% | 4096 |
| downsampled50_ear_x4_pointnet2 | B | 90.61% | 89.76% | 2048 |
| original_pdans_x4_pointnet2 | A | 91.62% | 90.87% | 4096 |
| downsampled50_pdans_x4_pointnet2 | B | 91.47% | 90.99% | 2048 |

## In progress

| Experiment | Stage |
| --- | --- |
| original_punet_x4_pointnet2 | PU-Net ×4 full generation running (job 1725588) |
| downsampled50_punet_x4_pointnet2 | PU-Net ×4 full generation running (job 1725589) |

## Blocked / pending

| Experiment | Blocker |
| --- | --- |
| original_pugcn_x4_pointnet2 | after PU-Net completes |
| downsampled50_pugcn_x4_pointnet2 | after PU-Net completes |

## Main protocol ×4

| Line | Baseline | EAR ×4 | PDANS ×4 | PU-Net ×4 |
| --- | --- | --- | --- | --- |
| A | original_baseline ✓ | original_ear_x4 ✓ | original_pdans_x4 ✓ | **running** |
| B | downsampled50_native512 ✓ | downsampled50_ear_x4 ✓ | downsampled50_pdans_x4 ✓ | **running** |

## Interpretation

See `reports/modelnet40_current_result_interpretation.md` for EAR/PDANS delta table. Next method after PU-Net: **PU-GCN**.
