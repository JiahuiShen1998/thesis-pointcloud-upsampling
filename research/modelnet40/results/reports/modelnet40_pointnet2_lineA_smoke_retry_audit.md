# ModelNet40 PointNet++ Line A Smoke Retry Audit

- Generated at: 2026-07-07 20:20:41 UTC
- Retry submit time: 2026-07-07 20:19:17 UTC
- Prerequisite: Line A v2 metadata fix (`reports/modelnet40_pointnet2_lineA_metadata_fix_audit.md`)
- Overall retry result: **4 / 4 PASS**

## Summary

| branch | method | old job | retry job | expected | actual | state | exit | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| lineA_ear_4096 | EAR | 1733476 | 1733487 | 4096 | 4096 | COMPLETED | 0:0 | **PASS** |
| lineA_pdans_4096 | PDANS | 1733477 | 1733488 | 4096 | 4096 | COMPLETED | 0:0 | **PASS** |
| lineA_punet_4096 | PU-Net | 1733478 | 1733489 | 4096 | 4096 | COMPLETED | 0:0 | **PASS** |
| lineA_pugcn_4096 | PU-GCN | 1733479 | 1733490 | 4096 | 4096 | COMPLETED | 0:0 | **PASS** |

## PASS detail: lineA_ear_4096

- Log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_smoke/lineA_ear_4096/slurm_1733487.out`
- Batch shape: `(8, 3, 4096)`
- First loss: **3.573660** (finite=true)
- Forward / backward / eval: PASS / PASS / PASS
- metrics.json: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_up/ear/metrics.json`
- Note: retry after metadata v2 fix; num_point=4096 allow_resample=False; metrics.json has final_test_overall_accuracy/final_test_class_accuracy; no NaN/Inf/OOM/metadata/dataloader errors observed

## PASS detail: lineA_pdans_4096

- Log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_smoke/lineA_pdans_4096/slurm_1733488.out`
- Batch shape: `(8, 3, 4096)`
- First loss: **3.693945** (finite=true)
- Forward / backward / eval: PASS / PASS / PASS
- metrics.json: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_up/pdans/metrics.json`
- Note: retry after metadata v2 fix; num_point=4096 allow_resample=False; metrics.json has final_test_overall_accuracy/final_test_class_accuracy; no NaN/Inf/OOM/metadata/dataloader errors observed

## PASS detail: lineA_punet_4096

- Log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_smoke/lineA_punet_4096/slurm_1733489.out`
- Batch shape: `(8, 3, 4096)`
- First loss: **3.809608** (finite=true)
- Forward / backward / eval: PASS / PASS / PASS
- metrics.json: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_up/pu_net/metrics.json`
- Note: retry after metadata v2 fix; num_point=4096 allow_resample=False; metrics.json has final_test_overall_accuracy/final_test_class_accuracy; no NaN/Inf/OOM/metadata/dataloader errors observed

## PASS detail: lineA_pugcn_4096

- Log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/pointnet2_smoke/lineA_pugcn_4096/slurm_1733490.out`
- Batch shape: `(8, 3, 4096)`
- First loss: **3.825351** (finite=true)
- Forward / backward / eval: PASS / PASS / PASS
- metrics.json: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke/lineA_original_up/pu_gcn/metrics.json`
- Note: retry after metadata v2 fix; num_point=4096 allow_resample=False; metrics.json has final_test_overall_accuracy/final_test_class_accuracy; no NaN/Inf/OOM/metadata/dataloader errors observed

## Conclusion

- **4 / 4 retry PASS:** Yes
