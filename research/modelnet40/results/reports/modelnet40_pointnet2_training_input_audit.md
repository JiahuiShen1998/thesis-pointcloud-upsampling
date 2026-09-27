# ModelNet40 PointNet++ Training Input Audit

- Generated at: 2026-07-07 07:29:07 UTC
- Expected samples per branch: train=9843, test=2468, total=12311

| line | branch | method | expected pts | min | max | exact | train | test | total | NaN | Inf | status | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| B | lineB_downsampled_x4_baseline | baseline | 256 | 256 | 256 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| B | lineB_downsampled_x4_up_ear | EAR | 1024 | 1024 | 1024 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| B | lineB_downsampled_x4_up_pdans | PDANS | 1024 | 1024 | 1024 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| B | lineB_downsampled_x4_up_pu_net | PU-Net | 1024 | 1024 | 1024 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| B | lineB_downsampled_x4_up_pu_gcn | PU-GCN | 1024 | 1024 | 1024 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| A | lineA_original_baseline | baseline | 1024 | 1024 | 1024 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| A | lineA_original_up_ear | EAR | 4096 | 4096 | 4096 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| A | lineA_original_up_pdans | PDANS | 4096 | 4096 | 4096 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| A | lineA_original_up_pu_net | PU-Net | 4096 | 4096 | 4096 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
| A | lineA_original_up_pu_gcn | PU-GCN | 4096 | 4096 | 4096 | 12311 | 9843 | 2468 | 12311 | 0 | 0 | PASS | all checks passed |
