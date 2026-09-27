# ModelNet40 PointNet++ Line B Smoke Retry Audit

- Generated at: 2026-07-07 12:41:00 UTC
- Retry round 2 submit: 2026-07-07 12:39:27 UTC
- Metadata fix v2: label maps only (no manifests) — dataloader scans `strict_N/<method>/train|test`
- Overall retry round 2: **4 / 4 PASS**

## Retry round 1 (failed — for record)

Full metadata dir symlink caused manifests to load 256-pt baseline `.npy` paths. Jobs 1733167–1733170 FAILED with `expected (1024, 3) got (256, 3)`.

## Retry round 2 results

| branch | old job | retry job | expected | actual | batch shape | first loss | state | status |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| lineB_ear_1024 | 1733160 | 1733171 | 1024 | 1024 | (8,3,1024) | 3.734455 | COMPLETED | **PASS** |
| lineB_pdans_1024 | 1733161 | 1733172 | 1024 | 1024 | (8,3,1024) | 3.692327 | COMPLETED | **PASS** |
| lineB_punet_1024 | 1733162 | 1733173 | 1024 | 1024 | (8,3,1024) | 3.828374 | COMPLETED | **PASS** |
| lineB_pugcn_1024 | 1733163 | 1733174 | 1024 | 1024 | (8,3,1024) | 3.644277 | COMPLETED | **PASS** |

All four: forward/backward/eval PASS, `metrics.json` with `final_test_overall_accuracy`, no NaN/Inf/OOM/metadata errors.

Previous round audit preserved: `reports/modelnet40_pointnet2_lineB_smoke_audit.md`
