# ModelNet40 PU-GCN Smoke Audit

- project_root: `/home/ra87racy/projects/modelnet40_pointnet2_upsampling`
- Line A PASS: `8/8`
- Line B PASS: `8/8`
- Overall smoke: **PASS**

## Criteria

- Line A: input (1024,3), strict (4096,3), no NaN/Inf
- Line B: input (256,3), strict (1024,3), no NaN/Inf

## Notes

- Smoke uses real PU-GCN inference via `run_pugcn_modelnet40_x4.py`.
- Full 12311-sample jobs require HPC PROJECT_ROOT with complete protocol datasets.
