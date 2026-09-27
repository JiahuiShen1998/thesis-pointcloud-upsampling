# ModelNet40 PointNet++ Line B Smoke Final Status

- Generated at: 2026-07-07 12:41:00 UTC
- **Overall status: LINE_B_SMOKE_ALL_PASS**

## Final summary (5 / 5 branches)

| branch | method | final job id | expected pts | actual pts | state | exit | status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| lineB_downsampled_x4_baseline_256 | baseline | 1733159 | 256 | 256 | COMPLETED | 0:0 | PASS |
| lineB_ear_1024 | EAR | 1733171 | 1024 | 1024 | COMPLETED | 0:0 | PASS |
| lineB_pdans_1024 | PDANS | 1733172 | 1024 | 1024 | COMPLETED | 0:0 | PASS |
| lineB_punet_1024 | PU-Net | 1733173 | 1024 | 1024 | COMPLETED | 0:0 | PASS |
| lineB_pugcn_1024 | PU-GCN | 1733174 | 1024 | 1024 | COMPLETED | 0:0 | PASS |

## Metadata fix applied

1. **v1 (failed):** symlink entire `modelnet40_downsampled_x4/metadata` → manifests pointed to 256-pt baseline paths.
2. **v2 (PASS):** per-method `metadata/` with only `class_to_idx.json` + `idx_to_class.json` symlinks; no manifests → dataloader directory scan loads 1024-pt upsampled `.npy` under `strict_N/<method>/`.

No `.npy` point cloud files were modified.

## Next step

**Line B full training is ready to submit** — see `reports/modelnet40_pointnet2_lineB_full_training_ready_commands.md`

Do not mix smoke metrics with final classification results.
