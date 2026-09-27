# ModelNet40 PointNet++ Line B Smoke Failure Diagnosis

- Generated at: 2026-07-07 12:27:00 UTC
- Smoke submit time: 2026-07-07 12:25:06 UTC

## Overview

| branch | job id | state | exit | failed stage |
| --- | ---: | --- | --- | --- |
| lineB_downsampled_x4_baseline_256 | 1733159 | COMPLETED | 0:0 | — (PASS) |
| lineB_ear_1024 | 1733160 | FAILED | 1:0 | dataloader init |
| lineB_pdans_1024 | 1733161 | FAILED | 1:0 | dataloader init |
| lineB_punet_1024 | 1733162 | FAILED | 1:0 | dataloader init |
| lineB_pugcn_1024 | 1733163 | FAILED | 1:0 | dataloader init |

## Failed branch: lineB_ear_1024

- **Job ID:** 1733160
- **Job state:** FAILED
- **Exit code:** 1:0
- **Log path:** `logs/pointnet2_smoke/lineB_ear_1024/slurm_1733160.err`
- **Exact error:**
  ```
  FileNotFoundError: Missing label map: .../pointnet2_inputs/lineB_downsampled_x4_up/ear/metadata/class_to_idx.json
  ```
- **Likely cause:** Upsampling strict output directories were generated with `train/` + `test/` only; `ModelNetNPYDataset` requires `metadata/class_to_idx.json` and split manifests under `data_root`.
- **Dataloader point count issue?** No — failure occurs before any point loading.
- **GPU OOM?** No — training never reached GPU forward pass.
- **Config path issue?** Partially — `data_root` points to strict upsampling dir which lacks metadata layout expected by dataloader.
- **Environment/module issue?** No — PyTorch 2.6.0 loaded, CUDA available; baseline smoke on same cluster PASS.

## Failed branch: lineB_pdans_1024

- **Job ID:** 1733161
- **Log:** `logs/pointnet2_smoke/lineB_pdans_1024/slurm_1733161.err`
- **Error:** `Missing label map: .../strict_N/pdans/metadata/class_to_idx.json`
- Same root cause as EAR.

## Failed branch: lineB_punet_1024

- **Job ID:** 1733162
- **Log:** `logs/pointnet2_smoke/lineB_punet_1024/slurm_1733162.err`
- **Error:** `Missing label map: .../strict_N/pu_net/metadata/class_to_idx.json`
- Same root cause as EAR.

## Failed branch: lineB_pugcn_1024

- **Job ID:** 1733163
- **Log:** `logs/pointnet2_smoke/lineB_pugcn_1024/slurm_1733163.err`
- **Error:** `Missing label map: .../strict_N/pu_gcn/metadata/class_to_idx.json`
- Same root cause as EAR.

## Why baseline passed

`pointnet2_inputs/lineB_downsampled_x4_baseline` → `datasets/modelnet40_downsampled_x4` includes:

```
metadata/class_to_idx.json
metadata/idx_to_class.json
metadata/train_manifest.csv
metadata/test_manifest.csv
```

Upsampling strict dirs (`datasets/lineB_downsampled_x4_up/strict_N/*`) have **no `metadata/`**.

## Recommended fix (do not apply in this round)

Choose one approach without modifying upsampling `.npy` data:

1. **PointNet++ input wrapper (preferred):** Replace direct symlinks with thin wrapper dirs under `pointnet2_inputs/lineB_downsampled_x4_up/<method>/` that symlink `train/`, `test/`, and `metadata/` (from `modelnet40_downsampled_x4` or `modelnet40_original`).
2. **Dataloader fallback:** Extend `ModelNetNPYDataset` to accept `--metadata-root` or fall back to `ORIGINAL_ROOT/metadata` when `data_root/metadata` is missing.
3. **Smoke/full sbatch:** Pass `--data-root` to a pre-wrapped input path that already includes metadata.

After fix:

1. Re-run only the 4 failed Line B smoke jobs.
2. Confirm all 5 PASS before submitting full training.

## Action for this round

- **Do not submit Line B full training.**
- **Do not treat baseline smoke metrics as final classification results.**
