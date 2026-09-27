# ModelNet40 PointNet++ Line A Smoke Failure Diagnosis

- Generated at: 2026-07-07 20:12:00 UTC
- Smoke submit time: 2026-07-07 20:10:18 UTC

## Overview

| branch | job id | state | exit | failed stage |
| --- | ---: | --- | --- | --- |
| lineA_original_baseline_1024 | 1733475 | COMPLETED | 0:0 | — (PASS) |
| lineA_ear_4096 | 1733476 | FAILED | 1:0 | dataloader init |
| lineA_pdans_4096 | 1733477 | FAILED | 1:0 | dataloader init |
| lineA_punet_4096 | 1733478 | FAILED | 1:0 | dataloader init |
| lineA_pugcn_4096 | 1733479 | FAILED | 1:0 | dataloader init |

## Failed branch: lineA_ear_4096

- **Job ID:** 1733476
- **Job state:** FAILED
- **Exit code:** 1:0
- **Log path:** `logs/pointnet2_smoke/lineA_ear_4096/slurm_1733476.err`
- **Exact error:**
  ```
  FileNotFoundError: Missing label map: .../pointnet2_inputs/lineA_original_up/ear/metadata/class_to_idx.json
  ```
- **Likely cause:** `pointnet2_inputs/lineA_original_up/ear` symlinks to `datasets/lineA_original_up/strict_4N/ear/`, which contains only `train/` + `test/` — no `metadata/`. `ModelNetNPYDataset` requires `metadata/class_to_idx.json`.
- **Dataloader point count issue?** No — failure occurs before any point loading.
- **GPU OOM?** No — training never reached GPU forward pass.
- **Config path issue?** Yes — `data_root` points to strict upsampling dir which lacks metadata layout expected by dataloader.
- **Environment/module issue?** No — PyTorch 2.6.0 loaded, CUDA available; baseline smoke on same cluster PASS.

## Failed branch: lineA_pdans_4096

- **Job ID:** 1733477
- **Log:** `logs/pointnet2_smoke/lineA_pdans_4096/slurm_1733477.err`
- **Error:** `Missing label map: .../strict_4N/pdans/metadata/class_to_idx.json`
- Same root cause as EAR.

## Failed branch: lineA_punet_4096

- **Job ID:** 1733478
- **Log:** `logs/pointnet2_smoke/lineA_punet_4096/slurm_1733478.err`
- **Error:** `Missing label map: .../strict_4N/pu_net/metadata/class_to_idx.json`
- Same root cause as EAR.

## Failed branch: lineA_pugcn_4096

- **Job ID:** 1733479
- **Log:** `logs/pointnet2_smoke/lineA_pugcn_4096/slurm_1733479.err`
- **Error:** `Missing label map: .../strict_4N/pu_gcn/metadata/class_to_idx.json`
- Same root cause as EAR.

## Why baseline passed

`pointnet2_inputs/lineA_original_baseline` → `datasets/modelnet40_original` includes:

```
metadata/class_to_idx.json
metadata/idx_to_class.json
metadata/train_manifest.csv
metadata/test_manifest.csv
```

Upsampling strict dirs (`datasets/lineA_original_up/strict_4N/*`) have **no `metadata/`**.

## Parallel with Line B (already fixed)

Line B had the identical failure pattern in smoke round 1. Fix applied (see `reports/modelnet40_pointnet2_lineB_metadata_fix_audit.md`):

- **v2 (PASS):** per-method `metadata/` with only `class_to_idx.json` + `idx_to_class.json` symlinks from `modelnet40_downsampled_x4/metadata`
- No manifests → directory scan loads upsampled `.npy` files at correct point count

Line A needs the same pattern under `pointnet2_inputs/lineA_original_up/<method>/metadata/`, sourcing label maps from `datasets/modelnet40_original/metadata`.

## Recommended fix (do not apply in this round)

Choose one approach **without modifying upsampling `.npy` data or `datasets/` contents**:

1. **PointNet++ input wrapper (preferred, mirrors Line B v2):** Add `metadata/class_to_idx.json` + `metadata/idx_to_class.json` symlinks under each `pointnet2_inputs/lineA_original_up/<method>/`, sourced from `modelnet40_original/metadata`. Do **not** symlink manifests (would point to 1024-pt baseline paths).
2. **Dataloader fallback:** Extend `ModelNetNPYDataset` to accept `--metadata-root` or fall back when `data_root/metadata` is missing.

After fix:

1. Re-run only the 4 failed Line A smoke jobs.
2. Confirm all 5 PASS before submitting full training.

## Action for this round

- **Do not submit Line A full training.**
- **Do not treat baseline smoke metrics as final classification results.**
- **Do not modify datasets** (per protocol constraint).
