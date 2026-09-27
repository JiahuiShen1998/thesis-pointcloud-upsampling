# Step 7 — Upsampling Smoke Test Preparation

- Generated at: 2026-06-20
- Status: **PREP ONLY** (no GPU jobs submitted, no upsampling executed)

## Step 7 Goal

Validate the end-to-end path on a small subset before batch upsampling:

```text
Original / Downsampled50 .npy
  → Upsampling method (PDANS or PU-Net)
  → Normalize to (1024, 3) .npy
  → PointNet++ DataLoader dry-run
  → (later) small train/eval smoke test
```

Cover **Line A** (original) and **Line B** (downsampled50).

---

## Current Pipeline Status

| Step | Status |
| --- | --- |
| Step 4 Original baseline | **Running** — Job `1709666`, ~104+/200 epochs |
| Step 6 Downsampled50 baseline | **Prepared, NOT submitted** |
| Step 7 upsampling smoke | **This document — prep only** |

---

## Upsampling Method Audit

| Method | Code Path | Checkpoint | Input Format | Output Format | CUDA Ext | CPU Smoke | ModelNet40 Ready | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **PDANS** | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS` | ✅ `checkpoints/PU1K_PDANS.pkl`, `PUGAN_PDANS.pkl` (~5.2 GB) | `.xyz` via Open3D (`example_eval.py`) | `.xyz` (R×N points) | ✅ `pointops_cuda.so`, Chamfer3D, PyTorch3D | ❌ GPU required | ⚠️ Needs wrapper: `.npy→.xyz→infer→1024` | PyTorch3D env, 4× output needs resample to 1024 |
| **PU-Net** | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-Net` | ✅ `model/generator2_new6/model-120.*` (TF1) | `.xyz` (`main.py --data_folder`) | `.xyz` in `result/` | ✅ TF custom ops (`code/tf_ops/`) | ❌ GPU typical | ⚠️ Needs TF1 env + `.npy→.xyz` | **TensorFlow 1.x** on Python 3.12 HPC is high risk |
| **PU-GCN** | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN` | ✅ `pretrained/pu1k-mpu/model-*.index/meta` | HDF5 / patches | point cloud | ✅ TF ops | ❌ | ❌ Later | Same TF1 issues as PU-Net |
| **EAR** | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/code/upsampling_methods/EAR` | N/A (deterministic) | KITTI `.bin` workflow | `.bin` | ❌ | Partial | ❌ KITTI-specific | Not ModelNet40-native; batch later |

Symlink/alternate paths under `/home/woody/iwnt/iwnt189h/thesis_pointcloud/code/upsampling_methods/` mirror partial copies; **use `external_lab_migrated/` as primary for PDANS/PU-Net/PU-GCN**.

---

## Recommended First Smoke Test Method

### Primary: **PDANS**

Reasons:
1. User priority and thesis method consistency with KITTI line
2. **PyTorch-based** — closer to current HPC PyTorch 2.6 stack than TF1 PU-Net
3. Checkpoints already on HPC (`PU1K_PDANS.pkl`, `PUGAN_PDANS.pkl`)
4. Precompiled `pointops_cuda.so` present
5. Single-file inference path exists: `pointnet2/example_eval.py`

Fallback if PDANS wrapper fails on ModelNet40 format: **PU-Net** — only if a dedicated TF1 conda env can be restored (currently **not** available on default modules).

---

## Smoke Sample Strategy

- **1 sample per class per split** (deterministic: first sorted `.npy` in each class folder)
- **Train:** 40 samples (40 classes)
- **Test:** 40 samples (40 classes)
- **Total:** 80 samples

Manifest:

`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/step7_smoke_sample_manifest.csv`

---

## Smoke Input Directories (symlinks)

| Line | Path | Raw Shape |
| --- | --- | --- |
| Line A | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/smoke_inputs/original` | `(1024, 3)` |
| Line B | `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/smoke_inputs/downsampled50` | `(512, 3)` |

Each contains `train/{class}/*.npy` and `test/{class}/*.npy` — **80 symlinks each**.

Regenerate:

```bash
module load python/pytorch2.6py3.12
python /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/scripts/create_step7_smoke_manifest.py
```

---

## Expected Smoke Output Paths (not created yet)

| Line | Method | Output Dir |
| --- | --- | --- |
| Line A | PDANS | `datasets/modelnet40_original_up/pdans_smoke/` |
| Line B | PDANS | `datasets/modelnet40_downsampled50_up/pdans_smoke/` |
| Line A | PU-Net (fallback) | `datasets/modelnet40_original_up/punet_smoke/` |
| Line B | PU-Net (fallback) | `datasets/modelnet40_downsampled50_up/punet_smoke/` |

Post-upsampling normalization target: **all outputs `(1024, 3)`** before PointNet++.

---

## PDANS Smoke Test Plan (next execution step)

1. Convert 80 smoke `.npy` → `.xyz` (temporary dir under `datasets/smoke_inputs/xyz/`)
2. Run PDANS inference with `R=2` or `R=4` depending on input size:
   - Line A: 1024 → 2048 or 4096 → **resample/FPS to 1024**
   - Line B: 512 → 1024 (R=2 ideal) or 2048 → **resample to 1024**
3. Convert output `.xyz` → `.npy` `(1024, 3)`
4. Run PointNet++ DataLoader dry-run on smoke outputs
5. Verify labels unchanged vs manifest

Wrapper scripts to create in Step 7 execution (not yet):

- `scripts/convert_npy_to_xyz.py`
- `scripts/run_pdans_smoke.sh`
- `scripts/normalize_upsampled_to_1024.py`
- `scripts/dryrun_upsampled_pointnet2.py`

---

## What We Must NOT Do Yet

- ❌ Submit Step 6 Downsampled50 baseline training (Step 4 GPU still busy)
- ❌ Submit any new GPU job for upsampling smoke (prep only this turn)
- ❌ Run full-dataset upsampling (12,311 × 2 lines × methods)
- ❌ PointNet++ full training on upsampled data

---

## GPU Job Confirmation

**No GPU jobs were submitted in this Step 7 preparation phase.**

---

## Next Prompt Should Do

After Step 4 completes and Step 6 is submitted (or queued):

**Execute PDANS smoke test on 80 samples:**

1. Build `.npy → .xyz` converter for smoke inputs
2. Create PDANS smoke wrapper using `external_lab_migrated/PDANS/pointnet2/example_eval.py` or `samples.py`
3. Normalize outputs to `(1024, 3)` and write to `*_up/pdans_smoke/`
4. DataLoader dry-run on both Line A and Line B smoke outputs
5. If PDANS fails on env/format, document failure and switch to PU-Net fallback plan

Suggested prompt: **「执行 Step 7 PDANS smoke test」**
