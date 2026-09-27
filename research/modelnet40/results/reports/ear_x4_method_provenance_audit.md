# EAR ×4 Method Provenance Audit

**Project:** `modelnet40_pointnet2_upsampling`  
**Audit date:** 2026-06-30  
**Scope:** Confirm EAR ×4 outputs used the EAR upsampling algorithm (not generic resample / loader padding).

---

## Executive summary

| Line | Input → Output | Status | Main method comparison |
|------|----------------|--------|------------------------|
| A (Original) | 1024 → 4096 (×4) | **method_confirmed** | **Valid** |
| B (Downsampled50) | 512 → 2048 (×4) | **method_confirmed** | **Valid** |

EAR ×4 generation **does call the EAR algorithm** (`ear_upsample_kitti`) via a dedicated wrapper. Generation is **not** generic `np.repeat` / random duplicate / loader-only resample. A wrapper-level `resample_to_target` postprocess exists but was **inactive** for all 12,311 samples per line (EAR raw output already matched target count).

---

## A. Generation script provenance

### 1. Generation script

| Item | Path |
|------|------|
| Full generation entrypoint | `scripts/run_ear_x4_chunk.py` |
| Shared EAR wrapper | `scripts/ear_modelnet40_utils.py` |
| EAR algorithm module | `modelnet40_experiments/external_lab_migrated/EAR/ear_upsampling.py` |
| Submit helper | `scripts/submit_ear_x4_full.sh` |
| Smoke test (pre-full) | `scripts/run_ear_x4_smoke.py` |

### 2. Slurm sbatch files

| Line | Sbatch | Job array ID |
|------|--------|--------------|
| A | `jobs/run_ear_x4_lineA_cpu_array.sbatch` | **1716174** (`ear_x4_a`, array 0–15) |
| B | `jobs/run_ear_x4_lineB_cpu_array.sbatch` | **1716175** (`ear_x4_b`, array 0–15); chunks 4 & 6 re-run as **1717034** |

Submit record: `reports/ear_x4_generation/submit_2026-06-25T21:16:02Z.md`

### 3. Wrapper call chain

```
run_ear_x4_chunk.py
  → load_ear_module(DEFAULT_EAR_SCRIPT)
  → run_ear_on_xyz(ear_module, points, target_n, up_factor=UP_FACTOR_X4=4.0, ...)
      → ear_module.ear_upsample_kitti(points_xyzi, target_n=..., up_factor=4.0, ...)
  → resample_to_target(raw_xyz, target_n, norm_seed)   # postprocess hook
  → np.save(final_output_path, final_xyz)
```

Key wrapper parameters (`run_ear_on_xyz` / chunk CLI defaults):

- `up_factor=4.0`
- `k_neighbors=20`, `edge_sensitivity=4.0`, `up_threshold=0.93`, `sigma_p=0.0`, `max_iter=5`
- `DEFAULT_EAR_SCRIPT` → `external_lab_migrated/EAR/ear_upsampling.py`

### 4. Does it call EAR itself?

**Yes.** `ear_modelnet40_utils.run_ear_on_xyz` dynamically imports and calls `ear_upsample_kitti`.

### 5. Generic padding function as primary upsampling?

**No.** Primary upsampling is `ear_upsample_kitti` (PCA normals, edge/density scoring, iterative midpoint insertion, optional EAR-internal fill step).

`resample_to_target` in `ear_modelnet40_utils.py` is a **post-EAR** random choice resampler for fixed point count — it is **not** the primary upsampling path.

### 6. np.repeat / random duplicate only?

**No.** No `np.repeat` in the generation path. `resample_to_target` uses `rng.choice` only **after** EAR output; chunk audits show `ear_raw_points == final_points` for **100%** of successful samples (postprocess was no-op).

### 7. Checkpoint / method-specific parameters

EAR here is **algorithmic** (not a neural checkpoint). Method-specific params are logged per sample in Slurm `.out` files:

- `Input sparse points: 1024` / `512`
- `Target output points: 4096` / `2048`
- `Estimating normals by PCA...`
- `Estimating EAR density threshold...`
- `Filling remaining points: N` (EAR-internal edge-aware fill when threshold insertion insufficient)

No DL checkpoint path is used.

### 8. TARGET_POINTS semantics

`TARGET_POINTS_X4_ORIGINAL=4096` and `TARGET_POINTS_X4_DOWN=2048` are **desired output sizes** passed to `ear_upsample_kitti(target_n=...)`. They do **not** replace the EAR algorithm with a generic resampler.

---

## B. Log evidence

### Commands (from sbatch)

```bash
python scripts/run_ear_x4_chunk.py --line A --chunk-id "${CHUNK_ID}" --num-chunks 16 --skip-existing
python scripts/run_ear_x4_chunk.py --line B --chunk-id "${CHUNK_ID}" --num-chunks 16 --skip-existing
```

### Sample log excerpts

**Line A** (`logs/ear_x4_generation/ear_x4_a_1716174_0.out`):

```
=== EAR ×4 Line A chunk 0/16 started: ...
Input sparse points: 1024
Target output points: 4096
Estimating normals by PCA...
Estimating EAR density threshold...
...
Line A chunk 0: 770 rows, failed=0, audit=.../lineA_chunk_audits/chunk_000.csv
```

**Line B** (`logs/ear_x4_generation/ear_x4_b_1716175_0.out`): same pattern with 512→2048.

### Logging assessment

| Signal | Present? |
|--------|----------|
| Method name "EAR" | Yes (job names, script docstrings, log banners) |
| Input/output paths | Yes (chunk audit CSVs) |
| Ratio / target points | Yes |
| EAR algorithm steps | Yes (PCA, density threshold, fill remaining) |
| Checkpoint config | N/A (algorithmic EAR) |
| Generic/fallback adapter as primary | No |
| Wrapper `resample_to_target` | In code only; not exercised in full generation |

**provenance_logging_insufficient:** **No** — logs + chunk audits + source code are sufficient.

### Chunk audit statistics

| Line | Successful samples | `ear_raw_points != final_points` | Unique `ear_raw_points` |
|------|-------------------|----------------------------------|---------------------------|
| A | 12,311 | 0 | {4096} |
| B | 12,311 | 0 | {2048} |

---

## C. Status classification

| Line | Status | Rationale |
|------|--------|-----------|
| A | `method_confirmed` | `ear_upsample_kitti` called with `up_factor=4`; wrapper postprocess inactive |
| B | `method_confirmed` | Same as Line A |

**Not** `generic_resample_only_invalid` — EAR algorithm is the primary path.  
**Not** `provenance_unclear_needs_manual_check` — script + logs + audits align.

### Postprocess note

Wrapper `resample_to_target(points, target_n, seed)`:

- Downsample: `rng.choice(..., replace=False)` if `count > target_n`
- Upsample: `rng.choice(..., replace=True)` if `count < target_n`
- No-op if `count == target_n`

For full EAR ×4 generation, EAR always produced exactly `target_n` points, so this step copied through unchanged.

EAR **internal** "Filling remaining points" (edge-aware midpoint projection) is part of the EAR algorithm, not wrapper postprocess.

---

## D. Data-layer sanity check (supplementary)

Script: `scripts/audit_method_output_similarity.py`  
Reports: `reports/ear_x4_output_similarity_sanity.csv`, `reports/ear_x4_output_similarity_sanity.md`

200 samples (100 per line, seed=42):

| Line | Mean duplicate ratio (6 dp) | Mean input-exact-in-output | Mean NN dist (out→in) | >50% duplicate flag |
|------|----------------------------|-----------------------------|----------------------|---------------------|
| A | 0.531 | 0.250 (1024/4096) | 0.028 | 89/100 |
| B | 0.536 | 0.250 (512/2048) | 0.041 | 89/100 |

**Interpretation:**

- **25% output points exactly match input** — consistent with EAR retaining original sparse points and adding ~3× new projected points (×4 total).
- **Non-zero NN distances** for new points — not a pure repeat of input.
- High "duplicate ratio" at 6-decimal rounding likely reflects numerically close projected midpoints, not wrapper `np.repeat`.

This sanity check **cannot alone prove EAR** but does **not** indicate trivial input-only duplication as the sole upsampling mechanism.

---

## E. PointNet++ training — no loader resample

### Configs

| Experiment | `num_point` | `allow_resample` | `data_root` |
|------------|-------------|------------------|-------------|
| `configs/original_ear_x4_pointnet2.yaml` | 4096 | **false** | `datasets/modelnet40_original_up/ear_x4` |
| `configs/downsampled50_ear_x4_pointnet2.yaml` | 2048 | **false** | `datasets/modelnet40_downsampled50_up/ear_x4` |

### Training logs (confirmed)

**`logs/original_ear_x4_pointnet2/train.log`:**

```
num_point=4096 allow_resample=False
First train sample shape after loader: (4096, 3)
```

**`logs/downsampled50_ear_x4_pointnet2/train.log`:**

```
num_point=2048 allow_resample=False
First train sample shape after loader: (2048, 3)
```

Training uses native upsampled point counts without dataloader resampling.

---

## F. Protocol alignment

| Branch | Input | Upsampling | Output | Baseline comparison |
|--------|-------|------------|--------|---------------------|
| Original + EAR ×4 | 1024 | EAR ×4 | 4096 | Baseline stays 1024 |
| Downsampled50 + EAR ×4 | 512 | EAR ×4 | 2048 | Baseline stays 512 |

Upsampling outputs are **not** cropped back to baseline point counts.

---

## G. Regeneration plan

**Not required.** Provenance is confirmed. No `invalid_or_unclear_for_main_method_comparison` flag.

If regeneration were ever needed (future reference only — **do not execute without user confirmation**):

- Entrypoint: `scripts/run_ear_x4_chunk.py` with `--ear-script` pointing to `ear_upsampling.py`
- Resubmit via `scripts/submit_ear_x4_full.sh`
- Ensure chunk audits record `ear_raw_points` vs `final_points`
- Avoid using `resample_to_target` as primary upsampling

---

## Related artifacts

| Artifact | Path |
|----------|------|
| Provenance CSV | `reports/ear_x4_method_provenance_audit.csv` |
| Similarity sanity CSV | `reports/ear_x4_output_similarity_sanity.csv` |
| Similarity sanity MD | `reports/ear_x4_output_similarity_sanity.md` |
| Full generation audit | `reports/ear_x4_full_generation_audit.md` |
| Future method requirements | `reports/modelnet40_x4_method_provenance_requirements.md` |
