# ModelNet40 ×4 Upsampling — Method Provenance Requirements

**Project:** `modelnet40_pointnet2_upsampling`  
**Protocol:** Main comparison uses **R=×4** upsampling on two input lines.

| Line | Input source | Input points | Target output |
|------|--------------|--------------|---------------|
| A (Original) | `datasets/modelnet40_original` | 1024 | 4096 |
| B (Downsampled50) | `datasets/modelnet40_downsampled50` | 512 | 2048 |

**Baseline branches** keep native point counts (1024 / 512). Upsampling outputs must **not** be cropped to baseline counts. PointNet++ training on upsampled data must use `allow_resample=false`.

---

## Methods in scope

| Method | Status | Output subdir |
|--------|--------|---------------|
| EAR | Audited — `method_confirmed` | `ear_x4` |
| PDANS | Pending full generation | `pdans_x4` |
| PU-Net | Pending full generation | `punet_x4` |
| PU-GCN | Pending full generation | `pugcn_x4` |

Reference implementation patterns:

- EAR: `scripts/ear_modelnet40_utils.py`, `scripts/run_ear_x4_chunk.py`
- PDANS / PU-Net / PU-GCN: `scripts/upsampling_x4_factory.py`, `scripts/upsampling_x4_common.py`, `scripts/*_modelnet40_utils.py`

---

## 1. Wrapper must call the real method

Before **full generation**, each method's wrapper must invoke that method's inference code:

| Method | Required call |
|--------|---------------|
| PDANS | `PDANSUpsampler.upsample_xyz()` via `load_pdans_upsampler()` — must load PDANS model weights and run model inference |
| PU-Net | `PUNetUpsampler.upsample_xyz()` via `load_punet_upsampler()` — must load PU-Net checkpoint and run model inference |
| PU-GCN | `PUGCNUpsampler.upsample_xyz()` via `load_pugcn_upsampler()` — must load PU-GCN checkpoint and run model inference |
| EAR | `ear_upsample_kitti()` via `run_ear_on_xyz()` — algorithmic EAR (reference audit complete) |

**Forbidden:** A wrapper that only calls `resample_to_target`, `np.repeat`, FPS padding, or dataloader resample and labels the output as method upsampling.

---

## 2. No generic resample disguised as method output

The following must **not** be the primary upsampling mechanism for any main-protocol method:

- `np.repeat` / tiling input points
- Random duplicate (`rng.choice(..., replace=True)` on input only)
- Interpolation-only upsampling without method inference
- FPS / padding-only to reach `target_points`
- PointNet++ `ModelNetNpyDataset` resample (`allow_resample=true`) presented as upsampling

`TARGET_POINTS` / `target_n` parameters define **output cardinality** passed to the method; they must not substitute for the method itself.

---

## 3. Required artifacts per method (before full generation approval)

Each method must produce the following **before** submitting full Slurm generation:

| Artifact | Purpose |
|----------|---------|
| `reports/{method}_x4_method_provenance_audit.md` | Human-readable provenance verdict |
| `reports/{method}_x4_method_provenance_audit.csv` | Machine-readable per-line status |
| `reports/{method}_x4_generation_manifest.csv` | Per-sample generation record (input path, output path, status, raw vs final points) |
| `reports/{method}_x4_output_similarity_sanity.csv` | Data-layer duplicate / NN-distance checks |
| `reports/{method}_x4_output_similarity_sanity.md` | Sanity summary |
| `reports/{method}_x4_full_generation_audit.md` | Post-completion completeness audit |

Smoke phase should use `scripts/audit_method_output_similarity.py` (or method-specific variant) with ≥100 samples per line.

### Provenance CSV fields (minimum)

```
line, method, input_path, output_path, generation_script, sbatch_file, log_file,
target_points, ratio, method_specific_call_detected, checkpoint_or_algorithm_detected,
generic_resample_detected, random_duplicate_detected, postprocess_detected, status, notes
```

### Status values

| Status | Meaning |
|--------|---------|
| `method_confirmed` | Method inference called; no generic-only resample as primary |
| `method_with_postprocess` | Method called; documented postprocess for fixed point count |
| `generic_resample_only_invalid` | **Invalid for main comparison** — regenerate required |
| `provenance_unclear_needs_manual_check` | Insufficient logs/code — manual review before full gen |

---

## 4. Postprocess rules (if fixed point count needed)

Some methods may output variable point counts. Postprocessing is allowed **only if**:

1. It runs **after** method inference (model forward pass / EAR algorithm), never instead of it.
2. It is **documented** in provenance audit (`postprocess_detected=yes` + description).
3. Chunk/manifest logs record **`raw_points` vs `final_points`** per sample.
4. Impact on point count is explicit (e.g. random downsample to 4096 when model outputs 4200).

Acceptable postprocess examples:

- `resample_to_target` after EAR when `ear_raw_points != target_n` (EAR reference: inactive for full gen)
- Truncation when model outputs slightly more than target

Unacceptable:

- Using postprocess as the only way to reach ×4 point count without method generating new geometry

---

## 5. PointNet++ training requirements (all upsampling experiments)

Config YAML must set:

```yaml
num_point: <native upsampled count>   # 4096 (Line A) or 2048 (Line B)
allow_resample: false
```

Training log must show:

```
num_point=<N> allow_resample=False
First train sample shape after loader: (<N>, 3)
```

---

## 6. Slurm / logging requirements

Full generation jobs must log at minimum:

- Exact `python` command and script path
- Method name and checkpoint path (for DL methods)
- `input_path` / `output_path` or chunk audit CSV
- `target_points` and `up_factor` / ratio
- Absence of generic-only fallback (or explicit flag if triggered)

Job submit record: `reports/{method}_x4_generation/submit_<timestamp>.md`

---

## 7. Reference: EAR ×4 audit outcome

EAR ×4 is the template for subsequent methods:

- **Verdict:** `method_confirmed`
- **Reports:** `reports/ear_x4_method_provenance_audit.md`, `.csv`
- **Sanity:** `reports/ear_x4_output_similarity_sanity.md`, `.csv`
- **Script:** `scripts/audit_method_output_similarity.py`

PDANS, PU-Net, and PU-GCN must meet the same bar before their full-generation arrays are treated as valid main-protocol results.
