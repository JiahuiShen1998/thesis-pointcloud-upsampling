# Step 8 Protocol Audit — Correction Log

- Generated at: 2026-06-25
- Canonical audit: `reports/step8_protocol_audit_downsampled50_baseline_vs_ear.md`

---

## Problem

Prior Step 8 audits oscillated between three incompatible framings:

| Draft | Claimed Step 6 role | Claimed Step 8 comparison | Within-line `num_point` rule |
| --- | --- | --- | --- |
| Draft v1 | Native 512 baseline | vs original_baseline | Unclear |
| Draft v2 | Loader 1024 = main baseline | vs Step 6 at 1024 | Baseline = upsampling = 1024 |
| Draft v3 (morning) | Native 512 = ablation | vs Step 6 at 1024 | Baseline = upsampling = 1024 |

Draft v2/v3 incorrectly treated **point-count equality between baseline and upsampling** as a protocol requirement, which contradicts the user's final rules.

---

## Final Resolution

### Baseline branch (Line B main)

| Field | Value |
| --- | --- |
| Experiment | `downsampled50_native512_pointnet2` |
| Disk | 512 |
| `num_point` | 512 |
| `allow_resample` | false |
| Job | `p2_ds512` |

### Upsampling branch (Step 8)

| Field | Value |
| --- | --- |
| Experiment | `downsampled50_ear_pointnet2` |
| Pipeline | 512 → EAR ×2 → 1024 |
| `num_point` | 1024 |
| Job | 1711437 (completed, 90.82%) |

### Ablation (Step 6 — relabeled)

| Field | Value |
| --- | --- |
| Experiment | `downsampled50_baseline` |
| Pipeline | 512 → loader resample → 1024 |
| `num_point` | 1024 |
| Job | 1710180 (completed, 91.17%) |
| Role | **Naive-resampled ablation only** |

---

## Key Corrections

1. **`downsampled50_baseline` loader 512→1024 is ablation only** — it pads/resamples baseline to upsampling count, which the final protocol forbids for the main baseline.

2. **Main Downsampled50 baseline should be native 512** — `downsampled50_native512_pointnet2` with `allow_resample=false`.

3. **`downsampled50_ear` at 1024 is a valid upsampling branch** — EAR output is not clipped; Step 8 Job 1711437 remains valid.

4. **Subsequent Downsampled50 + PU-Net / PU-GCN / PDANS must output 1024** — same upsampling factor (×2) and same output count as EAR within Line B upsampling branch.

5. **Step 8 primary comparison** is against main baseline (512), not Step 6 ablation (1024) and not `original_baseline` (Line A).

---

## Comparison Matrix (final)

| vs → | `native512` (512) | `downsampled50_baseline` ablation (1024) | `downsampled50_ear` (1024) | `original_baseline` (1024) |
| --- | --- | --- | --- | --- |
| **`downsampled50_ear`** | **Primary (Line B)** | Ablation footnote | Upsampling method comparison (future) | Invalid (cross-line) |

---

## Step 8 Result Interpretation (interim)

Until `p2_ds512` completes:

| Experiment | Overall acc | `num_point` | Usable as |
| --- | ---: | ---: | --- |
| `downsampled50_ear_pointnet2` | 90.82% | 1024 | Line B upsampling branch result |
| `downsampled50_baseline` (ablation) | 91.17% | 1024 | Ablation only — EAR −0.35 pp vs ablation |
| `downsampled50_native512_pointnet2` | pending | 512 | **Job 1716131 submitted — await for main comparison** |

Do **not** report "EAR hurts classification on Line B" as a final thesis conclusion until the native-512 main baseline result is available.
