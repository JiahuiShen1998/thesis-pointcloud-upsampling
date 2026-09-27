# ModelNet40 + PointNet++ — Final Experiment Protocol

- Project root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`
- Last updated: 2026-06-25
- Status: **CONFIRMED (final)**

---

## Core Principles

- **Original line and Downsampled50 line are independent.**
- **Baseline native count is preserved** — no padding/resample to match upsampling output.
- **Upsampled count is preserved** — no clipping back to baseline native count.
- **No baseline-to-upsampling point matching.**
- **No upsampling-to-baseline clipping.**
- **Same upsampling factor across methods** within a line's upsampling branch.
- **Same output point count across upsampling methods** within the same line.

---

## 1. Two Independent Lines, Two Branches Each

| Line | Baseline branch | Upsampling branch |
| --- | --- | --- |
| **Line A — Original** | `original_baseline` | `original_*_up` (EAR, PU-Net, PU-GCN, PDANS) |
| **Line B — Downsampled50** | `downsampled50_native512_pointnet2` | `downsampled50_*_up` (EAR Step 8, PU-Net, PU-GCN, PDANS) |

Each line has its own baseline and its own upsampling experiments. **Lines do not share point-count requirements.**

---

## 2. Point-Count Rules

### 2.1 Baseline branch

| Rule | Detail |
| --- | --- |
| **Baseline native count is preserved** | Feed PointNet++ the native/no-upsampling point count |
| **No baseline-to-upsampling point matching** | Do not pad or resample baseline to upsampled output count |
| Line A | Original native **1024** → PointNet++ `num_point=1024` |
| Line B | Downsampled native **512** → PointNet++ `num_point=512`, `allow_resample=false` |

### 2.2 Upsampling branch

| Rule | Detail |
| --- | --- |
| **Upsampled count is preserved** | Keep on-disk upsampling output as PointNet++ input |
| **No upsampling-to-baseline clipping** | Do not crop upsampled clouds back to baseline native count |
| **Same upsampling factor across methods** | e.g. Line B: 512 → ×2 → 1024 for all methods |
| **Same output point count across upsampling methods within the same line** | Line B upsampling branch: all methods → **1024** |

### 2.3 Cross-branch / cross-line

The four detector input categories **may all use different point counts**:

| Category | Example `num_point` |
| --- | ---: |
| Original baseline | 1024 |
| Original + upsampling | 2048 (if ×2 from 1024) |
| Downsampled50 baseline (main) | 512 |
| Downsampled50 + upsampling | 1024 |

**Within upsampling branch only** must point counts match across methods on the same line.

---

## 3. Line A — Original

```
Original baseline:
  original native points (1024) → PointNet++ (num_point=1024)

Original + EAR / PU-Net / PU-GCN / PDANS:
  original native points (1024)
    → same upsampling factor (e.g. ×2)
    → same upsampled point count (e.g. 2048)
    → PointNet++ (num_point=2048)
```

| Variant | Disk | `num_point` | `allow_resample` | Status |
| --- | ---: | ---: | --- | --- |
| `original_baseline` | 1024 | 1024 | true (pass-through) | Completed — 91.95% |
| `original_*_up` | TBD | **same across methods** | false | Future |

Original baseline stays at 1024 even if upsampling branch targets 2048.

---

## 4. Line B — Downsampled50

```
Downsampled50 baseline (main):
  downsampled native points (512) → PointNet++ (num_point=512, no resample)

Downsampled50 + EAR / PU-Net / PU-GCN / PDANS:
  downsampled native points (512)
    → same upsampling factor (×2)
    → same upsampled point count (1024)
    → PointNet++ (num_point=1024)
```

### 4.1 Main baseline — `downsampled50_native512_pointnet2`

| Field | Value |
| --- | --- |
| Dataset | `datasets/modelnet40_downsampled50` |
| Disk | **512** |
| `num_point` | **512** |
| `allow_resample` | **false** |
| Job name | `p2_ds512` |
| Role | **Line B main baseline** |

### 4.2 Upsampling branch — `downsampled50_ear_pointnet2` (Step 8)

| Field | Value |
| --- | --- |
| Path | 512 → EAR → **1024** |
| `num_point` | **1024** |
| `allow_resample` | false |
| Job | 1711437 (completed — 90.82%) |
| Role | Valid Line B upsampling branch |

Future `downsampled50_pdans`, `downsampled50_punet`, `downsampled50_pugcn` must also output **1024** (factor ×2).

### 4.3 Ablation only — `downsampled50_baseline` (Step 6)

| Field | Value |
| --- | --- |
| Disk | 512 |
| Loader | resample 512→1024 |
| `num_point` | 1024 |
| Result | 91.17% (Job 1710180) |
| Role | **Naive-resampled ablation only — NOT main baseline** |

If baseline is resampled/padded to 1024 while upsampling also uses 1024, that artificially matches point counts across branches and violates this protocol.

---

## 5. Comparison Scope

| Authorized | Not authorized |
| --- | --- |
| Line A baseline vs Line A upsampling methods | Line A vs Line B cross-line primary comparison |
| Line B main baseline (512) vs Line B upsampling (1024) | Step 8 EAR vs `original_baseline` |
| Within Line B upsampling: EAR vs PU-Net vs PDANS (all 1024) | Step 8 EAR vs `downsampled50_baseline` ablation (512→1024) as main comparison |

Baseline and upsampling branches on the same line **intentionally differ in point count**. Compare downstream task performance within the line, reporting both native and input counts explicitly.

---

## 6. Experiment Index

| Step | Line | Branch | Experiment | `num_point` | Role |
| ---: | --- | --- | --- | ---: | --- |
| 4 | A | baseline | `original_baseline` | 1024 | Main Line A baseline |
| 6 | B | ablation | `downsampled50_baseline` | 1024 (resampled) | Ablation only |
| 7b | B | upsampling prep | EAR full Line B | 1024 on disk | Dataset generation |
| 8 | B | upsampling | `downsampled50_ear_pointnet2` | 1024 | Line B EAR result |
| new | B | baseline | `downsampled50_native512_pointnet2` | 512 | **Main Line B baseline** (Job 1716131) |

---

## 7. Related Documents

| Document | Path |
| --- | --- |
| Step 8 audit | `reports/step8_protocol_audit_downsampled50_baseline_vs_ear.md` |
| Correction log | `reports/modelnet40_pointnet2_final_protocol_corrected.md` |
| Step 8 audit correction | `reports/step8_protocol_audit_corrected.md` |
