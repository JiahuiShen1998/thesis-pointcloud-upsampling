# Step 8 Protocol Audit — Downsampled50 Baseline vs EAR (Line B)

- Generated at: 2026-06-25
- Scope: Line B — `downsampled50_ear_pointnet2` (Job 1711437) vs main baseline
- Verdict: **UPSAMPLING BRANCH VALID**; **MAIN BASELINE PENDING** (`downsampled50_native512_pointnet2`)

---

## Executive Summary

| Statement | Status |
| --- | --- |
| **Original line and Downsampled50 line are independent.** | Confirmed |
| **`downsampled50_baseline` loader 512→1024 is ablation only.** | Confirmed |
| **Main Downsampled50 baseline = native 512 (`num_point=512`).** | Pending training (`p2_ds512`) |
| **`downsampled50_ear` at 1024 is a valid upsampling branch.** | Confirmed |
| **Future PU-Net / PU-GCN / PDANS must match EAR output (1024).** | Required |

Step 8 (`downsampled50_ear_pointnet2`) belongs to **Line B upsampling branch**. It must be compared against the **main Line B baseline** (`downsampled50_native512_pointnet2`, 512 pts), not against the Step 6 naive-resampled ablation or `original_baseline`.

---

## 1. Line B Branch Assignment

| Experiment | Branch | Disk | `num_point` | Job | Role |
| --- | --- | ---: | ---: | --- | --- |
| `downsampled50_native512_pointnet2` | **baseline (main)** | 512 | **512** | `p2_ds512` (pending) | Canonical Line B baseline |
| `downsampled50_baseline` | ablation | 512 | 1024 (loader) | 1710180 | Naive-resampled ablation only |
| `downsampled50_ear_pointnet2` | **upsampling** | 1024 | **1024** | 1711437 | Valid upsampling branch (Step 8) |

**Step 8 belongs to Line B upsampling branch.**

---

## 2. Point-Count Audit

### 2.1 Main baseline vs upsampling — different counts by design

| Variant | Pipeline | PointNet++ input |
| --- | --- | ---: |
| Main baseline | 512 native → no resample | **512** |
| Step 8 EAR | 512 → EAR ×2 → 1024 | **1024** |
| Step 6 ablation | 512 → loader resample → 1024 | 1024 |

- **No baseline-to-upsampling point matching** — main baseline stays at 512.
- **No upsampling-to-baseline clipping** — EAR output stays at 1024.
- Baseline (512) and upsampling (1024) **intentionally differ**; this is protocol-compliant.

### 2.2 Step 6 ablation — NOT main baseline

`downsampled50_baseline` (Job 1710180, 91.17%):

| Setting | Value | Protocol |
| --- | --- | --- |
| Disk | 512 | — |
| Loader | `allow_resample=true`, 512→1024 | Violates main baseline rule |
| `num_point` | 1024 | Matches upsampling branch count artificially |
| Classification | 91.17% overall | Valid as **ablation diagnostic only** |

**`downsampled50_baseline` loader 512→1024 is ablation only.** Do not use it as the primary comparison target for Step 8 or future upsampling methods.

### 2.3 Step 8 EAR — valid upsampling branch

`downsampled50_ear_pointnet2` (Job 1711437, 90.82%):

| Setting | Value | Protocol |
| --- | --- | --- |
| Upsampling | 512 → EAR → 1024 (×2) | Valid |
| Disk / output | 1024 | **Upsampled count is preserved** |
| `num_point` | 1024 | Valid |
| `allow_resample` | false | No silent crop/pad |

**`downsampled50_ear` at 1024 is a valid upsampling branch.**

---

## 3. Upsampling Branch Consistency (Line B)

All Line B upsampling methods must share:

| Requirement | Line B target |
| --- | --- |
| Input | 512 (downsampled native) |
| Upsampling factor | **×2** |
| Output / `num_point` | **1024** |

| Method | Status | Output pts |
| --- | --- | ---: |
| EAR (Step 7b + Step 8) | Completed | 1024 |
| PU-Net | Future | **must be 1024** |
| PU-GCN | Future | **must be 1024** |
| PDANS | Future | **must be 1024** |

**Subsequent Downsampled50 + PU-Net / PU-GCN / PDANS must keep the same upsampled output point count as EAR (1024).**

---

## 4. Authorized Comparisons (Line B)

### Primary (after `p2_ds512` completes)

```
downsampled50_native512_pointnet2  (512 → PointNet++)
        vs
downsampled50_ear_pointnet2        (512 → EAR → 1024 → PointNet++)
```

| Metric | Main baseline | Step 8 EAR | Notes |
| --- | --- | --- | --- |
| `num_point` | 512 | 1024 | Different by design |
| Overall acc | pending | 90.82% | Compare after baseline trains |

### Ablation reference (optional footnote)

```
downsampled50_baseline  (512 → loader → 1024)  vs  downsampled50_ear  (1024)
```

Result: 91.17% vs 90.82%. Both at 1024 input — useful only to isolate loader-resample effect; **not main protocol**.

### Not authorized

| Comparison | Reason |
| --- | --- |
| Step 8 vs `original_baseline` | Different line (B vs A) |
| Step 8 vs Step 6 ablation as **main** result | Step 6 is ablation, not canonical baseline |

---

## 5. Job Disposition

| Job | Experiment | Action |
| --- | --- | --- |
| **1711437** | `downsampled50_ear_pointnet2` | **KEEP** — valid upsampling branch result |
| **1710180** | `downsampled50_baseline` | **KEEP** — relabel as ablation only |
| **`p2_ds512`** | `downsampled50_native512_pointnet2` | **SUBMITTED** — Job **1716131** |

---

## 6. Configuration Evidence

### Main baseline (`downsampled50_native512_pointnet2`)

```yaml
# configs/downsampled50_native512_pointnet2.yaml
native_points: 512
num_point: 512
allow_resample: false
```

### Step 8 EAR

```yaml
# configs/downsampled50_ear_pointnet2.yaml
native_points: 1024
num_point: 1024
allow_resample: false
```

### Step 6 ablation

```bash
# scripts/train_pointnet2.sh — downsampled50_baseline
NUM_POINT=1024
ALLOW_RESAMPLE="--allow-resample"
```

---

## 7. Sign-off

| Check | Result |
| --- | --- |
| Step 8 = Line B upsampling branch | ✅ |
| EAR 1024 preserved, not clipped | ✅ |
| Step 6 512→1024 = ablation only | ✅ |
| Main baseline = native 512 | ⏳ Job **1716131** (`p2_ds512`) submitted |
| Future upsampling methods → 1024 | Required |
| Cross-line comparison avoided | ✅ |

**Audit: Step 8 upsampling branch is protocol-compliant.** Primary Line B comparison awaits `downsampled50_native512_pointnet2` training.
