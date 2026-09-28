# ModelNet40 Upsampling Ratio — Recommendation Report

- Last updated: 2026-06-25
- Project: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`
- Audit data: `reports/modelnet40_upsampling_ratio_audit.csv`
- Correction note: `reports/modelnet40_protocol_x4_correction_note.md`

> **Previous recommendation R=×2 is superseded. Final main protocol uses R=×4.**

---

## Executive Recommendation

**Adopt unified upsampling factor R = 4 (×4) for all main-protocol xyz methods on both lines.**

| Line | Baseline (native) | Upsampling branch target | PointNet++ `num_point` |
| --- | ---: | ---: | ---: |
| A — Original | 1024 | **4096** | 4096 |
| B — Downsampled50 | 512 | **2048** | 2048 |

**Main methods:** EAR, PU-Net, PU-GCN, PDANS  
**Supplementary:** TULIP (`supplementary_only` — does not affect main ratio choice)

Rationale: ×4 matches upstream defaults for PDANS (`R=4`), PU-Net (`up_ratio=4`), and PU-GCN (`up_ratio=4`). Main xyz methods share a single fixed factor and fixed output counts per line. TULIP is excluded (range-image / unstable xyz counts).

**Do not resample upsampling outputs back to baseline native counts.**

---

## Q1. EAR, what's the current real multiplier?

| Scope | Input | Output | Actual ratio | Role under ×4 protocol |
| --- | ---: | ---: | ---: | --- |
| **Line B full** (`downsampled50_up/ear`, 12311) | 512 | 1024 | **×2.0000** | **Ablation / preliminary** — keep, not main ×4 |
| **Line B smoke** | 512 | 1024 | **×2.0000** | Preliminary smoke only |
| **Line A smoke** | 1024 | 1024 | **×1.0000** | Identity — not valid upsampling |
| **Line A full** | — | — | not generated | Main needs **1024→4096** |
| **Main target (both lines)** | 1024 / 512 | **4096 / 2048** | **×4** | To be generated after ×4 wrapper smoke |

Current wrapper (`scripts/ear_modelnet40_utils.py`): `TARGET_POINTS=1024`, `up_factor=2.0` → produces ×2 on Line B only.

---

## Q2. PU-Net, what's the current real multiplier?

| Scope | On-disk outputs | Configured default | Main ×4 target |
| --- | --- | --- | --- |
| ModelNet40 | **None** | `up_ratio=4` | 1024→**4096** (A), 512→**2048** (B) |

Default upstream ratio **already matches** main R=×4. Wrapper not yet implemented.

---

## Q3. PU-GCN, what's the current real multiplier?

| Scope | On-disk outputs | Configured default | Main ×4 target |
| --- | --- | --- | --- |
| ModelNet40 | **None** | `up_ratio=4` | 1024→**4096** (A), 512→**2048** (B) |

Default upstream ratio **already matches** main R=×4.

---

## Q4. PDANS, what's the current real multiplier?

| Scope | On-disk outputs | Configured default | Main ×4 target |
| --- | --- | --- | --- |
| ModelNet40 | **None** | `R=4` | 1024→**4096** (A), 512→**2048** (B) |

Default upstream ratio **already matches** main R=×4.

---

## Q5. What methods can be directly used as main method?

**Main xyz group:** EAR, PDANS, PU-Net, PU-GCN. **TULIP excluded.**

| Method | Line | Current on-disk | Main ×4 ready? |
| --- | --- | --- | --- |
| **EAR** | B | ×2 dataset exists | **No** — regenerate 512→2048 |
| **EAR** | A | smoke ×1 only | **No** — regenerate 1024→4096 |
| **PDANS** | A/B | no outputs | **No** — wrapper at R=4 |
| **PU-Net** | A/B | no outputs | **No** — wrapper at up_ratio=4 |
| **PU-GCN** | A/B | no outputs | **No** — wrapper at up_ratio=4 |

No main-protocol production dataset exists at ×4 yet. Existing Line B EAR ×2 is **ablation / preliminary only**.

---

## Q6. What methods do you need to regenerate the output?

| Method | Action | Status |
| --- | --- | --- |
| **EAR Line B** | Regenerate **512→2048** (×4); keep existing 512→1024 as ablation | Await ×4 smoke |
| **EAR Line A** | Regenerate **1024→4096** (×4) | Await ×4 smoke |
| **PDANS / PU-Net / PU-GCN** | Implement wrappers at default **×4** | Await confirmation |
| **Existing ×2 EAR** | **Do not delete** | Ablation reference |

---

## Q7. Recommended Utility is x 2 or x 4?

**Final main protocol: R = ×4.** (Previous ×2 recommendation is superseded.)

| Criterion | R=×2 (superseded) | **R=×4 (final)** |
| --- | --- | --- |
| PDANS / PU-Net / PU-GCN defaults | Requires override to ×2 | **Native ×4** |
| Line B EAR on disk | Matched ×2 | ×2 kept as ablation; main needs ×4 |
| Line A target | 2048 | **4096** |
| Line B target | 1024 | **2048** |
| TULIP influence | None | **None** (supplementary_only) |

---

## Q8., what are the points of output for each line?(Final protocol)

| Branch | Line A | Line B |
| --- | ---: | ---: |
| Baseline (native) | **1024** | **512** |
| Upsampling (×4, main methods) | **4096** | **2048** |

PointNet++ training (`allow_resample=false`):

- Line A baseline: `num_point=1024`
- Line A upsampling: `num_point=4096`
- Line B baseline: `num_point=512`
- Line B upsampling: `num_point=2048`

---

## Q9. Current `downsampled50_ear_pointnet2` Does it match the final multiplication protocol?

**No — ablation / preliminary only (×2, not main ×4).**

| Check | Result |
| --- | --- |
| Upsampling ratio | **×2** (512→1024) |
| Main protocol target | **×4** (512→2048) |
| Step 8 `num_point` | 1024 |
| Job 1711437 result (90.82%) | **Keep as preliminary** — not main-protocol comparison |
| Delete? | **No** |

Main Line B upsampling comparison awaits native-512 baseline + **×4 EAR** PointNet++ run.

---

## Q10. `downsampled50_baseline` loader 512  /  1024 can only be used as ablation?

**Yes — naive-resampled ablation only, not main baseline.**

| Run | `num_point` | Role |
| --- | ---: | --- |
| `downsampled50_baseline` (Step 6) | 1024 (loader) | Ablation |
| `downsampled50_native512_pointnet2` | 512 (native) | **Main baseline** |

---

## Supplementary / Special Method: TULIP

**Status: `supplementary_only`** — unchanged by ×4 main protocol selection.

| Question | Answer |
| --- | --- |
| Theory Multiplication | ×4 range-image (16×1024 → 64×1024) |
| ModelNet40 Actual multiplier | N/A (no outputs) |
| KITTI xyz Actual multiplier | Not ×4; Line A μ≈0.41, Line B μ≈0.75 |
| Output is stable? | No |
| Into main protocol? | **No** |
| Keep supplementary? | **Yes** |

TULIP does **not** participate in fixed xyz point-count main comparison and did **not** drive the choice of R=×4 for EAR/PU-Net/PU-GCN/PDANS.

---

## Next Steps

1. Reports corrected (this file + audit + correction note). **Done.**
2. **Do not** submit PointNet++ training yet.
3. **Do not** delete existing ×2 EAR data or Step 8 artifacts.
4. **Next planned action:** prepare EAR ×4 wrapper smoke test:
   - Original + EAR: **1024→4096**
   - Downsampled50 + EAR: **512→2048**

---

## Related Reports

- `reports/modelnet40_protocol_x4_correction_note.md`
- `reports/modelnet40_upsampling_ratio_audit.md`
- `reports/modelnet40_pointnet2_final_protocol_corrected.md`
