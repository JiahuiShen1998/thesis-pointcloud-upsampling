# ModelNet40 Protocol Correction — R=×4 (Main Protocol)

- Generated at: 2026-06-25
- Project: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

---

## Summary

- **Previous recommendation R=×2 is superseded.**
- **Final main protocol uses R=×4.**
- **TULIP remains supplementary and does not affect the main ratio choice.**

---

## Main Protocol (R=×4)

| Branch | Line | Pipeline | PointNet++ `num_point` |
| --- | --- | --- | ---: |
| Baseline | A — Original | 1024 native | 1024 |
| Upsampling | A — Original + main methods | 1024 → ×4 → **4096** | 4096 |
| Baseline | B — Downsampled50 | 512 native | 512 |
| Upsampling | B — Downsampled50 + main methods | 512 → ×4 → **2048** | 2048 |

**Main methods:** EAR, PU-Net, PU-GCN, PDANS  
**Supplementary:** TULIP (`supplementary_only`)

---

## Reclassification of Existing Results

| Artifact | Current state | New role |
| --- | --- | --- |
| `downsampled50_up/ear` (512→1024) | ×2 full dataset | **Ablation / preliminary only** — keep, do not delete |
| `downsampled50_ear_pointnet2` (Step 8) | 1024 input, Job 1711437 | **Ablation / preliminary only** — not main ×4 result |
| `downsampled50_baseline` (512→1024 loader) | Step 6 | **Naive-resampled ablation only** |
| `downsampled50_native512_pointnet2` | native 512 | **Main Line B baseline** (pending training) |
| Main Line B + EAR | not yet generated | **Regenerate 512→2048** |
| Main Line A + EAR | not yet generated | **Regenerate 1024→4096** |

---

## Next Planned Action (after this correction)

Prepare **EAR ×4 wrapper smoke test** (no full run, no PointNet++ training yet):

- Original + EAR: **1024 → 4096**
- Downsampled50 + EAR: **512 → 2048**

---

## References

- `reports/modelnet40_upsampling_ratio_recommendation.md`
- `reports/modelnet40_upsampling_ratio_audit.md`
- `reports/modelnet40_pointnet2_final_protocol_corrected.md`
