# ModelNet40 + PointNet++ — Protocol Correction Log

- Last updated: 2026-06-25
- Canonical document: `reports/modelnet40_pointnet2_final_protocol.md`
- Latest ratio note: `reports/modelnet40_protocol_x4_correction_note.md`

---

## Latest Change (2026-06-25): R=×2 → R=×4

| Topic | Superseded (×2 draft) | **Final (×4)** |
| --- | --- | --- |
| Main upsampling factor | R=×2 | **R=×4** |
| Line A upsampling target | 2048 | **4096** |
| Line B upsampling target | 1024 | **2048** |
| Line B EAR full dataset (512→1024) | Called `main_compatible` | **Ablation / preliminary only** |
| Step 8 `downsampled50_ear_pointnet2` | Main Line B upsampling result | **Ablation / preliminary only** |
| PDANS/PU-Net/PU-GCN default ×4 | Mismatch with ×2 plan | **Matches main protocol** |
| TULIP | `supplementary_only` | **`supplementary_only` (unchanged)** |

**Previous recommendation R=×2 is superseded. Final main protocol uses R=×4. TULIP does not affect this choice.**

---

## Method Groups

### Main methods (fixed xyz point-count protocol, R=×4)

- EAR
- PU-Net
- PU-GCN
- PDANS

### Supplementary / special method

- TULIP — range-image / LiDAR-style; `supplementary_only`; not in main xyz comparison

---

## Point-Count Protocol (final)

| Branch | Line | Native input | Upsampling (×4) | PointNet++ `num_point` |
| --- | --- | ---: | ---: | ---: |
| Baseline | A | 1024 | — | 1024 |
| Upsampling | A | 1024 | **4096** | 4096 |
| Baseline | B | 512 | — | 512 |
| Upsampling | B | 512 | **2048** | 2048 |

Rules (unchanged):

- Baseline native count is preserved
- Upsampled count is preserved
- No baseline-to-upsampling point matching
- No upsampling-to-baseline clipping
- Same upsampling factor across main methods within each line
- Original line and Downsampled50 line are independent

---

## Experiment Reclassification

| Experiment | Line | `num_point` | Ratio | Role |
| --- | --- | ---: | ---: | --- |
| `original_baseline` | A | 1024 | — | Main baseline |
| `original_*_up` (future) | A | 4096 | ×4 | Main upsampling (to regenerate) |
| `downsampled50_native512_pointnet2` | B | 512 | — | **Main baseline** |
| `downsampled50_up/ear` + Step 8 | B | 1024 | ×2 | **Ablation / preliminary** — keep |
| `downsampled50_baseline` (Step 6) | B | 1024 (loader) | — | **Naive-resampled ablation** |
| TULIP | A/B | N/A | — | **Supplementary only** |

---

## Correction History

### Earlier correction (same day)

Separated baseline branch vs upsampling branch; baseline and upsampling may differ in `num_point` within a line.

### Current correction

Selected **R=×4** as main protocol; reclassified existing ×2 EAR artifacts as ablation/preliminary; aligned main targets with upstream PDANS/PU-Net/PU-GCN defaults.

---

## Next Steps

1. Correct reports (this note + recommendation + audit).
2. Prepare EAR ×4 wrapper smoke test: 1024→4096 (A), 512→2048 (B).
3. **Do not** submit PointNet++ training until ×4 smoke passes.
4. **Do not** delete existing ×2 EAR data or Step 8 artifacts.

---

## Stale Files (update when touched)

- `reports/modelnet40_upsampling_ratio_audit.csv` — Line B EAR still marked `main_compatible` until script re-run
- `scripts/ear_modelnet40_utils.py` — `TARGET_POINTS=1024` (×2 wrapper)
- `reports/step8_protocol_audit_*.md` — pre-×4 framing
