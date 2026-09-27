# PU-Net ×4 Smoke Audit

- Generated: 2026-07-01
- Jobs: **1725581** (Line A), **1725582** (Line B)
- Overall: **PASS**

| Line | Train | Test | Input | Output shape | NaN/Inf | Status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| A | 20 | 20 | 1024 | 4096×3 | no | **PASS** |
| B | 20 | 20 | 512 | 2048×3 | no | **PASS** |

## Method call evidence

- Wrapper: `scripts/punet_modelnet40_utils.py` → `PUNetUpsampler.upsample_xyz()`
- Inference: `PU-Net/code/main.py --phase test` with checkpoint `model/generator2_new6/model-120`
- `up_ratio=4`, native PU-Net generator (not generic resample)

## Output paths

| Line | Path |
| --- | --- |
| A | `datasets/modelnet40_original_up/punet_x4_smoke` |
| B | `datasets/modelnet40_downsampled50_up/punet_x4_smoke` |

## CSV

- Line A: `reports/punet_x4_smoke_audit_A.csv`
- Line B: `reports/punet_x4_smoke_audit_B.csv`
- Combined: `reports/punet_x4/punet_x4_smoke_audit.csv`
