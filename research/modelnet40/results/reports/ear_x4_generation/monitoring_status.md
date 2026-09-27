# EAR ×4 Full Generation — Monitoring Status

- Updated: 2026-06-29
- Project: `modelnet40_pointnet2_upsampling`

## Job states

| Job | Name | State | Notes |
| --- | --- | --- | --- |
| **1716174** | `ear_x4_a` | **COMPLETED** | Line A full generation |
| **1716175** | `ear_x4_b` | COMPLETED (tasks 4,6 failed) | Original array |
| **1717034** | `ear_x4_b_retry` | **COMPLETED** | Tasks 4 & 6 retry successful |

## Full output counts

| Line | Directory | train | test | total | Expected |
| --- | --- | ---: | ---: | ---: | ---: |
| **A** | `modelnet40_original_up/ear_x4` | 9843 | 2468 | **12311** | 12311 |
| **B** | `modelnet40_downsampled50_up/ear_x4` | 9843 | 2468 | **12311** | 12311 |

## Full audit — **PASS**

| Script | Report | Result |
| --- | --- | --- |
| `audit_ear_x4_full.py` | `reports/ear_x4_full_generation_audit.{md,csv}` | **PASS** |
| `audit_upsampling_ratios.py` | `reports/modelnet40_upsampling_ratio_audit.{md,csv}` | **PASS** |

### Audit criteria

| Line | Shape | Ratio | Counts | NaN/Inf |
| --- | --- | ---: | --- | --- |
| Original + EAR ×4 (A) | 4096×3 | 4.0 | 9843+2468 | none |
| Downsampled50 + EAR ×4 (B) | 2048×3 | 4.0 | 9843+2468 | none |

EAR ×4 PointNet++ training: still **BLOCKED** until explicit user approval (generation audit PASS is necessary but not sufficient alone per protocol).

## Actions this pass

- Retry 1717034 completed — Line B reached 12311
- Full audit executed and PASS
- No data deleted; no full Line B rerun
