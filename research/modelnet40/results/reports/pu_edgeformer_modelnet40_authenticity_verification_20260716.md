# PU-EdgeFormer ModelNet40 Authenticity Verification

- Generated: `2026-07-16 18:31:30 UTC`
- Final conclusion: **TRUSTED: can proceed to geometry metrics**

## Conclusion table

| Check | Line B | Line A | Status |
|------|--------|--------|--------|
| checkpoint restored | YES | YES | PASS |
| PU-EdgeFormer model code used | YES | YES | PASS |
| not simple repeat ×4 | YES | YES | PASS |
| not copy of input | YES | YES | PASS |
| not PU-GCN output | YES | YES | PASS |
| not smoke leakage | YES | YES | PASS |
| output fresh | YES | YES | PASS |
| data trustworthy for geometry | YES | YES | PASS |
| data trustworthy for PointNet++ | YES | YES | PASS |

## Component results

### 1) Repeat/copy audit — PASS
- Report: `reports/pu_edgeformer_repeat_copy_audit_20260716.md`
- CSV: `reports/pu_edgeformer_repeat_copy_audit_20260716.csv`
- Sampled: 200 / line

Line B means:
- exact_match_ratio = 0.000000
- rounded_1e6_match_ratio = 0.000000
- output_unique_count = 1024.0 (not ~256)
- duplicate_ratio = 0.000000 (not ~0.75)
- nn_median ≈ 2.55e-2
- nn_pct < 1e-8 / < 1e-6 = 0 / 0
- exact repeat/tile hits = 0
- shuffled_repeat frac_eq_4 = 0

Line A means:
- exact_match_ratio = 0.000000
- rounded_1e6_match_ratio = 0.000000
- output_unique_count = 4096.0 (not ~1024)
- duplicate_ratio = 0.000000
- nn_median ≈ 1.83e-2
- nn_pct < 1e-8 / < 1e-6 = 0 / 0
- exact repeat/tile hits = 0
- shuffled_repeat frac_eq_4 = 0

Fail conditions (exact/rounded >0.95, dup>0.60, unique≈input, repeat/tile/shuffle) triggered on **0** samples.

### 2) Model-code / call-path / log audit — PASS
- Report: `reports/pu_edgeformer_model_callpath_slurm_audit_20260716.md`
- tf_ops reused from PU-GCN: **YES**
- model definition from PU-EdgeFormer (`EdgeTransformer` via `--model edgetransformer`): **YES**
- Infer-log restore to `checkpoint_model100/model-100`: Line B 64/64, Line A 128/128
- Every batch CMD includes `--method PU-EdgeFormer --model edgetransformer`

### 3) Checkpoint variable audit — PASS
- Report: `reports/pu_edgeformer_checkpoint_variable_audit_20260716.txt`
- 135 variables; includes attention QKV (`conv_q/k/v`), `up_block/duplicate`, `fc_layer1/2/4`
- No `pugcn` variable-name hits

### 4) PU-EdgeFormer vs PU-GCN difference — PASS
- Report: `reports/pu_edgeformer_vs_pugcn_output_difference_audit_20260716.md`
- CSV: `reports/pu_edgeformer_vs_pugcn_output_difference_audit_20260716.csv`
- Sampled: 100 / line from 12311 common IDs
- identical/near: **0 / 0**
- Line B mean max_abs_diff ≈ 1.58, chamfer_like ≈ 2.70e-2
- Line A mean max_abs_diff ≈ 1.59, chamfer_like ≈ 1.67e-2

### 5) Smoke leakage — PASS
- Report: `reports/pu_edgeformer_smoke_leakage_audit_20260716.md`
- Overlapping smoke samples are content-equal (deterministic seed), but full mtimes are **newer than smoke** → re-inference, not file copy
- copy-suspect: 0 / 0

### 6) Output freshness — PASS
- Report: `reports/pu_edgeformer_output_freshness_audit_20260716.md`
- Line B 12311/12311 files inside job 1749211 window `20:15:11`–`20:17:47`
- Line A 12311/12311 files inside job 1749212 window `20:16:10`–`20:20:40`

## Final conclusion

**TRUSTED: can proceed to geometry metrics**

## Safety (this verification round)

- GEOMETRY_METRICS_STARTED=NO
- POINTNET_CLASSIFIER_STARTED=NO
- DETECTOR_EVAL_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO

## Report paths

- `reports/pu_edgeformer_modelnet40_authenticity_verification_20260716.md`
- `reports/pu_edgeformer_repeat_copy_audit_20260716.md`
- `reports/pu_edgeformer_repeat_copy_audit_20260716.csv`
- `reports/pu_edgeformer_model_callpath_slurm_audit_20260716.md`
- `reports/pu_edgeformer_checkpoint_variable_audit_20260716.txt`
- `reports/pu_edgeformer_vs_pugcn_output_difference_audit_20260716.md`
- `reports/pu_edgeformer_vs_pugcn_output_difference_audit_20260716.csv`
- `reports/pu_edgeformer_smoke_leakage_audit_20260716.md`
- `reports/pu_edgeformer_output_freshness_audit_20260716.md`
