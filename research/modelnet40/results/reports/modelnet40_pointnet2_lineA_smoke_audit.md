# ModelNet40 PointNet++ Line A Smoke Audit

- Generated at: 2026-07-07 20:12:00 UTC
- Submit time: 2026-07-07 20:10:18 UTC
- Submit command: `sbatch.tinygpu`
- Job IDs: 1733475, 1733476, 1733477, 1733478, 1733479
- Overall result: **1 / 5 PASS** — full training **blocked**

## Summary

| branch | method | expected pts | actual pts | job id | state | exit | status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| lineA_original_baseline_1024 | baseline | 1024 | 1024 | 1733475 | COMPLETED | 0:0 | **PASS** |
| lineA_ear_4096 | EAR | 4096 | — | 1733476 | FAILED | 1:0 | **FAIL** |
| lineA_pdans_4096 | PDANS | 4096 | — | 1733477 | FAILED | 1:0 | **FAIL** |
| lineA_punet_4096 | PU-Net | 4096 | — | 1733478 | FAILED | 1:0 | **FAIL** |
| lineA_pugcn_4096 | PU-GCN | 4096 | — | 1733479 | FAILED | 1:0 | **FAIL** |

## PASS branch detail: lineA_original_baseline_1024

- Input: `pointnet2_inputs/lineA_original_baseline`
- Log: `logs/pointnet2_smoke/lineA_original_baseline_1024/slurm_1733475.out`
- Train log: `pointnet2_results/x4_two_line_smoke/lineA_original_baseline/train.log`
- `num_point=1024`, `allow_resample=False`
- First train sample shape: `(1024, 3)`
- First train batch tensor shape: `(8, 3, 1024)` → actual point count **1024**
- First train batch loss: **3.902978** (finite)
- Forward / backward / eval loop: all completed (2 train batches, 2 eval batches)
- `metrics.json`: exists at `pointnet2_results/x4_two_line_smoke/lineA_original_baseline/metrics.json`
- Contains `final_test_overall_accuracy`, `final_test_class_accuracy`, `num_point=1024`
- No NaN, Inf, CUDA OOM, or dataloader point-count mismatch observed

## FAIL branches: EAR / PDANS / PU-Net / PU-GCN

All four upsampling branches failed **before the first training batch** during dataloader initialization:

```
FileNotFoundError: Missing label map: .../pointnet2_inputs/lineA_original_up/<method>/metadata/class_to_idx.json
```

- Root cause: `datasets/lineA_original_up/strict_4N/{ear,pdans,pu_net,pu_gcn}/` contain only `train/` and `test/` — **no `metadata/` directory**.
- Baseline `modelnet40_original` has full metadata (`class_to_idx.json`, manifests) and therefore smoke PASS.
- This is **not** a dataloader point-count issue, GPU OOM, or environment/module failure.
- PyTorch 2.6.0 + CUDA available on GPU node; failure occurs on CPU-side dataset init.
- Same failure pattern as Line B smoke round 1 (fixed via metadata symlinks — see `reports/modelnet40_pointnet2_lineB_metadata_fix_audit.md`).

## Error scan

```bash
grep -RniE "traceback|error|failed|exception" logs/pointnet2_smoke/lineA_* 2>/dev/null
```

All four failures share the same `FileNotFoundError` for `metadata/class_to_idx.json`.

## Conclusion

- **5 / 5 Line A smoke PASS:** No
- **Line A full training ready:** No — blocked until metadata issue is fixed and upsampling smokes re-run
- See: `reports/modelnet40_pointnet2_lineA_smoke_failure_diagnosis.md`
