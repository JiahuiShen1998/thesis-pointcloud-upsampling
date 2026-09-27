# ModelNet40 PointNet++ Line B Smoke Audit

- Generated at: 2026-07-07 12:27:00 UTC
- Submit time: 2026-07-07 12:25:06 UTC
- Submit command: `sbatch.tinygpu`
- Job IDs: 1733159, 1733160, 1733161, 1733162, 1733163
- Overall result: **1 / 5 PASS** — full training **blocked**

## Summary

| branch | method | expected pts | actual pts | job id | state | exit | status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| lineB_downsampled_x4_baseline_256 | baseline | 256 | 256 | 1733159 | COMPLETED | 0:0 | **PASS** |
| lineB_ear_1024 | EAR | 1024 | — | 1733160 | FAILED | 1:0 | FAIL |
| lineB_pdans_1024 | PDANS | 1024 | — | 1733161 | FAILED | 1:0 | FAIL |
| lineB_punet_1024 | PU-Net | 1024 | — | 1733162 | FAILED | 1:0 | FAIL |
| lineB_pugcn_1024 | PU-GCN | 1024 | — | 1733163 | FAILED | 1:0 | FAIL |

## PASS branch detail: lineB_downsampled_x4_baseline_256

- Input: `pointnet2_inputs/lineB_downsampled_x4_baseline`
- Log: `logs/pointnet2_smoke/lineB_downsampled_x4_baseline_256/slurm_1733159.out`
- Train log: `pointnet2_results/x4_two_line_smoke/lineB_downsampled_x4_baseline/train.log`
- `num_point=256`, `allow_resample=False`
- First train sample shape: `(256, 3)`
- First train batch tensor shape: `(8, 3, 256)` → actual point count **256**
- First train batch loss: **3.952901** (finite)
- Forward / backward / eval loop: all completed (2 train batches, 2 eval batches)
- `metrics.json`: exists at `pointnet2_results/x4_two_line_smoke/lineB_downsampled_x4_baseline/metrics.json`
- Contains `final_test_overall_accuracy`, `final_test_class_accuracy`, `num_point=256`
- No NaN, Inf, CUDA OOM, or dataloader point-count mismatch observed

## FAIL branches: EAR / PDANS / PU-Net / PU-GCN

All four upsampling branches failed **before the first training batch** during dataloader initialization:

```
FileNotFoundError: Missing label map: .../pointnet2_inputs/lineB_downsampled_x4_up/<method>/metadata/class_to_idx.json
```

- Root cause: `datasets/lineB_downsampled_x4_up/strict_N/{ear,pdans,pu_net,pu_gcn}/` contain only `train/` and `test/` — **no `metadata/` directory**.
- Baseline `modelnet40_downsampled_x4` has full metadata (`class_to_idx.json`, manifests) and therefore smoke PASS.
- This is **not** a dataloader point-count issue, GPU OOM, or environment/module failure.
- PyTorch 2.6.0 + CUDA available on GPU node; failure occurs on CPU-side dataset init.

## Error scan

```bash
grep -RniE "traceback|error|failed|exception" logs/pointnet2_smoke/lineB_* 2>/dev/null
```

All four failures share the same `FileNotFoundError` for `metadata/class_to_idx.json`.

## Conclusion

- **5 / 5 Line B smoke PASS:** No
- **Line B full training ready:** No — blocked until metadata issue is fixed and upsampling smokes re-run
- See: `reports/modelnet40_pointnet2_lineB_smoke_failure_diagnosis.md`
