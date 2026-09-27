# Line B PU-Net Smoke Failure Diagnosis

- Generated at: 2026-07-03 (post-mortem on job 1727073)
- PROJECT_ROOT: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

## Job summary

| Field | Value |
| --- | --- |
| Job ID | **1727073** |
| Job name | `lineB_x4_pu_net_smoke` |
| State | **FAILED** |
| Exit code | **1:0** |
| Elapsed | ~17 minutes |
| Full job 1727074 | **CANCELED** (was `PENDING`, `DependencyNeverSatisfied`) |

## Log paths

- stdout: `logs/lineB_downsampled_x4_up/lineB_x4_pu_net_smoke_1727073.out`
- stderr: `logs/lineB_downsampled_x4_up/lineB_x4_pu_net_smoke_1727073.err`
- smoke audit CSV: `reports/lineB_downsampled_x4_smoke_pu_net.csv`

## Last log summary

1. Environment activated successfully (`tf15_upsampling`, CUDA 11.8, TF 1.15).
2. TF custom ops recompiled for PU-Net (~15 min).
3. Smoke script ran 40 samples (20 train + 20 test).
4. **All 40 samples failed** with identical root cause.

## Exact error message

```
FileNotFoundError: [Errno 2] No such file or directory:
  '.../datasets/lineB_downsampled_x4_up_smoke/strict_N/pu_net/test/airplane/airplane_0627.npy'
```

Traceback points to `run_lineB_downsampled_x4_smoke.py` when saving outputs after PU-Net inference.

## Likely cause

**Not a PU-Net model / CUDA / input-shape failure.**

PU-Net inference subprocess completed; failure occurred when writing outputs because the **parent directory for `strict_N/pu_net/...` was not created** before `np.save()`. The smoke script only called `raw_out.parent.mkdir()` but not `strict_out.parent.mkdir()`.

This is the same class of bug that affected PDANS smoke (job 1727075) and was fixed on 2026-07-02 by adding:

```python
strict_out.parent.mkdir(parents=True, exist_ok=True)
```

Job 1727073 ran **before** that fix was deployed.

## Proposed fix

1. ✅ `strict_out.parent.mkdir(parents=True, exist_ok=True)` already in `scripts/run_lineB_downsampled_x4_smoke.py` (lines 135–136).
2. ✅ Same fix in `scripts/run_lineB_downsampled_x4_upsampling_chunk.py` for full generation.
3. Cancel dependent full job 1727074 — **done**.
4. Resubmit **smoke only** with `--max-per-split 5` (10 samples) via new job script.
5. Do **not** submit full until smoke PASS.

## PU-Net protocol note

- Input: `datasets/modelnet40_downsampled_x4/` — **256 points** (confirmed in smoke CSV `input_points=256`).
- PU-Net wrapper passes `--num_point 256 --up_ratio 4` — valid ×4 upsampling path.
- Expected output: ~1024 points → strict normalized to exactly 1024.
- No Line A 4096 or old 512→2048 outputs involved.

## Full job action

- Job **1727074** canceled via `scancel 1727074`.
- Full job will be resubmitted only after new smoke PASS.
