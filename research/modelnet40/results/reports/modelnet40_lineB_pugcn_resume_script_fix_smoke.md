# PU-GCN Line B Resume Script Fix — Smoke Test

**Time:** 2026-07-06 19:55 CEST

## Fix applied

**File:** `scripts/run_lineB_pugcn_resume_missing.py`

**Problem:** `global EXPECTED_INPUT_POINTS, EXPECTED_OUTPUT_POINTS` at line 160, after those names were already used in `argparse` defaults → **SyntaxError** at import time.

**Fix (approach A):**
- Removed `global` declarations entirely.
- Renamed module constants to `DEFAULT_EXPECTED_INPUT_POINTS` / `DEFAULT_EXPECTED_OUTPUT_POINTS`.
- Pass `expected_input_points` / `expected_output_points` as explicit parameters to `process_row()`.
- Added `--max-samples` for smoke / partial runs.

## py_compile

```
python3 -m py_compile scripts/run_lineB_pugcn_resume_missing.py
→ PASS
```

## Smoke job

| Field | Value |
| --- | --- |
| Job ID | **1732385** |
| sbatch | `jobs/pugcn_x4/run_lineB_pugcn_resume_smoke2.sbatch` |
| Node | tg06a (RTX 2080 Ti) |
| Samples | 2 (`sofa_0425`, `sofa_0426`) |
| Elapsed | 00:01:26 |
| Exit code | 0:0 |

## Smoke results

```
[lineB_pugcn_resume] chunk=0 2/2 success=2 skipped=0 failed=0 elapsed=52.8s
PASS .../train/sofa/sofa_0425.npy
PASS .../train/sofa/sofa_0426.npy
SMOKE_AUDIT_PASS
```

- No SyntaxError
- Missing list read OK
- Input `.npy` loaded
- Raw + strict outputs written
- Strict shape = **(1024, 3)**, no NaN/Inf

## Note

Smoke produced 2 valid outputs from the missing set (10010 → 10012 strict). Full resume uses `--resume` and will skip these.
