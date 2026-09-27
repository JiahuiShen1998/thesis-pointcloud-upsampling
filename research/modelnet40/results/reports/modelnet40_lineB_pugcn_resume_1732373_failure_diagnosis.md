# PU-GCN Line B Resume Job 1732373 — Failure Diagnosis

**Diagnosis time:** 2026-07-06

## Job summary

| Field | Value |
| --- | --- |
| Job ID | **1732373** |
| Job name | `mn40_B_pugcn_resume` |
| Array | 0–7 |
| State | **All FAILED** (ExitCode 1:0) |
| Elapsed | ~23–51 seconds per task |

## Node / GPU assignment

Tasks were scheduled on **RTX 2080 Ti** nodes (not tg084 / RTX 3080):

| Task | Node | Elapsed |
| --- | --- | --- |
| 1732373_0 | tg06a | 00:00:48 |
| 1732373_1 | tg06a | 00:00:51 |
| 1732373_2 | tg06b | 00:00:32 |
| 1732373_3 | tg06b | 00:00:23 |
| 1732373_4 | tg06b | 00:00:32 |
| 1732373_5 | tg060 | 00:00:24 |
| 1732373_6 | tg060 | 00:00:25 |
| 1732373_7 | tg060 | 00:00:35 |

GPU was allocated normally (`nvidia-smi` printed in logs). **This was not a GPU inference failure.**

## Root cause

**Python SyntaxError** — script failed at parse/import time before any PU-GCN inference.

```
File "scripts/run_lineB_pugcn_resume_missing.py", line 160
    global EXPECTED_INPUT_POINTS, EXPECTED_OUTPUT_POINTS
SyntaxError: name 'EXPECTED_INPUT_POINTS' is used prior to global declaration
```

The `main()` function referenced `EXPECTED_INPUT_POINTS` / `EXPECTED_OUTPUT_POINTS` in `argparse` defaults (lines 147–148) before declaring them `global` (line 160).

## Impact

- Runner **never started** processing missing samples.
- No PU-GCN inference was attempted.
- **raw / strict counts unchanged:** 10010 / 12311
- Existing 10010 outputs unaffected (quick audit still PASS).

## Fix

Remove `global` usage; pass `expected_input_points` / `expected_output_points` as local variables into `process_row()`.

## Not this failure

- Not `knn_point_2` GPU error (that was job 1731794 on tg084 / RTX 3080).
- Not node allocation failure.
- Not checkpoint / missing-list path error.
