# PU-GCN Line B Job 1731794 — Cancelled for Resume

**Action time:** 2026-07-06 19:43 CEST

## Original job

| Field | Value |
| --- | --- |
| Job ID | **1731794** |
| Job name | `mn40_B_pugcn_256to1024` |
| Array | 0–15 |

## Line B state at cancellation

| Metric | Value |
| --- | --- |
| raw | **10010 / 12311** |
| strict_N | **10010 / 12311** |
| missing | **2301** |

## Stall reason

- Array tasks **13 / 14 / 15** ran on **tg084** with **NVIDIA RTX 3080**.
- No new raw or strict `.npy` files in the last **120+ minutes** before cancel.
- Logs showed **100% inference failure** (`success=0`, `failed=N`) with:
  - `PU-GCN inference failed (rc=1)`
  - traceback at **`knn_point_2`** in PU-GCN tf_ops
- Tasks **0–12** on **RTX 2080 Ti** completed successfully (~2.5 h each).

## Existing output validation

- Quick audit of **10010** strict outputs: **PASS** (all `(1024, 3)`, zero NaN/Inf)
- Report: `reports/modelnet40_lineB_pugcn_existing_quick_audit_after_cancel.csv`

## Decision

**`scancel 1731794`** executed — tasks 13/14/15 were **CANCELLED** (not natural TIMEOUT).

Post-cancel sacct:

| Task | State |
| --- | --- |
| 1731794_0 … _12 | COMPLETED |
| 1731794_13 | CANCELLED |
| 1731794_14 | CANCELLED |
| 1731794_15 | CANCELLED |

## Next step

Immediately submit resume job using:

- Missing list: `reports/modelnet40_lineB_pugcn_missing_samples.txt` (2301 samples)
- Exclude RTX 3080 nodes / tg084
- Prefer RTX 2080 Ti (`--gres=gpu:rtx2080ti:1`, `--exclude=tg080–tg085`)
- **Retain** all 10010 existing raw/strict outputs; do not re-run them

## Important

**This is not dropping PU-GCN.** This is resuming PU-GCN to full completion (12311/12311).
