# PU-GCN Line B Stall Diagnosis — Job 1731794

**Diagnosis time:** 2026-07-06 18:55 CEST (16:55 UTC)

---

## 1. data count

| Indicators | Number |
|------|------|
| Line A strict (`strict_4N/pu_gcn`) | **12311 / 12311** (completed) |
| Line B raw (`raw/pu_gcn`) | **10010** |
| Line B strict (`strict_N/pu_gcn`) | **10010** |
|  Expected input  (`modelnet40_downsampled_x4`) | **12311** |
| Line B missing | **2301** |

**Judgement:** `raw == strict == 10010`, the reasoning with strict writing stagnates,**It's not.** raw>strict's Back-up Problem.

---

## 2. Slurm Status (1731794))

| Task | State | Elapsed | ExitCode | Node | GPU |
|------|-------|---------|----------|------|-----|
| 0–12 | COMPLETED | ~2h31m–2h41m | 0:0 | tg066/tg068/tg069, etc. | RTX **2080 Ti** |
| **1731794_13** | **RUNNING** | **21:39:01** | 0:0 | tg084 | RTX **3080** |
| **1731794_14** | **RUNNING** | **21:39:01** | 0:0 | tg084 | RTX **3080** |
| **1731794_15** | **RUNNING** | **21:39:01** | 0:0 | tg084 | RTX **3080** |

- No TIMEOUT / FAILED / CANCELLED / OUT_OF_MEMORY (sacct Record)
- `#SBATCH --time=24:00:00` ~ About TIMEOUT **2h21m**
- The three remaining task are at the same node tg084

---

## 3. Recent Document Writing

| Window | strict New File | raw New File |
|------|--------------|-----------|
| 30 min | 0 | 0 |
| 60 min | 0 | 0 |
| 120 min | 0 | 0 |

**Conclusions:** Although Slurm shows RUNNING, **120 No valid output within minutes**.

---

## 4. Log Path

Main logs:
- `logs/pugcn_x4/lineB_pugcn_1731794_13.out` / `.err`
- `logs/pugcn_x4/lineB_pugcn_1731794_14.out` / `.err`
- `logs/pugcn_x4/lineB_pugcn_1731794_15.out` / `.err`

Failed sample record:
- `logs/pugcn_x4/lineB_chunk_13_failed.csv` (~442 failures)
- `logs/pugcn_x4/lineB_chunk_14_failed.csv` (~454 failures)
- `logs/pugcn_x4/lineB_chunk_15_failed.csv` (~458 failures)

Full log list: `reports/modelnet40_pugcn_lineB_log_files.txt`

---

## 5. Error Analysis

### The phenomenon
The Tasks 13–15 log continues to print:
```
[lineB] N/770 success=0 skipped=0 failed=N elapsed=...
```
That's... **success has always been 0, failed linear growth**, about 170s/ sample, but none `.npy` Write.

### Error Type
Failed All As `PU-GCN inference failed (rc=1)`, subprocess TensorFlow traceback points to:
```
knn_point_2() → dil_knn() → densegcn() → batch_mat_mul_v2
```
Example of first failed sample:
- chunk 13: `train/sofa/sofa_0425`
- chunk 14: `train/table/table_0301`
- chunk 15: `train/tv_stand/tv_stand_0172`

### Key linkages
- Tasks **0–12**(RTX 2080 Ti): All COMPLETED,`success=770` or `success=765 skipped=5`
- Tasks **13–15** (RTX 3080, tg084): **100% Failed to reason**

I doubt it. **RTX 3080 on PU-GCN tf_ops / CUDA**instead of the problem of the sample itself.

---

## 6. Missing Samples

- List: `reports/modelnet40_lineB_pugcn_missing_samples.txt`
- Total: **2301**(770×3, corresponding to chunks 13/14/15)
- Distribution: **All in train split**
  - train/vase: 475
  - train/table: 392
  - train/toilet: 344
  - train/tv_stand: 267
  - train/sofa: 256
  - ... (for more details) `reports/modelnet40_lineB_pugcn_missing_distribution.txt`)

---

## 7. Completed Output Quick Audit

- Inspection: **10010** A strict `.npy`
- Failed: **0**
- All PASS: shape `(1024, 3)`, no NaN/Inf
- Reports: `reports/modelnet40_lineB_pugcn_existing_quick_audit.csv`

**Completed 10010 outputs can be safely preserved.**

---

## 8. Comprehensive judgement

| Dimensions | Conclusions |
|------|------|
| still progressing | • No new document output |
| likely stalled but jobs still running | ✅ **Yes.** - CPU is running but 100% failed in reasoning. |
| timeout expected soon |  /  running 21h39m / 24h limit, about 2h21m after TIMEOUT |
| failed |  /  sacct still RUNNING but functionally failed (zero output) |
| safe to resume missing samples |  /  There are already 10010 PASS, but resume 2301 missing |

**Diagnosis:** **Don't wait any longer.** The current three task failed to retest the sample empty and will not produce any new `.npy`. Continue to wait only to consume the quota until 24h TIMEOUT.

---

## 9. recommends next steps

### Recommended scheme (B + C group)

1. **You don't have to take the initiative, scancel**(This round is not implemented) - But wait ~ 2h Natural TIMEOUT, or manual if you are anxious to release resources `scancel 1731794_13 1731794_14 1731794_15`.
2. **Reservations** There are already 10010 raw/strict outputs (audit PASS).
3. **Create resume missing job**(Next round, not submitted in current round):
   - Input list: `reports/modelnet40_lineB_pugcn_missing_samples.txt`(2301 sample)
   - Use `--resume` Skip Existing Output
   - **Key:** Limit to RTX 2080 Ti (or the same type of GPU as successful chunks) to avoid reassigning to tg084/RTX 3080
   -  Consider it.  resume  Previous  3080  Recompile Up  tf_ops  Or add  GPU  Constraints
4. **Don't.** Rerun Line A or completed 10010 Line B output.
5. **Don't.** Run post-process (raw == strict, no reprocessing backlog).

### Not applicable
- **Programme D (raw) > strict):** Not applicable, raw == strict.

---

## 10. Subsidiary report

| Documentation | Contents |
|------|------|
| `reports/modelnet40_pugcn_lineB_1731794_slurm_status.txt` | Slurm State |
| `reports/modelnet40_pugcn_lineB_count_status.txt` | Count |
| `reports/modelnet40_pugcn_lineB_recent_write_status.txt` | Recent Writing |
| `reports/modelnet40_pugcn_lineB_error_scan.txt` | Error Scan |
| `reports/modelnet40_lineB_pugcn_missing_samples.txt` | Missing list |
| `reports/modelnet40_lineB_pugcn_missing_distribution.txt` | Missing distribution |
| `reports/modelnet40_lineB_pugcn_existing_quick_audit.csv` | Output audit already in place |
