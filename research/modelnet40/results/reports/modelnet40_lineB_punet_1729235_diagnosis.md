# Line B PU-Net Job 1729235 Diagnosis

- Generated at: 2026-07-05
- PROJECT_ROOT: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

## Job summary

| Field | Value |
| --- | --- |
| Job ID | **1729235** (`lineB_x4_pu_net`, array 0–15) |
| Overall outcome | **Partially completed** — 9150/12311 strict outputs |
| Time limit | `--time=24:00:00` per array task |

## Array task states (sacct)

| Task | State | Elapsed | Exit |
| ---: | --- | --- | ---: |
| 0 | COMPLETED | 05:02:54 | 0:0 |
| 1 | **TIMEOUT** | 1-00:00:10 | 0:0 |
| 2 | **TIMEOUT** | 1-00:00:10 | 0:0 |
| 3 | **TIMEOUT** | 1-00:00:10 | 0:0 |
| 4 | COMPLETED | 04:49:02 | 0:0 |
| 5 | **FAILED** | 05:01:23 | 1:0 |
| 6 | **TIMEOUT** | 1-00:00:13 | 0:0 |
| 7 | **TIMEOUT** | 1-00:00:13 | 0:0 |
| 8 | COMPLETED | 05:08:14 | 0:0 |
| 9 | COMPLETED | 05:10:55 | 0:0 |
| 10–15 | **TIMEOUT** | 1-00:00:28 | 0:0 |

**Primary stop reason:** **TIMEOUT** (11/16 chunks hit 24h wall clock before finishing ~770 samples/chunk).  
**Secondary:** chunk 5 **FAILED** with `PU-Net inference failed (rc=-6)` on `bookshelf_0424` (SIGABRT in TF subprocess).

## Log paths

- Pattern: `logs/lineB_downsampled_x4_up/lineB_x4_pu_net_1729235_{0..15}.{out,err}`
- Chunk audits: `logs/lineB_downsampled_x4_up/pu_net/chunk_audits/chunk_*.csv`

## Exact errors

1. **TIMEOUT:** Slurm killed tasks at exactly 24h while PU-Net subprocess loop still running (~140–290s per sample on RTX 3080).
2. **FAILED chunk 5:** `RuntimeError: PU-Net inference failed (rc=-6)` for `train/bookshelf/bookshelf_0424.npy` — likely TF/CUDA abort on that sample.

## Safe to resume?

**Yes.** Existing 9150 outputs quick-audit **PASS** (all `(1024,3)`, no NaN/Inf).  
Resume with missing-only list + `--skip-existing` / valid strict check — **do not** reprocess completed samples.

## Recommendation

1. Use `reports/modelnet40_lineB_punet_missing_samples.txt` (3161 paths).
2. Submit `jobs/lineB_upsampling/run_lineB_punet_256to1024_resume_missing.sbatch` (8 chunks × ~395 samples, 24h each).
3. Failed samples appended to `reports/modelnet40_lineB_punet_resume_failed_samples.csv`.
4. After 12311/12311: run final count audit + add PU-Net to PointNet++ inputs + schedule PU-Net quality metrics job.
