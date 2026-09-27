# Line B PU-Net Smoke Audit

- Generated at: 2026-07-03
- Status: **PASS**
- Job: 1729225 (`lineB_punet_smoke`)
- Fix: `strict_out.parent.mkdir()` before save (mkdir bug from job 1727073)

| metric | value |
| --- | --- |
| samples | 10 (5 train + 5 test) |
| failed | 0 |
| input points | 256 |
| output points | 1024 |
| input source | `datasets/modelnet40_downsampled_x4/` |

All 10 samples: `status=success`, `final_points=1024`, no NaN/Inf.

CSV: `reports/modelnet40_lineB_punet_smoke_audit.csv`
