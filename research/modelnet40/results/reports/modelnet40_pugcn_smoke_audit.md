# PU-GCN Smoke Audit

- Generated at: 2026-07-05 19:15 UTC (post-fix combined audit)
- Overall: **PASS**
- Line A: **PASS** (5 train + 5 test, 1024→4096, shape (4096,3), zero NaN/Inf)
- Line B: **PASS** (5 train + 5 test, 256→1024, shape (1024,3), zero NaN/Inf)

Smoke jobs:
- Line A: **1731790** — 10/10 success
- Line B: **1731791** — 10/10 success

Fixes applied before PASS:
1. `open3d` / `plyfile` env deps for TF15 upsampling
2. Per-process `PUGCN_OUT_FOLDER` to avoid parallel result-dir race
3. Smoke sample selection aligned between runner and audit (sorted manifest, 5/split)

CSV: `reports/modelnet40_pugcn_smoke_audit.csv`
