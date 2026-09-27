# ModelNet40 ×4 Parallel Generation Status

- Updated: 2026-07-01 17:45 CEST
- Project: `modelnet40_pointnet2_upsampling`

---

## EAR ×4 — **COMPLETE**

| Stage | Status |
| --- | --- |
| Generation audit | **PASS** |
| Method provenance | **PASS** — `method_confirmed` |
| PointNet++ training | **COMPLETED** (200/200) |

| Experiment | Best acc | vs baseline |
| --- | ---: | ---: |
| original_ear_x4_pointnet2 | 91.48% | -0.47 pp vs 91.95% |
| downsampled50_ear_x4_pointnet2 | 90.61% | -0.65 pp vs 91.26% |

---

## PDANS ×4 — **COMPLETE**

| Stage | Status |
| --- | --- |
| GPU validate | **PASS** |
| Smoke (40+40/line) | **PASS** |
| Full generation | **PASS** — 12311/line |
| Full audit | **PASS** |
| Method provenance | **PASS** — `method_confirmed` |
| Output similarity sanity | **PASS** |
| PointNet++ dry-run | **PASS** |
| PointNet++ training | **COMPLETED** (200/200) |

| Experiment | Best acc | vs baseline |
| --- | ---: | ---: |
| original_pdans_x4_pointnet2 | 91.62% | -0.33 pp vs 91.95% |
| downsampled50_pdans_x4_pointnet2 | 91.47% | +0.21 pp vs 91.26% |

---

## PU-Net ×4 — **IN PROGRESS (full generation running)**

| Stage | Status |
| --- | --- |
| TF custom ops validate | **PASS** (job 1725557) |
| Smoke (40+40/line) | **PASS** (jobs 1725581, 1725582) |
| Smoke audit | **PASS** |
| Full generation | **RUNNING** — jobs **1725588** (A), **1725589** (B), array 0–15 |
| Full generation progress | Line A ~982/12311, Line B ~542/12311 (partial) |
| Full audit | pending |
| Method provenance | pending (after full audit) |
| Output similarity sanity | pending |
| PointNet++ dry-run | pending |
| PointNet++ training | **not submitted** (blocked) |

### PU-Net env fixes applied

1. TF ops: CUDA10 runtime compat, cudnn/cudatoolkit in `tf15_upsampling`
2. `opencv-python-headless` for PU-Net `data_provider`
3. `main.py` test phase: GPU enabled + weight-only checkpoint restore
4. `punet_modelnet40_utils.py`: subprocess `LD_LIBRARY_PATH` / `PYTHONPATH`

---

## PU-GCN ×4 — **PENDING (after PU-Net)**

TF ops validate **PASS** (compiled with PU-Net in job 1725557). Smoke/full not started.

---

## TULIP

Supplementary only — not in main protocol.
