# ModelNet40 ×4 Final Protocol Report

- Generated at: 2026-07-05 (PU-Net resume complete + post-process)
- PROJECT_ROOT: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

## 1. Protocol overview

This experiment retains **two main classification lines** plus an independent **quality metrics module**.

| Line | Baseline | Upsampling branch |
| --- | --- | --- |
| **Line A** | Original baseline (1024 pts) | Original + Upsampling → 4096 pts |
| **Line B** | Downsampled ×4 baseline (256 pts) | Downsampled ×4 + Upsampling → 1024 pts |

CD / HD / P2F / NUC are **not** a third classification line. They evaluate Line B intermediate
geometry (upsampled point cloud vs original point cloud / CAD mesh) **before** PointNet++.

## 2. Point counts (detected from data)

- N (original baseline): **1024**
- N/4 (downsampled ×4): **256**
- 4N (Line A upsampling output): **4096**
- N (Line B upsampling output): **1024**

## 3. Downsampled ×4 database

- Path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4`
- Reused existing: **True**

Legacy `modelnet40_downsampled50` (512 pts, N/2) is preserved but **not** used as x4 downsample.

## 4. Line A method status

| method | train | test | expected | status |
| --- | ---: | ---: | ---: | --- |
| EAR | 9843 | 2468 | 4096 | reused_legacy |
| PU-Net | 9843 | 2468 | 4096 | reused_legacy |
| PU-GCN | 0 | 0 | 4096 | legacy_incomplete |
| PDANS | 9843 | 2468 | 4096 | reused_legacy |

## 5. Line B method status (updated 2026-07-05)

| method | strict train+test | expected pts | status |
| --- | ---: | ---: | --- |
| EAR | 12311 / 12311 | 1024 | **completed** — count audit PASS |
| PDANS | 12311 / 12311 | 1024 | **completed** — count audit PASS |
| PU-Net | 12311 / 12311 | 1024 | **completed** — resume job **1731611** COMPLETED; final count audit **PASS** |
| PU-GCN | 10012 / 12311 | 1024 | **resume running** (job **1732386**; retry after 1732373 SyntaxError fix) |

Legacy `downsampled50_up/*_x4` (512→2048) is **not** valid for this protocol.

## 5b. Line B 256→1024 current upsampling status and PU-Net diagnosis

1. **Downsampled ×4 reused** — `datasets/modelnet40_downsampled_x4/`, 256 pts, 12311 samples. No rebuild.
2. **EAR Line B** — 256 → EAR inference → raw → strict **1024**. 12311/12311 PASS.
3. **PDANS Line B** — 256 → PDANS inference → raw → strict **1024**. 12311/12311 PASS.
4. **PU-Net smoke** — job 1727073 FAILED (missing `strict_out` parent mkdir). Full 1727074 **canceled**.
5. **PU-Net fix** — mkdir fix applied; smoke **1729225 PASS** (10 samples, 256→1024).
6. **PU-Net full** — job **1729235** stopped at **9150/12311** (11× TIMEOUT @24h, 1× FAILED rc=-6).
7. **PU-Net resume** — job **1731611** submitted (3161 missing only, 8-chunk array).
8. **PU-GCN** — pending, not failed.
8. **Provenance audit** — EAR/PDANS: 20/20 spot checks PASS, no Line A / old protocol contamination.
9. **Shape audit** — EAR/PDANS: all 12311 exact 1024, zero NaN/Inf.
10. **PointNet++ inputs** — prepared for baseline + EAR + PDANS (`pointnet2_inputs/lineB_downsampled_x4_*`).
11. **Quality metrics** — CD/HD/NUC for EAR/PDANS vs original: job submitted (`run_lineB_completed_quality_metrics.sbatch`). P2F pending (meshes exist; full computation slow).
12. **Next steps (superseded)** — see §5e below.

## 5d. Line B PU-Net resume after partial completion (2026-07-05)

1. Job **1729235** produced **9150/12311** then stopped (no longer in `squeue`).
2. **Stop reason:** predominantly **TIMEOUT** (24h wall clock per chunk); chunk 5 **FAILED** (`PU-Net inference failed rc=-6` on `bookshelf_0424`).
3. Existing **9150** strict outputs: quick audit **PASS** — all `(1024,3)`, zero NaN/Inf (`reports/modelnet40_lineB_punet_existing_9150_quick_audit.csv`).
4. **Missing:** 3161 samples — `reports/modelnet40_lineB_punet_missing_samples.txt` / `.csv`.
5. **Resume sbatch:** `jobs/lineB_upsampling/run_lineB_punet_256to1024_resume_missing.sbatch`.
6. **Resume job ID:** **1731611** (`lineB_x4_pu_net_resume`, array 0–7).
7. EAR / PDANS remain **completed** (12311/12311). PU-GCN **pending**.
8. **Metrics job 1729250:** **COMPLETED** — EAR+PDANS CD/HD/NUC in `reports/modelnet40_lineB_completed_methods_quality_metrics_summary.csv`. PU-Net metrics pending until 12311 complete.
9. **Next (superseded):** see §5e below.

## 5e. Line B PU-Net completion post-process (2026-07-05)

1. **Original full job 1729235** stopped at **9150/12311** — predominantly **TIMEOUT** (24h/chunk); chunk 5 **FAILED** (`bookshelf_0424`, rc=-6).
2. **Resume job 1731611** (`lineB_x4_pu_net_resume`, array 0–7): **COMPLETED**, all tasks exit **0:0**, elapsed ~3h10m per chunk.
3. **PU-Net final count audit: PASS** — raw **12311/12311**, strict_N **12311/12311**, all `(1024,3)`, zero NaN/Inf, zero missing.
   - `reports/modelnet40_lineB_punet_final_count_audit.md`
4. **Line B completed methods:** EAR / PDANS / PU-Net = **completed**; PU-GCN = **pending**, intentionally skipped.
   - `reports/modelnet40_lineB_completed_methods_count_audit.md`
5. **PointNet++ inputs** prepared for baseline + EAR + PDANS + PU-Net (symlinks, 12311 samples each).
   - `reports/modelnet40_lineB_pointnet2_input_audit_completed_methods.md`
6. **Quality metrics**
   - EAR + PDANS: job **1729250** COMPLETED → `reports/modelnet40_lineB_completed_methods_quality_metrics_summary.csv`
   - PU-Net: job **1731753** **COMPLETED** (~18m) → `reports/modelnet40_lineB_punet_quality_metrics_summary.csv`; merged into `reports/modelnet40_lineB_completed_methods_quality_metrics_summary.csv` (4 comparison groups).
   - **P2F:** pending — original `.off` meshes exist in manifest, but full mesh-distance computation is slow.
7. **Next step:** Line B PointNet++ training (baseline + EAR + PDANS + PU-Net). PU-GCN remains separate/pending.

Audit reports:
- `reports/modelnet40_lineB_current_status_audit.md`
- `reports/modelnet40_lineB_ear_pdans_provenance_audit.md`
- `reports/modelnet40_lineB_ear_pdans_shape_audit.md`
- `reports/modelnet40_lineB_completed_methods_count_audit.md`
- `reports/modelnet40_lineB_punet_smoke_failure_diagnosis.md`
- `reports/modelnet40_lineB_punet_smoke_audit.md`
- `reports/modelnet40_lineB_pointnet2_input_audit_completed_methods.md`
- `reports/modelnet40_lineB_punet_final_count_audit.md`
- `reports/modelnet40_lineB_punet_resume_job.md`

## 5f. PU-GCN Line B resume after RTX 3080 stall (2026-07-06)

### Line A PU-GCN — completed

| metric | value |
| --- | --- |
| strict_4N | **12311 / 12311** |
| expected shape | **(4096, 3)** |
| status | **completed** (job 1731793) |

### Line B PU-GCN — original full job 1731794

| metric | value |
| --- | --- |
| raw / strict_N at stall | **10010 / 12311** |
| missing | **2301** |
| stalled tasks | 13 / 14 / 15 on **tg084 / RTX 3080** |
| error | `PU-GCN inference failed (rc=1)` at **`knn_point_2`** |
| recent output | zero new raw/strict files in 120+ min before cancel |
| existing 10010 audit | **PASS** — all `(1024,3)`, zero NaN/Inf |

### Handling

1. **`scancel 1731794`** — tasks 13/14/15 **CANCELLED** (not natural TIMEOUT).
2. **Retained** all 10010 existing raw/strict outputs (not deleted, not re-run).
3. Missing list regenerated: `reports/modelnet40_lineB_pugcn_missing_samples.txt` (**2301** samples).
4. **Resume job 1732373** submitted → **FAILED** (Python SyntaxError in runner; see `reports/modelnet40_lineB_pugcn_resume_1732373_failure_diagnosis.md`).
5. **Script fixed** + smoke **1732385 PASS** → **retry resume job 1732386** submitted (array 0–7, RTX 2080 Ti only).

### PU-GCN Line B final status (live)

| item | status |
| --- | --- |
| strict_N count | **running** — retry resume job **1732386** (1732373 failed: SyntaxError; fixed + smoke PASS) |
| final count audit | **pending** (after 12311/12311) |
| provenance audit | **pending** |
| quality metrics (CD/HD/NUC) | **pending** (submit after final audit PASS) |
| PointNet++ inputs | **pending** (after final audit PASS) |

### Other methods (unchanged)

| method | status |
| --- | --- |
| PU-EdgeFormer | **pending_checkpoint** — checkpoint not transferred |
| TULIP | **pending / not included this run** |
| P2F | **pending** — `.off` meshes exist; full mesh-distance computation slow |

**Note:** This is **not** dropping PU-GCN — resuming to full 12311/12311 completion.

Reports:
- `reports/modelnet40_lineB_pugcn_1731794_cancelled_for_resume.md`
- `reports/modelnet40_pugcn_lineB_stall_diagnosis.md`
- `reports/modelnet40_lineB_pugcn_resume_job.md`

## 5c. Line B method status (legacy snapshot 2026-07-02 — superseded)

| method | train | test | expected | status |
| --- | ---: | ---: | ---: | --- |
| EAR | 0 | 0 | 1024 | legacy_wrong_protocol_512_to_2048 |
| PU-Net | 0 | 0 | 1024 | legacy_wrong_protocol_512_to_2048 |
| PU-GCN | 0 | 0 | 1024 | pending_generation |
| PDANS | 0 | 0 | 1024 | legacy_wrong_protocol_512_to_2048 |

## 6. Quality metrics (Line B intermediate)

### Definitions

- **CD**: Chamfer distance (forward + backward mean squared min-distance)
- **HD**: Hausdorff distance (symmetric max-min distance)
- **P2F**: Mean/median/95th/max distance from upsampled points to original CAD .off mesh surface
- **NUC**: Neighborhood Uniformity Coefficient — CV of neighbor counts at radii (0.02, 0.05, 0.10); lower = more uniform

### Summary

| group | method | split | source_pts | target_pts | cd_mean | hd_mean | p2f_mean | nuc_mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| downsampled_x4_vs_original | downsampled_x4_baseline | train | 256 | 1024 | 0.003738522381653883 | 0.19684416923210604 | nan | 2.0418197641914477 |
| downsampled_x4_vs_original | downsampled_x4_baseline | test | 256 | 1024 | 0.003899775657808386 | 0.1985869943005798 | nan | 2.0979202574617526 |
| downsampled_x4_vs_original | downsampled_x4_baseline | all | 256 | 1024 | 0.0037708490070741834 | 0.19719355533144756 | nan | 2.0530662930998314 |

## 7. PointNet++ classification summary

| line | branch | method | input_pts | upsampling | accuracy | status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | baseline | original | 1024 | 1 | 0.9195 | completed |
| A | upsampling | EAR | 4096 | 4 | 91.4800 | completed |
| A | upsampling | PU-Net | 4096 | 4 |  | blocked |
| A | upsampling | PU-GCN | 4096 | 4 |  | blocked |
| A | upsampling | PDANS | 4096 | 4 | 91.6200 | completed |
| B | baseline | downsampled_x4 | 256 | 1 |  | pending |
| B | upsampling | EAR | 1024 | 4 |  | pending |
| B | upsampling | PU-Net | 1024 | 4 |  | pending |
| B | upsampling | PU-GCN | 1024 | 4 |  | pending |
| B | upsampling | PDANS | 1024 | 4 |  | pending |

## 8. Difference from old protocol

| Aspect | Old protocol | New protocol |
| --- | --- | --- |
| Line B downsample | N/2 = 512 | N/4 = 256 |
| Line B upsampling output | 2048 | 1024 |
| Line A upsampling output | 4096 | 4096 (unchanged) |
| Quality metrics | None | CD/HD/P2F/NUC on Line B intermediate |

## 11. PU-GCN x4 generation status (2026-07-05)

### Capability audit

- Path: `reports/modelnet40_pugcn_capability_audit.csv`, `reports/modelnet40_pugcn_capability_audit.md`
- **status: ready** — code, checkpoint (`pretrained/pu1k-pugcn`), conda env `tf15_upsampling` verified
- Env fixes: installed `plyfile`; `open3d` optional import for `.xyz` inference path

### Smoke result

- **PASS** (both lines) — see `reports/modelnet40_pugcn_smoke_audit.csv` / `.md`
- Line A smoke job **1731790**: 10/10 success (1024→4096)
- Line B smoke job **1731791**: 10/10 success (256→1024)

### Full generation (running)

| line | job ID | sbatch | target strict count | current status |
| --- | --- | --- | ---: | --- |
| Line A 1024→4096 | **1731793** | `jobs/pugcn_x4/run_lineA_pugcn_1024to4096.sbatch` | 12311 | **running** (array 0–15) |
| Line B 256→1024 | **1731794** → **1732373** (resume) | `run_lineB_pugcn_256to1024.sbatch` → `run_lineB_pugcn_resume_missing_2080ti.sbatch` | 12311 | **resume running** (1732373, array 0–7, RTX 2080 Ti only) |

Output paths:
- Line A raw: `datasets/lineA_original_up/raw/pu_gcn/`
- Line A strict: `datasets/lineA_original_up/strict_4N/pu_gcn/` (4096 pts)
- Line B raw: `datasets/lineB_downsampled_x4_up/raw/pu_gcn/`
- Line B strict: `datasets/lineB_downsampled_x4_up/strict_N/pu_gcn/` (1024 pts)

Runner: `scripts/run_pugcn_modelnet40_x4.py` (real PU-GCN inference via TF1 wrapper)

### Post-full audits (pending completion)

- Final count audit: `reports/modelnet40_pugcn_final_count_audit.csv` (run after 12311/12311)
- Provenance audit: `reports/modelnet40_pugcn_provenance_audit.csv` (20 samples/line spot check)
- Line B quality metrics: `scripts/compute_lineB_pugcn_quality_metrics.py` → merge into `reports/modelnet40_lineB_completed_methods_quality_metrics_summary.csv`
- PointNet++ inputs: `scripts/prepare_pugcn_pointnet2_inputs.py` → `reports/modelnet40_pointnet2_input_audit.csv`
- Post-process sbatch (submit after final audit PASS): `jobs/pugcn_x4/run_lineB_pugcn_postprocess.sbatch`

### PU-EdgeFormer / TULIP (not in this run)

| method | status | note |
| --- | --- | --- |
| PU-EdgeFormer | **pending_checkpoint** | checkpoint not transferred yet; not included in this run |
| TULIP | **pending / not included** | not touched in this run |

### Completed / pending method sets

**Completed (Line B upsampling + metrics):** EAR, PDANS, PU-Net

**In progress:** PU-GCN Line B resume (job **1732373**); Line A PU-GCN **completed**

**Pending:** PU-EdgeFormer (pending_checkpoint), TULIP (not included)

### Monitoring

```bash
watch -n 60 '
PROJECT=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

echo "===== PU-GCN jobs ====="
squeue -u $USER | grep -Ei "pugcn|pu_gcn|mn40_A_pugcn|mn40_B_pugcn" || true

echo ""
echo "===== Line A PU-GCN counts ====="
find "$PROJECT/datasets/lineA_original_up/strict_4N/pu_gcn" -name "*.npy" 2>/dev/null | wc -l

echo ""
echo "===== Line B PU-GCN counts ====="
find "$PROJECT/datasets/lineB_downsampled_x4_up/strict_N/pu_gcn" -name "*.npy" 2>/dev/null | wc -l
'
```

## 9. Failures / missing (updated 2026-07-05)

- Line A PU-GCN: **completed** (12311/12311, job 1731793)
- Line B PU-GCN: **resume running** (job **1732386**; 1732373 failed SyntaxError, fixed; 1731794 cancelled after RTX 3080 stall)
- Line B PU-Net: **completed** (12311/12311; resume 1731611)
- Line B PointNet++ training: **next** — inputs ready for baseline + EAR + PDANS + PU-Net
- P2F for upsampled methods: **pending** — `.off` meshes exist; full mesh-distance computation slow
- PU-Net quality metrics job **1731753**: **COMPLETED** (CD/HD/NUC vs original, 12311 samples)

## 10. Thesis conclusion draft

We evaluate point cloud upsampling on ModelNet40 with two parallel lines. Line A compares native 1024-point inputs against 4096-point upsampled clouds. Line B first downsamples to 256 points (×4 reduction), then upsamples back to 1024 points. Geometric fidelity of Line B upsampling is quantified independently via Chamfer/Hausdorff distances, point-to-surface error against CAD meshes, and neighborhood uniformity (NUC). Classification with PointNet++ is reported separately for each line and method.

## File index

- Data audit: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_data_audit.md`
- Line A count audit: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_lineA_original_up_count_audit.md`
- Line B count audit: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_lineB_downsampled_x4_up_count_audit.md`
- Quality per-sample: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_lineB_quality_metrics_per_sample.csv`
- Quality summary: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_lineB_quality_metrics_summary.csv`
- Classification summary: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_classification_summary.csv`
- PointNet++ inputs: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_inputs`
- PointNet++ results: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results`
## ModelNet40 geometric quality metrics

- Updated at: 2026-07-06 22:04:21 UTC

### Primary comparison

**Original baseline (1024 pts)** vs **Downsampled ×4 + Upsampling (1024 pts)**
All metrics computed against dense GT (10000 surface points from `.off` mesh).

### Main groups

| Method | CD total | HD total | NUC | P2F |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 0.049548 | 0.119720 | 1.061943 | 0.08246926348028343 |
| Downsampled x4 + EAR | 0.081356 | 0.208793 | 0.957795 | 0.08115380757410291 |
| Downsampled x4 + PDANS | 0.063137 | 0.205927 | 1.089924 | 0.07958593782259225 |
| Downsampled x4 + PU-Net | 0.057115 | 0.137674 | 1.058212 | 0.08006658204365182 |
| Downsampled x4 + PU-GCN | 0.054827 | 0.165212 | 1.486469 | 0.08377861363436016 |

### Supplementary groups

- Downsampled ×4 baseline (256 pts)
- Original + Upsampling methods (4096 pts): EAR, PDANS, PU-Net, PU-GCN

### Metric definitions

- **CD/HD**: source vs dense mesh-sampled GT (L2, lower better)
- **P2F**: source vs original `.off` mesh surface (exact)
- **NUC**: local density uniformity (lower better)

### GT reference

`quality_metrics/gt_reference_points_10000/` — triangle-area weighted surface sampling, seed=42

### Output files

- `reports/modelnet40_main_quality_metrics_summary.csv`
- `reports/modelnet40_original_vs_downsampled_up_metrics_summary.csv`
- `reports/modelnet40_quality_metrics_extended_comparison_summary.csv`
- `reports/modelnet40_quality_metrics_definition.md`

### Supplementary summary snapshot

| group | CD | HD | NUC |
| --- | ---: | ---: | ---: |
| downsampled_x4_baseline_vs_gt | 0.077123 | 0.214251 | 2.049289 |
| original_up_ear_vs_gt | 0.049197 | 0.117408 | 0.806502 |
| original_up_pdans_vs_gt | 0.041018 | 0.115437 | 0.646813 |
| original_up_pu_gcn_vs_gt | 0.036304 | 0.090933 | 0.551304 |
| original_up_pu_net_vs_gt | 0.044525 | 0.114043 | 0.591942 |

## ModelNet40 geometric quality metrics completed

- Updated at: 2026-07-07 07:03:13 UTC
- Coverage: **12311 samples × 10 groups**
- Metrics: **CD / HD / NUC / exact P2F completed**
- Main CSV: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_original_vs_downsampled_up_metrics_summary.csv`
- Full run log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/quality_metrics/full_run.log`

### Main observation

In the main Line B comparison, **PU-GCN** and **PU-Net** are the closest to the Original baseline on **CD**, but the final interpretation should also consider **HD**, **exact P2F**, and **NUC** rather than CD alone.

### Next step

Prepare and run **PointNet++ classification training** for both Line A and Line B branches, then compare downstream accuracy against the geometric quality tables.

<!-- POINTNET2_TRAINING_PREPARATION_START -->
## PointNet++ training preparation

- Updated at: 2026-07-07 07:29:07 UTC
- 10 branches ready: **10 / 10**
- Input audit status: see `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_training_input_audit.md`
- Input symlink audit: see `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_input_symlink_audit.md`
- Dataloader pointcount audit status: see `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_dataloader_pointcount_audit.md`
- Smoke plan path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_smoke_plan.md`
- Full training job plan path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_full_training_job_plan.md`
- Executable checklist path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_executable_training_checklist.md`
- Next step: submit smoke jobs.
<!-- POINTNET2_TRAINING_PREPARATION_END -->

<!-- POINTNET2_LINE_B_SMOKE_START -->
## PointNet++ Line B smoke test

- Updated at: 2026-07-07 12:27:00 UTC
- Submit time: 2026-07-07 12:25:06 UTC (`sbatch.tinygpu`)
- Job ID report: `reports/modelnet40_pointnet2_lineB_smoke_job_ids.md`
- Smoke audit: `reports/modelnet40_pointnet2_lineB_smoke_audit.md` / `.csv`
- Failure diagnosis: `reports/modelnet40_pointnet2_lineB_smoke_failure_diagnosis.md`

### Submitted branches

| branch | job id | expected pts | actual pts | status |
| --- | ---: | ---: | ---: | --- |
| lineB_downsampled_x4_baseline_256 | 1733159 | 256 | 256 | **PASS** |
| lineB_ear_1024 | 1733160 | 1024 | — | FAIL |
| lineB_pdans_1024 | 1733161 | 1024 | — | FAIL |
| lineB_punet_1024 | 1733162 | 1024 | — | FAIL |
| lineB_pugcn_1024 | 1733163 | 1024 | — | FAIL |

### Result

- **1 / 5 smoke PASS** (baseline only).
- Four upsampling branches failed at dataloader init: missing `metadata/class_to_idx.json` under `strict_N/{ear,pdans,pu_net,pu_gcn}`.
- Baseline smoke verified: `num_point=256`, batch shape `(8,3,256)`, finite loss, forward/backward/eval OK.
- **Full training blocked until smoke issue is fixed.**

### Next step

Fix PointNet++ input metadata for Line B upsampling branches, re-run 4 failed smokes, then consider Line B full training submission.
<!-- POINTNET2_LINE_B_SMOKE_END -->

<!-- POINTNET2_LINE_B_SMOKE_METADATA_FIX_START -->
## PointNet++ Line B smoke test metadata fix and retry

- Updated at: 2026-07-07 12:41:00 UTC

### Round 1 (initial smoke)

- Baseline **PASS** (job 1733159, 256 pts)
- 4 upsampling branches **FAIL** — missing `metadata/class_to_idx.json`

### Metadata fix

- Metadata source audit: `reports/modelnet40_pointnet2_metadata_source_audit.md`
- Metadata fix audit: `reports/modelnet40_pointnet2_lineB_metadata_fix_audit.csv` / `.md`
- **v1:** full metadata dir symlink → retry round 1 FAILED (manifests pointed to 256-pt baseline)
- **v2:** `class_to_idx.json` + `idx_to_class.json` only; no manifests → directory scan → **PASS**

### Retry round 2 job IDs

| branch | retry job id |
| --- | ---: |
| lineB_ear_1024 | 1733171 |
| lineB_pdans_1024 | 1733172 |
| lineB_punet_1024 | 1733173 |
| lineB_pugcn_1024 | 1733174 |

- Retry job IDs: `reports/modelnet40_pointnet2_lineB_smoke_retry_job_ids.md`
- Retry smoke audit: `reports/modelnet40_pointnet2_lineB_smoke_retry_audit.csv` / `.md`
- Final Line B smoke status: `reports/modelnet40_pointnet2_lineB_smoke_final_status.csv` / `.md`

### Final Line B smoke status (5 branches)

| branch | job id | expected | actual | status |
| --- | ---: | ---: | ---: | --- |
| lineB_downsampled_x4_baseline_256 | 1733159 | 256 | 256 | PASS |
| lineB_ear_1024 | 1733171 | 1024 | 1024 | PASS |
| lineB_pdans_1024 | 1733172 | 1024 | 1024 | PASS |
| lineB_punet_1024 | 1733173 | 1024 | 1024 | PASS |
| lineB_pugcn_1024 | 1733174 | 1024 | 1024 | PASS |

**LINE_B_SMOKE_ALL_PASS** — Line B full training is ready to submit.

Ready commands: `reports/modelnet40_pointnet2_lineB_full_training_ready_commands.md`
<!-- POINTNET2_LINE_B_SMOKE_METADATA_FIX_END -->

<!-- POINTNET2_LINE_B_FULL_TRAINING_START -->
## PointNet++ Line B full training

- Updated at: 2026-07-07 19:50:00 UTC
- Submit time: 2026-07-07 13:04:28 UTC (`sbatch.tinygpu`)
- Job IDs: 1733191, 1733192, 1733193, 1733194, 1733195
- **Line B full training completed — 5 / 5 PASS**

### Full training job IDs

| branch | job id | elapsed | best overall | status |
| --- | ---: | --- | ---: | --- |
| lineB_downsampled_x4_baseline_256 | 1733191 | 06:23:27 | 90.85% | PASS |
| lineB_ear_1024 | 1733192 | 06:42:40 | 88.81% | PASS |
| lineB_pdans_1024 | 1733193 | 05:33:23 | 90.34% | PASS |
| lineB_punet_1024 | 1733194 | 05:21:51 | **91.27%** | PASS |
| lineB_pugcn_1024 | 1733195 | 05:22:52 | 90.06% | PASS |

### Reports

- Job IDs: `reports/modelnet40_pointnet2_lineB_full_training_job_ids.md`
- Full training audit: `reports/modelnet40_pointnet2_lineB_full_training_audit.csv` / `.md`
- Classification summary: `reports/modelnet40_pointnet2_lineB_classification_summary.csv` / `.md`
- Geometry vs classification: `reports/modelnet40_lineB_geometry_vs_classification_summary.csv` / `.md`
- Result interpretation: `reports/modelnet40_pointnet2_lineB_result_interpretation.md`
- Next steps: `reports/modelnet40_pointnet2_next_after_lineB_plan.md`

### Key finding

PU-Net is the only upsampling method that slightly exceeds the downsampled ×4 baseline on classification accuracy. Geometry metrics and classification do not show a simple monotonic relationship.

### Next step

Proceed to **Line A smoke jobs**, then Line A full training after smoke PASS.
<!-- POINTNET2_LINE_B_FULL_TRAINING_END -->

<!-- POINTNET2_LINE_A_SMOKE_START -->
## PointNet++ Line A smoke test

- Updated at: 2026-07-07 20:12:00 UTC
- Submit time: 2026-07-07 20:10:18 UTC (`sbatch.tinygpu`)
- Job IDs: 1733475, 1733476, 1733477, 1733478, 1733479
- Job ID report: `reports/modelnet40_pointnet2_lineA_smoke_job_ids.md`
- Smoke audit: `reports/modelnet40_pointnet2_lineA_smoke_audit.csv` / `.md`
- Failure diagnosis: `reports/modelnet40_pointnet2_lineA_smoke_failure_diagnosis.md`

### Smoke results

| branch | job id | expected | actual | status |
| --- | ---: | ---: | ---: | --- |
| lineA_original_baseline_1024 | 1733475 | 1024 | 1024 | PASS |
| lineA_ear_4096 | 1733476 | 4096 | — | FAIL |
| lineA_pdans_4096 | 1733477 | 4096 | — | FAIL |
| lineA_punet_4096 | 1733478 | 4096 | — | FAIL |
| lineA_pugcn_4096 | 1733479 | 4096 | — | FAIL |

- **1 / 5 smoke PASS** (baseline only).
- Baseline smoke verified: `num_point=1024`, batch shape `(8,3,1024)`, finite loss, forward/backward/eval OK.
- 4 upsampling branches **FAIL** — missing `metadata/class_to_idx.json` under `pointnet2_inputs/lineA_original_up/<method>/`.
- Same root cause as Line B smoke round 1 (fixed for Line B via metadata symlinks).
- **Line A full training blocked until metadata fix and smoke retry.**

### Next step

Apply Line B v2 metadata fix pattern to Line A upsampling inputs (symlink `class_to_idx.json` + `idx_to_class.json` from `modelnet40_original/metadata`), re-run 4 failed smokes, then consider Line A full training.
<!-- POINTNET2_LINE_A_SMOKE_END -->

<!-- POINTNET2_LINE_A_SMOKE_METADATA_FIX_START -->
## PointNet++ Line A smoke test metadata fix and retry

- Updated at: 2026-07-07 20:21:00 UTC

### Round 1 (initial smoke)

- Baseline **PASS** (job 1733475, 1024 pts)
- 4 upsampling branches **FAIL** — missing `metadata/class_to_idx.json`

### Metadata fix (v2)

- Metadata source audit: `reports/modelnet40_pointnet2_lineA_metadata_source_audit.md`
- Metadata fix audit: `reports/modelnet40_pointnet2_lineA_metadata_fix_audit.csv` / `.md`
- **v2:** per-method `metadata/` with only `class_to_idx.json` + `idx_to_class.json` symlinks from `modelnet40_original/metadata`; no manifests → directory scan → **PASS**

### Retry job IDs

| branch | old job id | retry job id |
| --- | ---: | ---: |
| lineA_ear_4096 | 1733476 | 1733487 |
| lineA_pdans_4096 | 1733477 | 1733488 |
| lineA_punet_4096 | 1733478 | 1733489 |
| lineA_pugcn_4096 | 1733479 | 1733490 |

- Retry job IDs: `reports/modelnet40_pointnet2_lineA_smoke_retry_job_ids.md`
- Retry smoke audit: `reports/modelnet40_pointnet2_lineA_smoke_retry_audit.csv` / `.md`
- Final Line A smoke status: `reports/modelnet40_pointnet2_lineA_smoke_final_status.csv` / `.md`

### Final Line A smoke status (5 branches)

| branch | job id | expected | actual | status |
| --- | ---: | ---: | ---: | --- |
| lineA_original_baseline_1024 | 1733475 | 1024 | 1024 | PASS |
| lineA_ear_4096 | 1733487 | 4096 | 4096 | PASS |
| lineA_pdans_4096 | 1733488 | 4096 | 4096 | PASS |
| lineA_punet_4096 | 1733489 | 4096 | 4096 | PASS |
| lineA_pugcn_4096 | 1733490 | 4096 | 4096 | PASS |

**LINE_A_SMOKE_ALL_PASS** — Line A full training is ready to submit.

Ready commands: `reports/modelnet40_pointnet2_lineA_full_training_ready_commands.md`
<!-- POINTNET2_LINE_A_SMOKE_METADATA_FIX_END -->

