# ModelNet40 x4 Two-Line Protocol — Final Report

Last updated: 2026-07-07

## PU-GCN x4 generation status

### 1. Capability audit

- Result: **ready** (PU-GCN code, checkpoint, GPU, wrapper present)
- Report: `reports/modelnet40_pugcn_capability_audit.csv`
- Report: `reports/modelnet40_pugcn_capability_audit.md`
- Code: `/home/ra87racy/projects/baseline_detectors/PointRCNN/external/PU-GCN`
- Checkpoint: `.../pretrained/pu1k-pugcn` (model-100)
- Environment: `conda env pugcn` (TensorFlow 1.x, GPU available)

**HPC note:** Canonical `PROJECT_ROOT` is `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`. On agent host `lms41-24` this path is **not mounted**; local staging root used for smoke validation: `/home/ra87racy/projects/modelnet40_pointnet2_upsampling`.

### 2. Smoke result

- Result: **PASS** (local staging, 8 samples per line)
- Report: `reports/modelnet40_pugcn_smoke_audit.csv`
- Report: `reports/modelnet40_pugcn_smoke_audit.md`
- Line A: 8/8 PASS — input (1024,3), strict (4096,3), no NaN/Inf
- Line B: 8/8 PASS — input (256,3), strict (1024,3), no NaN/Inf
- Inference: real PU-GCN via `scripts/run_pugcn_modelnet40_x4.py`

### 3. Line A PU-GCN 1024→4096 status

- Full generation: **pending HPC submission**
- Local smoke strict count: **8 / 12311**
- Output paths (protocol):
  - raw: `datasets/lineA_original_up/raw/pu_gcn/`
  - strict: `datasets/lineA_original_up/strict_4N/pu_gcn/`
- Job script (ready): `jobs/pugcn_x4/run_lineA_pugcn_1024to4096.sbatch`
- Submit on HPC: `sbatch.tinygpu jobs/pugcn_x4/run_lineA_pugcn_1024to4096.sbatch`

### 4. Line B PU-GCN 256→1024 status

- Full generation: **pending HPC submission**
- Local smoke strict count: **8 / 12311**
- Output paths (protocol):
  - raw: `datasets/lineB_downsampled_x4_up/raw/pu_gcn/`
  - strict: `datasets/lineB_downsampled_x4_up/strict_N/pu_gcn/`
- Job script (ready): `jobs/pugcn_x4/run_lineB_pugcn_256to1024.sbatch`
- Submit on HPC: `sbatch.tinygpu jobs/pugcn_x4/run_lineB_pugcn_256to1024.sbatch`

### 5. Final count audit

- Path: `reports/modelnet40_pugcn_final_count_audit.csv`
- Path: `reports/modelnet40_pugcn_final_count_audit.md`
- Status: **FAIL** (expected until full 12311/12311 on HPC)

### 6. Provenance audit

- Path: `reports/modelnet40_pugcn_provenance_audit.csv`
- Path: `reports/modelnet40_pugcn_provenance_audit.md`
- Checked on smoke outputs; no evidence of EAR/PDANS/PU-Net reuse or Line A→Line B cross-contamination

### 7. Quality metrics status

- **Not started** — requires Line B PU-GCN final audit PASS (12311/12311)
- Planned outputs:
  - `reports/modelnet40_lineB_pugcn_quality_metrics_per_sample.csv`
  - `reports/modelnet40_lineB_pugcn_quality_metrics_summary.csv`
  - merge into `reports/modelnet40_lineB_completed_methods_quality_metrics_summary.csv`
- P2F: pending because original `.off` meshes exist but full mesh-distance computation is slow

### 8. PointNet++ input status

- **Not started** — requires PU-GCN final audit PASS
- Planned targets:
  - `pointnet2_inputs/lineA_original_up/pu_gcn/` (4096 pts)
  - `pointnet2_inputs/lineB_downsampled_x4_up/pu_gcn/` (1024 pts)

### 9. PU-EdgeFormer

- Status: **pending_checkpoint**
- Checkpoint not transferred yet; not included in this run

### 10. TULIP

- Status: **pending / not included in this run**
- Not touched in this run

### 11. Completed method set (Line B upsampling)

- EAR (12311/12311) — pre-existing on HPC
- PDANS (12311/12311) — pre-existing on HPC
- PU-Net (12311/12311) — pre-existing on HPC
- PU-GCN — smoke PASS; full generation pending

### 12. Pending method set

- PU-EdgeFormer (`pending_checkpoint`)
- TULIP (`pending / not included`)

---

## Monitoring (HPC)

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

## HPC deployment checklist

1. Copy/sync `scripts/`, `jobs/pugcn_x4/` to woody `PROJECT_ROOT`
2. Confirm on HPC: `datasets/modelnet40_original_1024/`, `datasets/modelnet40_downsampled_x4/` (12311 each)
3. Set env if paths differ: `PUGCN_REPO`, `PUGCN_CKPT`, `PUGCN_PYTHON`, `MODELNET40_PROJECT_ROOT`
4. Re-run smoke on HPC: `python scripts/pugcn_smoke_test.py --max-samples 10`
5. Submit full jobs after smoke PASS
6. After 12311/12311: run `scripts/pugcn_final_count_audit.py`, `scripts/pugcn_provenance_audit.py`, quality metrics, PointNet++ input prep

## PointNet++ Line B smoke test

- Date attempted: **2026-07-07**
- Branches targeted (5):
  1. `lineB_downsampled_x4_baseline_256` — expected **256** pts
  2. `lineB_ear_1024` — expected **1024** pts
  3. `lineB_pdans_1024` — expected **1024** pts
  4. `lineB_punet_1024` — expected **1024** pts
  5. `lineB_pugcn_1024` — expected **1024** pts
- Job IDs: **NOT_SUBMITTED** (see `reports/modelnet40_pointnet2_lineB_smoke_job_ids.md`)
- Smoke audit:
  - `reports/modelnet40_pointnet2_lineB_smoke_audit.csv`
  - `reports/modelnet40_pointnet2_lineB_smoke_audit.md`
- Actual point counts: **N/A** (jobs not run)
- Smoke PASS / FAIL: **0/5 PASS — 5/5 BLOCKED**
- Failure diagnosis: `reports/modelnet40_pointnet2_lineB_smoke_failure_diagnosis.md`
- **Full training blocked until smoke issue is fixed** (submit from HPC login node)

## Files created this run (local staging)

- `scripts/run_pugcn_modelnet40_x4.py`
- `scripts/pugcn_capability_audit.py`
- `scripts/pugcn_smoke_test.py`
- `scripts/pugcn_final_count_audit.py`
- `scripts/pugcn_provenance_audit.py`
- `scripts/prepare_pugcn_smoke_inputs.py` (local smoke helper only)
- `jobs/pugcn_x4/run_lineA_pugcn_1024to4096.sbatch`
- `jobs/pugcn_x4/run_lineB_pugcn_256to1024.sbatch`
- `reports/modelnet40_pugcn_*` audit artifacts
- `reports/modelnet40_x4_final_protocol_report.md` (this file)
