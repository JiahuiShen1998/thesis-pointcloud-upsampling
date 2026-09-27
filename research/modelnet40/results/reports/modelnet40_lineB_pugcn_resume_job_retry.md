# PU-GCN Line B Resume Job — Retry After SyntaxError Fix

**Submit time:** 2026-07-06 19:56 CEST

## Previous failed job

| Field | Value |
| --- | --- |
| Job ID | **1732373** |
| Failure | **Python SyntaxError** (not GPU / knn_point_2) |
| Diagnosis | `reports/modelnet40_lineB_pugcn_resume_1732373_failure_diagnosis.md` |

## Fix

| Item | Value |
| --- | --- |
| Script | `scripts/run_lineB_pugcn_resume_missing.py` |
| Issue | `global EXPECTED_INPUT_POINTS` after argparse default usage |
| Fix | Local variables + pass to `process_row()`; removed `global` |
| py_compile | **PASS** |
| Smoke job | **1732385** — **PASS** (2/2 samples, tg06a RTX 2080 Ti) |
| Smoke report | `reports/modelnet40_lineB_pugcn_resume_script_fix_smoke.md` |

## New resume job

| Field | Value |
| --- | --- |
| Job ID | **1732386** |
| Job name | `mn40_B_pugcn_resume` |
| sbatch | `jobs/pugcn_x4/run_lineB_pugcn_resume_missing_2080ti.sbatch` |
| Array | 0–7 |
| Missing list | `reports/modelnet40_lineB_pugcn_missing_samples.txt` |
| Missing count (list file) | **2301** (2 completed in smoke → **2299** remaining) |
| Existing retained | **10010** + 2 smoke = **10012** strict at submit |
| Target | raw / strict **12311 / 12311** |

## Node constraints (unchanged)

- `#SBATCH --gres=gpu:rtx2080ti:1`
- `#SBATCH --exclude=tg080,tg081,tg082,tg083,tg084,tg085`

## Logs

- `logs/pugcn_x4/lineB_pugcn_resume_1732386_<chunk>.out`

## Monitoring

```bash
watch -n 60 '
PROJECT=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

echo "===== PU-GCN resume jobs ====="
squeue.tinygpu -u $USER | grep -Ei "pugcn|pu_gcn|mn40_B_pugcn_resume" || true

echo ""
echo "===== PU-GCN Line B counts ====="
echo -n "raw: "
find "$PROJECT/datasets/lineB_downsampled_x4_up/raw/pu_gcn" -name "*.npy" 2>/dev/null | wc -l

echo -n "strict: "
find "$PROJECT/datasets/lineB_downsampled_x4_up/strict_N/pu_gcn" -name "*.npy" 2>/dev/null | wc -l

echo ""
echo "===== recent strict writes last 60 min ====="
find "$PROJECT/datasets/lineB_downsampled_x4_up/strict_N/pu_gcn" -name "*.npy" -mmin -60 2>/dev/null | wc -l
'
```
