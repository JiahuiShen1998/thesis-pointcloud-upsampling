# PU-GCN Line B Resume Job

| Field | Value |
| --- | --- |
| Submit time | 2026-07-06 19:43 CEST |
| Resume job ID | **1732373** |
| Job name | `mn40_B_pugcn_resume` |
| sbatch path | `jobs/pugcn_x4/run_lineB_pugcn_resume_missing_2080ti.sbatch` |
| Runner | `scripts/run_lineB_pugcn_resume_missing.py` |
| Array | 0–7 (8 chunks) |
| Missing samples | **2301** |
| Existing strict (retained) | **10010** |

## Node / GPU constraints

- `#SBATCH --gres=gpu:rtx2080ti:1`
- `#SBATCH --exclude=tg080,tg081,tg082,tg083,tg084,tg085` (all RTX 3080 nodes)

## Reason

Previous full job **1731794** stalled on **tg084 / RTX 3080** — 100% inference failure at `knn_point_2`. Tasks 13/14/15 cancelled; existing 10010 outputs retained.

## Target

Line B PU-GCN:

- raw = **12311 / 12311**
- strict_N = **12311 / 12311**
- expected shape = **(1024, 3)**

## Missing list

- `reports/modelnet40_lineB_pugcn_missing_samples.txt`
- `reports/modelnet40_lineB_pugcn_missing_samples.csv`

## Failed samples (per chunk)

- `reports/modelnet40_lineB_pugcn_resume_failed_samples_<chunk_id>.csv`

## Logs

- `logs/pugcn_x4/lineB_pugcn_resume_1732373_<chunk>.out`
- `logs/pugcn_x4/lineB_pugcn_resume_1732373_<chunk>.err`

## Post-completion (pending)

| Step | Status |
| --- | --- |
| Final count audit | pending (after 12311/12311) |
| Provenance audit | pending |
| Quality metrics (CD/HD/NUC) | pending |
| PointNet++ input prep | pending |
| Final report update | partial (see `modelnet40_x4_final_protocol_report.md` §5f) |

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
