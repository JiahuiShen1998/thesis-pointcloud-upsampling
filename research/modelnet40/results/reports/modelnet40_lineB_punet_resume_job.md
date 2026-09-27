# Line B PU-Net Resume Job

- Submitted at: 2026-07-05
- Job ID: **1731611**
- Job name: `lineB_x4_pu_net_resume`
- Script: `jobs/lineB_upsampling/run_lineB_punet_256to1024_resume_missing.sbatch`
- Python: `scripts/run_lineB_punet_resume_missing.py`
- Missing list: `reports/modelnet40_lineB_punet_missing_samples.txt` (3161 samples)
- Array: 0–7 (8 chunks, ~395 samples each)
- Time limit: 24:00:00 per chunk
- Skips existing valid strict `(1024,3)` outputs — preserves 9150 completed samples

## Monitor

```bash
watch -n 60 '
PROJECT=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

echo "===== PU-Net resume jobs ====="
squeue -u $USER | grep -Ei "pu_net|punet|lineB_x4" || true

echo ""
echo "===== STRICT_N COUNTS ====="
for m in ear pdans pu_net; do
  c=$(find "$PROJECT/datasets/lineB_downsampled_x4_up/strict_N/$m" -name "*.npy" 2>/dev/null | wc -l)
  echo "$m: $c / 12311"
done

echo ""
echo "===== RAW COUNTS ====="
for m in ear pdans pu_net; do
  c=$(find "$PROJECT/datasets/lineB_downsampled_x4_up/raw/$m" -name "*.npy" 2>/dev/null | wc -l)
  echo "$m raw: $c / 12311"
done
'
```

## Failed samples log

`reports/modelnet40_lineB_punet_resume_failed_samples.csv` (append on failure)

## Final status (post-resume)

- Checked at: 2026-07-05 16:49:29 UTC
- Job ID: **1731611**
- Array tasks: 8
- Overall: **COMPLETED**

| JobID | JobName | State | Elapsed | ExitCode |
| --- | --- | --- | --- | --- |
| 1731611_0 | lineB_x4_pu_net_resume | COMPLETED | 03:09:11 | 0:0 |
| 1731611_1 | lineB_x4_pu_net_resume | COMPLETED | 03:08:43 | 0:0 |
| 1731611_2 | lineB_x4_pu_net_resume | COMPLETED | 03:10:21 | 0:0 |
| 1731611_3 | lineB_x4_pu_net_resume | COMPLETED | 03:10:37 | 0:0 |
| 1731611_4 | lineB_x4_pu_net_resume | COMPLETED | 03:10:16 | 0:0 |
| 1731611_5 | lineB_x4_pu_net_resume | COMPLETED | 03:11:56 | 0:0 |
| 1731611_6 | lineB_x4_pu_net_resume | COMPLETED | 03:09:07 | 0:0 |
| 1731611_7 | lineB_x4_pu_net_resume | COMPLETED | 03:00:34 | 0:0 |
