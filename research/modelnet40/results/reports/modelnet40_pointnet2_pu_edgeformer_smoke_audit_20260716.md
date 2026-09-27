# ModelNet40 PointNet++ PU-EdgeFormer Smoke Audit

- Generated: `2026-07-16 19:27:51 UTC`
- Submit command: `sbatch.tinygpu`
- Job IDs: Line B=`1749430`, Line A=`1749431`
- Overall: **PASS** (2/2)

## Safety flags

- POINTNET_SMOKE_STARTED=YES
- POINTNET_FULL_TRAINING_STARTED=NO
- DETECTOR_EVAL_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO
- Full-training EdgeFormer sbatch present: NO
- Full-result dir Line B exists: False
- Full-result dir Line A exists: False
- Data modification: NO (symlink-only inputs; `.npy` untouched)

## Summary

| Line | expected pts | actual batch shape | first loss | train/test | status |
|---|---:|---|---:|---|---|
| B | 1024 | `(8, 3, 1024)` | 3.662344 | 9843/2468 | **PASS** |
| A | 4096 | `(8, 3, 4096)` | 3.735890 | 9843/2468 | **PASS** |

## Line B detail

- Input path: `pointnet2_inputs/lineB_downsampled_x4_up/pu_edgeformer`
- Symlink target: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineB_downsampled_x4_up/strict_N/pu_edgeformer`
- Config: `configs/pointnet2_x4_two_line/lineB_downsampled_x4_up_pu_edgeformer_1024.yaml`
- Smoke job: `jobs/pointnet2_smoke/smoke_lineB_pu_edgeformer_1024.sbatch` (job `1749430`, COMPLETED, exit 0:0)
- Expected points: **1024**
- First sample shape: `(1024, 3)`
- Actual batch shape: `(8, 3, 1024)` → points **1024**
- First loss: **3.662344** (finite=True)
- Train/test count: 9843/2468 (expected 9843/2468)
- Metadata: class maps OK=True; wrong manifest present=False
- Comparison primary: Downsampled x4 baseline 256
- Comparison secondary: Original baseline 1024
- No NaN loss / OOM / CUDA traceback: True / True / True
- Status: **PASS**
- Slurm log: `logs/pointnet2_smoke/lineB_pu_edgeformer_1024/slurm_1749430.out`
- Train log: `pointnet2_results/x4_two_line_smoke/lineB_downsampled_x4_up/pu_edgeformer/train.log`

## Line A detail

- Input path: `pointnet2_inputs/lineA_original_up/pu_edgeformer`
- Symlink target: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineA_original_up/strict_4N/pu_edgeformer`
- Config: `configs/pointnet2_x4_two_line/lineA_original_up_pu_edgeformer_4096.yaml`
- Smoke job: `jobs/pointnet2_smoke/smoke_lineA_pu_edgeformer_4096.sbatch` (job `1749431`, COMPLETED, exit 0:0)
- Expected points: **4096**
- First sample shape: `(4096, 3)`
- Actual batch shape: `(8, 3, 4096)` → points **4096**
- First loss: **3.735890** (finite=True)
- Train/test count: 9843/2468 (expected 9843/2468)
- Metadata: class maps OK=True; wrong manifest present=False
- Comparison primary: Original baseline 1024
- Comparison secondary: n/a
- No NaN loss / OOM / CUDA traceback: True / True / True
- Status: **PASS**
- Slurm log: `logs/pointnet2_smoke/lineA_pu_edgeformer_4096/slurm_1749431.out`
- Train log: `pointnet2_results/x4_two_line_smoke/lineA_original_up/pu_edgeformer/train.log`

## Checks

- dataloader reads train/test by directory scan: YES (no manifests)
- labels via class_to_idx.json: YES
- no wrong train/test_manifest.csv: YES
- no full training started: YES

## Conclusion

- Smoke overall: **PASS**
- Full training: still **NOT** started
