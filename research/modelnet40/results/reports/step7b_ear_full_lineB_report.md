# Step 7b — EAR Full Line B Upsampling Report

- Generated at: 2026-06-21 18:13:13 UTC
- Validation status: **PASS**

## Output Counts

- Train: 9843 (expected 9843)
- Test: 2468 (expected 2468)
- Total: 12311 (expected 12311)
- Class dirs: 40 (expected 40)

## Quality Checks

- Shape violations: 0
- NaN/Inf files: 0

## DataLoader Dry-Run

- `train`: samples=9843, batch_points=(24, 1024, 3), batch_labels=(24,), finite=True
- `test`: samples=2468, batch_points=(24, 1024, 3), batch_labels=(24,), finite=True

## Step 8 Readiness

- Ready for Downsampled50+EAR → PointNet++ training: **Yes**

- Output root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled50_up/ear`
- Audit CSV: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/step7b_ear_full_lineB_audit.csv`
