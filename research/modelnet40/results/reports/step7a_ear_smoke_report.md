# Step 7a — EAR Smoke Test Report

- Generated at: 2026-06-21 15:27:30 UTC
- Status: **PASS**

## EAR Configuration

- EAR code path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/EAR/ear_upsampling.py`
- Alternate HPC wrappers: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/code/upsampling_methods/EAR` (KITTI batch scripts; symlinks same core `ear_upsampling.py`)
- Input format: ModelNet40 `.npy` `(N, 3)` wrapped as `(N, 4)` with zero intensity for `ear_upsample_kitti`
- Output format: `.npy` `(1024, 3)` after EAR + resample normalization
- Native EAR I/O: KITTI `.bin` (N×4 float32); **no native `.npy`/`.xyz`** — wrapper converts in-memory
- Compilation: **None** (pure Python + numpy + scikit-learn)
- GPU required: **No** (numpy + scikit-learn only)
- Checkpoint required: **No** (deterministic edge-aware resampling)
- Upsampling control: `target_n` parameter in `ear_upsample_kitti()`; `up_factor` used when `target_n` is None
- Smoke executed: **Yes** (CPU on login node; `scikit-learn` installed via `pip install --user scikit-learn`)

### Line A Identity Behavior (Important)

When input already has 1024 points and `target_n=1024`, `ear_upsample_kitti()` returns an **identity copy** (no geometric refinement). Smoke audit confirms: all 80 original samples `input_points=1024`, `ear_raw_points=1024`. Line B (512→1024) performs real EAR upsampling.

## Smoke Results

- Original success count (audit): 80/80
- Downsampled50 success count (audit): 80/80
- Failed samples (audit): 0

### Original Output Check

- Output root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original_up/ear_smoke`
- File count: 80 (expected 80)
- Shape violations: 0
- NaN/Inf files: 0
- Split counts: {'train': 40, 'test': 40}
- Class dirs present: 40

### Downsampled50 Output Check

- Output root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled50_up/ear_smoke`
- File count: 80 (expected 80)
- Shape violations: 0
- NaN/Inf files: 0
- Split counts: {'train': 40, 'test': 40}
- Class dirs present: 40

## DataLoader Dry-Run

### original

- `train`: samples=40, batch_points=(8, 1024, 3), batch_labels=(8,), finite=True
- `test`: samples=40, batch_points=(8, 1024, 3), batch_labels=(8,), finite=True

### downsampled50

- `train`: samples=40, batch_points=(8, 1024, 3), batch_labels=(8,), finite=True
- `test`: samples=40, batch_points=(8, 1024, 3), batch_labels=(8,), finite=True

## Readiness

- Can proceed to EAR full upsampling: **Yes**
- Can proceed to PDANS smoke test: **Yes** (independent method; PDANS needs GPU)

## Full EAR Upsampling Estimate

- Hardware: CPU sufficient (no GPU required)
- Dataset size: 9843 train + 2468 test = 12311 samples per line
- Total full runs: 2 lines × 12311 ≈ 24622 EAR inferences
- Smoke wall time: ~492 s for 160 tasks (80 identity + 80 real upsample)
- Per-sample CPU time (512→1024): ~6 s/sample (from smoke downsampled50 branch)
- Line A full (original 1024): near-instant identity copies (~minutes total)
- Line B full (downsampled50 512→1024): ~12311 × 6 s ≈ **20.5 h** sequential; parallelize with chunk scripts
- Recommendation: submit CPU batch job on standard partition with 8–16 workers per chunk

## Artifacts

- Audit CSV: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/step7a_ear_smoke_audit.csv`
- Raw EAR outputs: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/outputs/ear_smoke/raw`
- Log: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/logs/ear_smoke/run_ear_smoke.log`
