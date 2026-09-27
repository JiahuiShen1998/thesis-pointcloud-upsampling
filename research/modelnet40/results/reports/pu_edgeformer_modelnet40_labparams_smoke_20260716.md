# PU-EdgeFormer ModelNet40 Lab-Params Smoke Report

- Generated: `2026-07-16 18:02:08 UTC`
- project path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`
- package path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/transfer_pu_edgeformer_to_hpc_20260716.tar.gz`
- package SHA256: `2da8f2811b5529dc1c9de92d92372a505da5702af54429e41546a5bb31110d84`
- checkpoint used: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/external/pu_edgeformer_ops_reuse/checkpoint_model100` (model-100; HPC-relative checkpoint pointer)
- config used: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/external/pu_edgeformer_ops_reuse/checkpoint_model100/args.txt` + `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/imports/transfer_pu_edgeformer_to_hpc_20260716/config/configs.py`
- wrapper used: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/imports/transfer_pu_edgeformer_to_hpc_20260716/wrappers/tf_pugcn_family_patch_infer_many.py`
- repo_dir (code_snapshot + PU-GCN tf_ops reuse): `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/external/pu_edgeformer_ops_reuse`
- environment used: `tf15_upsampling` / `/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/bin/python`
- python: `3.7.12 | packaged by conda-forge | (default, Oct 26 2021, 06:08:21)  [GCC 9.4.0]`
- tensorflow: `1.15.0`
- gpu: `tf.test.is_gpu_available=True`
- inference seed: `20260702`

## Dataset audit (read-only)

- original 1024 train: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original/train` count=9843
- original 1024 test: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original/test` count=2468
- downsampled-x4 256 train: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4/train` count=9843
- downsampled-x4 256 test: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4/test` count=2468

## Smoke outputs (new dirs, no overwrite of old results)

- Line A: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_smoke_20260716`
- Line B: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_smoke_20260716`

## Line B smoke (direct_256_to_1024)

- status: **PASS**
- samples: 10/10 PASS
- any NaN/Inf: False
- any raw shortage: False
- padding/jitter/interpolation/fake points: NO
- runtime total sec: 29.300
- manifest: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_smoke_20260716/manifests/lineB_direct_256_to_1024_smoke_manifest.csv`

## Line A A1 smoke (direct_1024_to_4096)

- status: **PASS**
- samples: 10/10 PASS
- any NaN/Inf: False
- any raw shortage: False
- padding/jitter/interpolation/fake points: NO
- runtime total sec: 13.162
- manifest: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_smoke_20260716/manifests/lineA_A1_direct_1024_to_4096_smoke_manifest.csv`

## Line A A2 smoke (deterministic_4x256_chunks)

- status: **PASS**
- samples: 10/10 PASS
- any NaN/Inf: False
- any raw shortage: False
- padding/jitter/interpolation/fake points: NO
- runtime total sec: 13.994
- manifest: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_smoke_20260716/manifests/lineA_A2_4x256_chunks_smoke_manifest.csv`

## Recommendation

- recommended Line A mode for full run: **A1_direct_1024_to_4096**
- reason: A1 direct whole-object inference PASS and finite; prefers whole-object context.
- A2 also PASS (exact 4096, 4 patches/sample) and remains a fallback if A1 later shows quality issues on full set.

## HPC notes (smoke-only fixes)

- First smoke attempt FAIL: TF1.15 + NumPy 1.21 `tf.meshgrid` bug in `gen_grid`/`duplicate` (`NotImplementedError: Cannot convert a symbolic Tensor ... to a numpy array`).
- Fix applied only in working repo `external/pu_edgeformer_ops_reuse/tf_lib/gcn_lib/vertex.py`: static NumPy grid → `tf.constant` (numerically equivalent for fixed up_ratio=4). Transfer package original left untouched.
- Checkpoint pointer rewritten under `external/pu_edgeformer_ops_reuse/checkpoint_model100/checkpoint` to relative `model-100` (lab absolute paths would not restore on HPC).
- `tf_ops` symlinked from existing PU-GCN compiled ops (same lab ops-reuse protocol).
- EdgeTransformer code path hardcodes upsampler `'duplicate'` (matches code_snapshot used with model-100 restore on lab).

## Safety flags

- DETECTOR_EVAL_STARTED=NO
- POINTNET_CLASSIFIER_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO
- full run executed: NO (smoke only)

## Per-sample notes

### lineB_direct_256_to_1024

- `train/airplane/airplane_0001`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `train/airplane/airplane_0002`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `train/airplane/airplane_0003`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `train/airplane/airplane_0004`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `train/airplane/airplane_0005`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `test/airplane/airplane_0627`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `test/airplane/airplane_0628`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `test/airplane/airplane_0629`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `test/airplane/airplane_0630`: PASS raw=1024 final=1024 crop=exact infer=True err=''
- `test/airplane/airplane_0631`: PASS raw=1024 final=1024 crop=exact infer=True err=''

### lineA_A1_direct_1024_to_4096

- `train/airplane/airplane_0001`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `train/airplane/airplane_0002`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `train/airplane/airplane_0003`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `train/airplane/airplane_0004`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `train/airplane/airplane_0005`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0627`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0628`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0629`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0630`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0631`: PASS raw=4096 final=4096 crop=exact infer=True err=''

### lineA_A2_4x256_chunks

- `train/airplane/airplane_0001`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `train/airplane/airplane_0002`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `train/airplane/airplane_0003`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `train/airplane/airplane_0004`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `train/airplane/airplane_0005`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0627`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0628`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0629`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0630`: PASS raw=4096 final=4096 crop=exact infer=True err=''
- `test/airplane/airplane_0631`: PASS raw=4096 final=4096 crop=exact infer=True err=''

## Estimated full-run commands (NOT executed)

After your confirmation, planned full-run shape (scripts/sbatch to be created then):

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

# Line B full: 256 -> 1024, all train+test (9843+2468)
# output under a NEW dir, e.g. datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_20260716/
sbatch jobs/pu_edgeformer/run_lineB_pu_edgeformer_full.sbatch

# Line A full: recommended A1 direct 1024 -> 4096
# output under a NEW dir, e.g. datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_20260716/
sbatch jobs/pu_edgeformer/run_lineA_pu_edgeformer_full_A1.sbatch
```

Equivalent direct CLI sketch (also NOT executed now):

```bash
source scripts/activate_upsampling_env.sh pugcn
python scripts/run_pu_edgeformer_modelnet40_labparams_full.py \
  --line lineB --mode direct_256_to_1024 --max-per-split 0
python scripts/run_pu_edgeformer_modelnet40_labparams_full.py \
  --line lineA --mode A1_direct_1024_to_4096 --max-per-split 0
```

Rough runtime extrapolate from smoke (10 samples / mode, includes one-time restore):
- Line B ~3 s/sample after warmup → order ~10–12 GPU-hours for 12311 samples (single GPU; chunking recommended).
- Line A A1 ~1.3 s/sample after warmup → order ~4–6 GPU-hours for 12311 samples.
