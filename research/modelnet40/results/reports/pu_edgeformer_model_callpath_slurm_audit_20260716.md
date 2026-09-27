# PU-EdgeFormer Model / Call-Path / Slurm / Infer-Log Audit

- Generated: `2026-07-16 18:31:00 UTC`

## Call path

- sbatch Line B: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pu_edgeformer/run_lineB_pu_edgeformer_full.sbatch`
- sbatch Line A: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pu_edgeformer/run_lineA_pu_edgeformer_full_A1.sbatch`
- python script: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/scripts/run_pu_edgeformer_modelnet40_labparams_full.py`
- wrapper: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/imports/transfer_pu_edgeformer_to_hpc_20260716/wrappers/tf_pugcn_family_patch_infer_many.py`
- imported model module: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/external/pu_edgeformer_ops_reuse/Upsampling/generator.py`
- model utils: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/external/pu_edgeformer_ops_reuse/Common/model_utils.py`
- checkpoint: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/external/pu_edgeformer_ops_reuse/checkpoint_model100`
- tf_ops: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN/tf_ops` (symlink from `external/pu_edgeformer_ops_reuse/tf_ops`)
- model construction: `get_model_cls(FLAGS.model)` with `--model edgetransformer` → `EdgeTransformer`
- inference function: `Model.patch_prediction` inside wrapper loop
- tf_ops reused from PU-GCN: **YES**
- model definition from PU-EdgeFormer: **YES**
- Note: `generator.py` also contains `class PUGCN`, but it is not selected when `model=edgetransformer`.

## Full sbatch commands used

```bash
sbatch jobs/pu_edgeformer/run_lineB_pu_edgeformer_full.sbatch   # job 1749211
sbatch jobs/pu_edgeformer/run_lineA_pu_edgeformer_full_A1.sbatch # job 1749212
```

## Infer-log restore evidence (authoritative)

Wrapper stdout/stderr was captured in per-batch infer logs under dataset `logs/`, not only Slurm `.out` files.

| Line | infer logs | restore to `checkpoint_model100/model-100` | CMD has `--method PU-EdgeFormer --model edgetransformer` |
|------|------------|---------------------------------------------|----------------------------------------------------------|
| B | 64/64 | YES | YES |
| A | 128/128 | YES | YES |

Example restore line:

```
Restoring parameters from .../external/pu_edgeformer_ops_reuse/checkpoint_model100/model-100
```

Example CMD fragment:

```
.../tf_pugcn_family_patch_infer_many.py --repo_dir .../external/pu_edgeformer_ops_reuse --method PU-EdgeFormer --model edgetransformer --checkpoint_dir .../checkpoint_model100 ...
```

## Slurm array logs

- Job 1749211: 16/16 COMPLETED
- Job 1749212: 16/16 COMPLETED
- Chunk audit summaries: 16/16 PASS each line (`audits/*_chunk*_summary.json`)
- No fallback/dummy/repeat-input/copy-input hits in infer logs

## Checkpoint variable audit

- Report: `reports/pu_edgeformer_checkpoint_variable_audit_20260716.txt`
- Total variables: 135
- Non-Adam highlights match EdgeTransformer:
  - `generator/*/conv_q|conv_k|conv_v` (attention-style QKV)
  - `generator/up_block/duplicate/up_layer*`
  - `generator/fc_layer1|2|4`
- `pugcn` name hits in variable names: **0**
- `nodeshuffle` name hits: **0** (EdgeTransformer hardcodes duplicate upsampler in code_snapshot)

## Verdict

- checkpoint restored: **YES** (both lines)
- PU-EdgeFormer model code used: **YES**
- only tf_ops reused from PU-GCN: **YES** (acceptable)
