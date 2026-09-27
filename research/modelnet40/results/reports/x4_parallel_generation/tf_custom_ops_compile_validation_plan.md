# TF Custom Ops — GPU Compile Validation Plan

- Generated: 2026-06-29
- Methods: **PU-Net**, **PU-GCN** (TensorFlow 1.15)
- Mode: **compile-only** — no smoke, no full generation, no PointNet++ training

---

## Problem summary

Corrected smoke jobs **1717039–1717045** failed because TF custom ops could not compile:

```
/apps/cuda/11.8.0/bin/nvcc: not found
cannot find -lcudart
cannot find -ltensorflow_framework
```

**Fix already in repo** (not yet validated on GPU):

| Script | Purpose |
| --- | --- |
| `scripts/setup_cuda_toolkit_env.sh` | `module load cuda/11.8.0`; set `CUDA_HOME` from `which nvcc` |
| `scripts/setup_tf_framework_link.sh` | Symlink `libtensorflow_framework.so` → `.so.1`; extend `LD_LIBRARY_PATH` |
| `scripts/activate_upsampling_env.sh` | Sources both for `punet` / `pugcn` |

---

## Diagnosed toolchain paths

| Component | Path / value |
| --- | --- |
| CUDA module | `module load gcc/11.5.0 cuda/11.8.0` |
| nvcc | `/apps/SPACK/.../cuda-11.8.0-.../bin/nvcc` |
| CUDA_HOME | Same SPACK tree (not `/apps/cuda/11.8.0`) |
| libcudart | `$CUDA_HOME/lib64/libcudart.so.11.8.89` |
| TensorFlow | 1.15.0 in `tf15_upsampling` |
| TF lib dir | `.../site-packages/tensorflow_core` |
| Framework lib | `libtensorflow_framework.so.1` (37 MB) |

**Compile must run on GPU compute node** — login-node nvcc hit `Illegal instruction` during test compile.

---

## New artifacts (this pass)

| File | Role |
| --- | --- |
| `scripts/validate_tf_custom_ops_gpu.sh` | Compile + import test for punet & pugcn |
| `jobs/validate_tf_custom_ops_gpu.sbatch` | GPU Slurm wrapper (`tf_ops_validate`) |

### What the validation job does

1. `activate_upsampling_env.sh punet` → diagnostics → `recompile_tf_ops.sh punet`
2. Import: `from tf_ops.sampling.tf_sampling import farthest_point_sample`
3. Repeat for `pugcn` with `from tf_ops.grouping.tf_grouping import knn_point_2`
4. Write report: `reports/x4_parallel_generation/tf_custom_ops_compile_validate_<timestamp>.md`

### What it does **not** do

- No `run_upsampling_x4_smoke.py`
- No full array generation
- No PointNet++ training
- No deletion of existing `.so` outputs in external repos (recompile overwrites in place)

---

## Submit command (when ready — **not submitted this pass**)

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling
sbatch.tinygpu jobs/validate_tf_custom_ops_gpu.sbatch
```

| Field | Value |
| --- | --- |
| Job name | `tf_ops_validate` |
| Partition | work |
| GPU | 1 |
| Time limit | 2 h |
| Logs | `logs/tf_custom_ops_validate_<jobid>.{out,err}` |

---

## PASS criteria

| Check | PU-Net | PU-GCN |
| --- | --- | --- |
| nvcc found | yes | yes |
| `recompile_tf_ops.sh` exit 0 | yes | yes |
| Python import of compiled op | `tf_sampling` | `tf_grouping` |
| No link errors | `-lcudart`, `-ltensorflow_framework` resolved | same |

---

## After compile PASS — next gates (future passes)

| Step | Action |
| ---: | --- |
| 1 | Submit smoke only: `SMOKE_TAG=rerun2`, job names `punet_x4_{a,b}_smoke2`, `pugcn_x4_{a,b}_smoke2` |
| 2 | Audit smoke: Line A 4096×3, Line B 2048×3, no NaN/Inf |
| 3 | Submit full arrays with **new** `afterok:<smoke_job_id>` |
| 4 | Full generation audit PASS → PointNet++ (separate gate) |

**Do not** reuse failed smoke deps (1717035–1717045) or cancelled full arrays.

---

## PU-Net / PU-GCN compile entry points

| Method | Compile script | CWD |
| --- | --- | --- |
| PU-Net | `PU-Net/code/tf_ops/compile.sh` | calls `tf_*_compile_abi.sh` per op |
| PU-GCN | `PU-GCN/tf_ops/compile.sh linux` | sources per-op compile scripts |

Linker flags (from `tf_sampling_compile_abi.sh`):

```bash
g++ ... -L$TF_LIB -ltensorflow_framework -lcudart -L $cudalib
```

Requires `setup_tf_framework_link.sh` symlink for `-ltensorflow_framework`.

---

## Parallel work note

EAR ×4 PointNet++ jobs **1722408 / 1722409** are running — **do not cancel or resubmit**. This TF validation is independent and can run on a separate GPU when queue allows.

---

## References

- `reports/x4_parallel_generation/tf_custom_ops_cuda_diagnosis.md`
- `reports/x4_parallel_generation/corrected_smoke_rerun_failure_report.md`
- `scripts/recompile_tf_ops.sh`
