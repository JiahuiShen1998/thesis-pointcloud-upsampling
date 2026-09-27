# PU-Net / PU-GCN TF1 Custom Ops — CUDA Diagnosis

- Updated: 2026-06-29 (gate pass)
- Conda env: `~/.conda/envs/tf15_upsampling` (Python 3.7, TensorFlow 1.15.0)
- Status: **blocked** — env fixes applied in scripts but compile/smoke not re-run yet

---

## Q&A summary

| Question | Answer |
| --- | --- |
| nvcc available? | **Yes**, after `module load gcc/11.5.0 cuda/11.8.0` |
| nvcc path | `/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/nvcc` |
| CUDA_HOME should be | Same SPACK tree: `dirname $(dirname $(which nvcc))` — **not** `/apps/cuda/11.8.0` (no nvcc there) |
| libcudart.so | `$CUDA_HOME/lib64/libcudart.so.11.8.89` |
| TensorFlow version | 1.15.0 |
| libtensorflow_framework.so | `.../tensorflow_core/libtensorflow_framework.so.1` (37 MB) exists; `.so` symlink needed for `-ltensorflow_framework` |
| GPU vs login compile? | **GPU compute node required** — login-node compile hit `Illegal instruction` / host compiler issues |
| PU-Net / PU-GCN smoke | **Do not resubmit** until compile validated on GPU |

---

## Module setup

```bash
module load gcc/11.5.0 cuda/11.8.0 python/3.12-conda
```

Available: `cuda/11.8.0`, `cuda/12.4.1`, `cuda/12.6.2`, `cuda/12.8.0` — use **11.8.0** to match TF 1.15 + existing wheels.

---

## TensorFlow paths (`tf15_upsampling`)

| Item | Path |
| --- | --- |
| `tf.__file__` | `.../site-packages/tensorflow/__init__.py` |
| `tf.sysconfig.get_lib()` | `.../site-packages/tensorflow_core` |
| Framework lib | `.../tensorflow_core/libtensorflow_framework.so.1` |

`glob` from `tensorflow/` package root finds **0** framework libs — must use `tf.sysconfig.get_lib()`.

---

## Compile script requirements

### PU-Net (`PU-Net/code/tf_ops/sampling/tf_sampling_compile_abi.sh`)

Uses:

```bash
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
nvcc=${CUDA_HOME}/bin/nvcc
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ ... -L$TF_LIB -ltensorflow_framework -lcudart -L $cudalib
```

### PU-GCN (`PU-GCN/tf_ops/compile.sh`)

Derives `tf_lib`, `cuda_dir`, `cuda_lib` from env; exports:

```bash
export LD_LIBRARY_PATH=$tf_lib:$cuda_lib:$LD_LIBRARY_PATH
export PATH=$cuda_dir/bin:$PATH
```

### Required env (via `scripts/activate_upsampling_env.sh`)

```bash
source scripts/setup_cuda_toolkit_env.sh      # CUDA_HOME + nvcc on PATH
source scripts/setup_tf_framework_link.sh     # libtensorflow_framework.so symlink + LD_LIBRARY_PATH
```

| Variable | Value |
| --- | --- |
| `CUDA_HOME` | SPACK cuda-11.8.0 tree |
| `PATH` | `$CUDA_HOME/bin:$PATH` |
| `LD_LIBRARY_PATH` | `$TF_LIB_DIR:$CUDA_HOME/lib64:...` |
| `-I` | `$TF_INC` from `tf.sysconfig.get_include()`; `$CUDA_HOME/include` |
| `-L` | `$TF_LIB` (tensorflow_core); `$CUDA_HOME/lib64` |

Do **not** use stale precompiled `.so` from May 2025 if CUDA/toolchain mismatch — recompile on GPU node after env fix.

---

## Previous failure symptoms (smoke 1717039–1717045)

```
/apps/cuda/11.8.0/bin/nvcc: not found
cannot find -lcudart
cannot find -ltensorflow_framework
cuda_runtime.h: No such file or directory
```

All addressed by `setup_cuda_toolkit_env.sh` + `setup_tf_framework_link.sh` when sourced before compile.

---

## Next recommended steps (not executed this pass)

1. Submit GPU compile-only validation (extend `x4_env_validate` or separate job) on compute node.
2. After compile + import PASS, smoke rerun with new tag:

```bash
export SMOKE_TAG=rerun2
# smoke-only submit (no full arrays until smoke PASS)
sbatch --job-name=punet_x4_a_smoke2 --export=METHOD=punet,LINE=A,SMOKE_TAG=rerun2 \
  jobs/run_upsampling_x4_smoke.sbatch
sbatch --job-name=punet_x4_b_smoke2 --export=METHOD=punet,LINE=B,SMOKE_TAG=rerun2 \
  jobs/run_upsampling_x4_smoke.sbatch
# repeat for pugcn
```

3. New full arrays only with `afterok:<new_smoke_id>` — never reuse 1717035–1717045 deps.

---

## Old failed full arrays — cancelled 2026-06-29

| Job | Depended on failed smoke |
| ---: | ---: |
| 1717036 | 1717035 |
| 1717038 | 1717037 |
| 1717040 | 1717039 |
| 1717042 | 1717041 |
| 1717044 | 1717043 |
| 1717046 | 1717045 |

Queue cleared; no `DependencyNeverSatisfied` jobs remain.
