# PU-Net TF Custom Ops Precheck

- Date: 2026-07-01
- Project: `modelnet40_pointnet2_upsampling`

## Previous validate job (1722423)

| Check | Result |
| --- | --- |
| TensorFlow import | **PASS** — TF 1.15.0 in `tf15_upsampling` |
| CUDA / nvcc | **PASS** — cuda/11.8.0 via SPACK, nvcc 11.8.89 |
| GPU visible | **PASS** — RTX 3080 on tg080 |
| libcudart | **PASS** — `$CUDA_HOME/lib64` in LD_LIBRARY_PATH |
| libtensorflow_framework | **configured** — `setup_tf_framework_link.sh` |
| PU-Net ops compile | **NOT REACHED** |
| PU-Net ops import | **NOT REACHED** |
| PU-GCN ops compile/import | **NOT REACHED** |
| **Overall status** | **FAIL** |

### Failure

```
ModuleNotFoundError: No module named 'torch'
```

**Root cause:** `scripts/print_gpu_cuda_diagnostics.sh` unconditionally imported `torch`, but `tf15_upsampling` is a TF-only env (no PyTorch). The validate job exited before `recompile_tf_ops.sh` ran.

**Blocker type:** diagnostic script false failure (not a CUDA/TF ops compile issue)

### Fix applied (2026-07-01)

- `print_gpu_cuda_diagnostics.sh`: torch diagnostics now optional (skip if torch not installed)
- `validate_tf_custom_ops_gpu.sh`: writes final `status = PASS/FAIL` to `tf_custom_ops_compile_validation_report.md`

## Related files

| Category | Path |
| --- | --- |
| Validate sbatch | `jobs/validate_tf_custom_ops_gpu.sbatch` |
| Validate script | `scripts/validate_tf_custom_ops_gpu.sh` |
| Recompile ops | `scripts/recompile_tf_ops.sh` |
| PU-Net wrapper | `scripts/punet_modelnet40_utils.py` |
| PU-Net checkpoint | `external/.../PU-Net/model/generator2_new6/checkpoint` |
| Previous log | `logs/tf_custom_ops_validate_1722423.out` |

## Gate (after rerun job 1725557)

- **status = PASS** (job 1725557)
- PU-Net smoke: **allowed** ✓
- PU-GCN smoke: **allowed** ✓

### Fixes applied for PASS

1. `print_gpu_cuda_diagnostics.sh` — torch optional in TF env
2. `validate_tf_custom_ops_gpu.sbatch` — proper conda activation via `setup_cuda_toolkit_env.sh`
3. `setup_tf_cuda10_runtime_compat.sh` — libcudart.so.10.0 symlink for TF 1.15
4. `tf_sampling_compile_abi.sh` — CUDA arch sm_75/sm_86
5. `cudatoolkit=10.0` + `cudnn=7.6` in `tf15_upsampling`
6. `opencv-python-headless` in `tf15_upsampling`
7. `PU-Net/code/main.py` — GPU inference + checkpoint weight-only restore
