# Corrected Smoke Rerun — Failure Report

- Generated: 2026-06-26 (monitoring pass)
- Submit batch: `reports/x4_parallel_generation/submit_corrected_2026-06-26T20:26:05Z.md`
- **METHOD `}` bug:** fixed and validated in job logs (`Validated METHOD=pdans` etc.)
- **All 6 smoke jobs FAILED** — full arrays correctly blocked (`DependencyNeverSatisfied`)

## Summary

| Method | Line | Smoke ID | State | Exit | Output `.npy` | Smoke PASS |
| --- | --- | ---: | --- | ---: | ---: | --- |
| PDANS | A | 1717035 | FAILED | 1 | 0 | **NO** |
| PDANS | B | 1717037 | FAILED | 1 | 0 | **NO** |
| PU-Net | A | 1717039 | FAILED | 1 | 0 | **NO** |
| PU-Net | B | 1717041 | FAILED | 1 | 0 | **NO** |
| PU-GCN | A | 1717043 | FAILED | 1 | 0 | **NO** |
| PU-GCN | B | 1717045 | FAILED | 1 | 0 | **NO** |

Full arrays (1717036, 1717038, 1717040, 1717042, 1717044, 1717046): **PENDING / DependencyNeverSatisfied** — not started (correct).

## Root causes

### PDANS (1717035, 1717037)

```
RuntimeError: PDANS requires CUDA
```

- Job allocated GPU (`#SBATCH --gres=gpu:1`; nvidia-smi shows RTX 3080).
- PyTorch in `pdans_x4` env reports no CUDA at `load_pdans_upsampler()` time.
- Likely env/module mismatch: `activate_upsampling_env.sh` loads `python/3.12-conda` then activates `pdans_x4` (Python 3.8); CUDA libs may not be visible to torch.

**Logs:** `logs/pdans_x4_{a,b}_smoke_rerun_171703{5,7}.{out,err}`

### PU-Net (1717039, 1717041)

TF custom ops failed to compile/import:

- `/apps/cuda/11.8.0/bin/nvcc: not found`
- `cannot find -lcudart`, `cannot find -ltensorflow_framework`
- `cuda_runtime.h: No such file or directory`

Inference then fails on import:

```
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
```

All 40 smoke samples failed (`failed=40`). Audit CSV: `reports/punet_x4_smoke_audit_{A,B}_rerun.csv`

**Logs:** `logs/punet_x4_{a,b}_smoke_rerun_171703{9,41}.{out,err}`

### PU-GCN (1717043, 1717045)

Same TF ops / CUDA toolchain failure as PU-Net:

```
from tf_ops.grouping.tf_grouping import knn_point_2
```

**Logs:** `logs/pugcn_x4_{a,b}_smoke_rerun_171704{3,5}.{out,err}`

## Smoke PASS criteria check

| Criterion | Result |
| --- | --- |
| Line A shape 4096×3 | **Not met** — no outputs written |
| Line B shape 2048×3 | **Not met** — no outputs written |
| No NaN/Inf | N/A |
| Label mapping OK | N/A |
| Class count OK | N/A |

## Full array status

| Full array | Depends on | State |
| ---: | ---: | --- |
| 1717036 | 1717035 | DependencyNeverSatisfied |
| 1717038 | 1717037 | DependencyNeverSatisfied |
| 1717040 | 1717039 | DependencyNeverSatisfied |
| 1717042 | 1717041 | DependencyNeverSatisfied |
| 1717044 | 1717043 | DependencyNeverSatisfied |
| 1717046 | 1717045 | DependencyNeverSatisfied |

**Action:** Do not manually start full arrays. Fix CUDA/TF ops env first, then resubmit smoke (outside this monitoring pass).

## Suggested fix direction (not applied this pass)

1. **PDANS:** Ensure `torch.cuda.is_available()` in batch job — align module load / `LD_LIBRARY_PATH` with working EAR pytorch env.
2. **PU-Net / PU-GCN:** Fix `recompile_tf_ops.sh` / `CUDA_HOME` so `nvcc` and `cudart` resolve on compute nodes; or precompile ops on login node and skip runtime compile when `.so` exists.
