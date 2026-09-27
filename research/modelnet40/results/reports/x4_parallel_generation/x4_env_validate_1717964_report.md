# x4_env_validate Job 1717964 Report

- Job ID: **1717964** (`x4_env_validate`)
- State: **FAILED** (exit 1, elapsed ~23s)
- Logs: `logs/x4_env_validate_1717964.{out,err}`
- Generated: 2026-06-29

## Conclusion

**`FAIL_PDANS_IMPORT_OR_COMPILE`**

CUDA is visible and `torch.cuda.is_available()` is True, but PDANS model import fails due to **pytorch3d ABI mismatch** after torch reinstall.

PDANS smoke **not** resubmitted (gate not met).

---

## Environment extraction

| Field | Value |
| --- | --- |
| hostname | `tg066` |
| CUDA_VISIBLE_DEVICES | `0` |
| nvidia-smi | OK — NVIDIA GeForce RTX 2080 Ti |
| python | `/home/hpc/iwnt/iwnt189h/.conda/envs/pdans_x4/bin/python` (3.8.20) |
| torch.__version__ | `2.0.1+cu118` |
| torch.version.cuda | `11.8` |
| torch.cuda.is_available() | **True** |
| torch.cuda.device_count() | **1** |
| GPU name | NVIDIA GeForce RTX 2080 Ti |
| CUDA_HOME (after activate) | `/apps/SPACK/.../cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb` |
| nvcc | available |

## PDANS import / compile

| Step | Result |
| --- | --- |
| `torch.cuda.is_available()` | **PASS** |
| PDANS `load_upsampler()` / import | **FAIL** |

### Error

```
ImportError: .../pytorch3d/_C.cpython-38-x86_64-linux-gnu.so: undefined symbol:
_ZN2at4_ops15to_dtype_layout4callE...
```

Stack: `pointnet2_ops` → `pytorch3d.ops.knn` → `pytorch3d._C` ABI break.

**Root cause:** `torch==2.0.1+cu118` was force-reinstalled, but `pytorch3d==0.7.7` binary was built against a different torch ABI (likely pre-reinstall torch 2.4.0 metadata era).

## Single-sample smoke

| Line | Target shape | Result |
| --- | --- | --- |
| A | 4096×3 | **Not reached** (import failed) |
| B | 2048×3 | **Not tested** (job failed before punet/pugcn sections) |

No output written to `*_x4_smoke_envfix`.

## Gate decision

| Gate | Required | Actual |
| --- | --- | --- |
| PASS_PDANS_READY_FOR_SMOKE | all checks pass | **NO** |
| Submit PDANS smoke rerun2 | only if PASS | **Skipped** |

## Recommended fix (next pass, not applied here)

Reinstall pytorch3d against current torch:

```bash
conda activate ~/.conda/envs/pdans_x4
pip install --force-reinstall pytorch3d==0.7.7 \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html
```

Or rebuild pytorch3d from source with `torch==2.0.1+cu118`, then re-run env validate before smoke rerun2.
