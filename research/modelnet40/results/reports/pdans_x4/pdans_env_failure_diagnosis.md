# PDANS Environment Failure Diagnosis

- Diagnosed: 2026-06-30 09:44 CEST
- Prior validate job: **1722422** (`x4_env_validate_pdans2`)

## Job 1722422 outcome: **FAIL**

| Step | Result |
| --- | --- |
| 1. `import torch` | **PASS** — torch 2.0.1+cu118 |
| 2. `torch.cuda.is_available()` | **PASS** — NVIDIA GeForce RTX 3080 |
| 3. `pytorch3d.ops.knn_points` on GPU | **PASS** |
| 4. PDANS `load_upsampler("pdans")` | **FAIL** |
| 5. Checkpoint check | Not reached |
| 6. Single-sample smoke | Not reached |

## Root cause

**PDANS import / missing Python dependency** — not pytorch3d ABI, not CUDA runtime, not checkpoint.

```
ModuleNotFoundError: No module named 'open3d'
```

Traceback chain:

- `pdans_modelnet40_utils.py` → `from util import calc_diffusion_hyperparams, sampling_ddim`
- `PDANS/pointnet2/util.py:600` → `import open3d`

Secondary warnings (non-fatal at import stage):

- `pointnet2_ops` cpp extension JIT compile
- `pointops_cuda` cpp extension unavailable (warning only)

## Failure category

| Category | Verdict |
| --- | --- |
| pytorch3d ABI | **Not the blocker** (GPU knn_points passed) |
| PDANS import | **Yes — missing open3d** |
| Checkpoint | **Exists** (`PU1K_PDANS.pkl`, `PUGAN_PDANS.pkl`) — untested in 1722422 |
| GPU runtime | **OK** |

## Fix applied (2026-06-30)

```bash
/home/hpc/iwnt/iwnt189h/.conda/envs/pdans_x4/bin/pip install 'open3d==0.17.0'
```

- CPU import test: `import open3d` → **PASS** (0.17.0)

## Stale jobs cleaned

- **1722424**, **1722425**: `DependencyNeverSatisfied` — **scancelled** 2026-06-30

## Can we continue smoke?

- **Not yet** — need GPU validate re-run after open3d install.
- Next: submit `jobs/validate_pdans_x4_gpu.sbatch`, wait for `reports/pdans_x4/pdans_x4_gpu_validate_report.md` = **PASS**.

## If re-validate still fails

1. Check `pointnet2_ops` / `pointops_cuda` JIT compile on GPU node (undefined symbol / nvcc).
2. Confirm `PU1K_PDANS.pkl` loads without `FileNotFoundError`.
3. Watch for CUDA OOM on single-sample smoke (1024→4096 / 512→2048).
