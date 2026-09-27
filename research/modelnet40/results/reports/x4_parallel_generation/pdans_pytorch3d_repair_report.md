# PDANS pytorch3d Repair Report

- Updated: 2026-06-29
- Env: `~/.conda/envs/pdans_x4` (Python 3.8.20)

## Repair action

| Step | Result |
| --- | --- |
| torch check | **2.0.1+cu118**, `cuda=11.8` |
| Python version | **3.8** — matches `py38_cu118_pyt201` wheel |
| Removed | `pytorch3d==0.7.7` (ABI mismatch) |
| Installed | `pytorch3d==0.7.4` from official wheel |

```bash
pip install iopath fvcore
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html
```

## CPU validation (login node) — **PASS**

```
torch: 2.0.1+cu118 cuda: 11.8
pytorch3d: 0.7.4
knn_points CPU: (1,16,1) (1,16,1) (1,16,1,3)
```

## GPU validation job

| Field | Value |
| --- | --- |
| Job ID | **1722422** |
| Job name | `x4_env_validate_pdans2` |
| Script | `jobs/run_x4_env_validate_pdans2.sbatch` |
| Report (on completion) | `reports/x4_parallel_generation/x4_env_validate_pdans2_*.md` |

Gate: **PASS_PDANS_READY_FOR_SMOKE** required before production smoke.

## PDANS smoke (queued, afterok on validate)

| Line | Job ID | Job name | Output dir | Dependency |
| --- | ---: | --- | --- | --- |
| A | **1722424** | `pdans_x4_a_smoke3` | `datasets/modelnet40_original_up/pdans_x4_smoke_rerun3` | afterok:1722422 |
| B | **1722425** | `pdans_x4_b_smoke3` | `datasets/modelnet40_downsampled50_up/pdans_x4_smoke_rerun3` | afterok:1722422 |

**No full generation submitted.** Full arrays only after smoke PASS with new `afterok` deps.

## If GPU validate fails

Fallback per `pdans_pytorch3d_repair_plan.md` Option B: source compile `pytorch3d@v0.7.7` on GPU node.

Cancel pending smoke 1722424/1722425 if validate fails (`scancel` only if needed).
