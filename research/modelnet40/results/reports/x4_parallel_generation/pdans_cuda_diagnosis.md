# PDANS CUDA Environment Diagnosis

- Generated: 2026-06-26 (env fix pass)
- Conda env: `~/.conda/envs/pdans_x4` (Python 3.8)
- Smoke sbatch: `jobs/run_upsampling_x4_smoke.sbatch` (`#SBATCH --gres=gpu:1`)

## Symptom (smoke jobs 1717035 / 1717037)

```
RuntimeError: PDANS requires CUDA
```

GPU was allocated (nvidia-smi showed RTX 3080) but `torch.cuda.is_available()` returned **False**.

## Root cause: corrupted PyTorch metadata

| Check | Before fix | After fix |
| --- | --- | --- |
| `pip show torch` | 2.0.1+cu118 | 2.0.1+cu118 |
| `torch/__version__` | **2.4.0** (mismatch) | 2.0.1+cu118 |
| `torch.version.cuda` | **None** | **11.8** |
| `libc10_cuda.so` | present | present |

`torch/version.py` had been overwritten with CPU-build metadata (`cuda = None`) while CUDA binaries remained — PyTorch reported no CUDA even on GPU nodes.

## Secondary issue: wrong CUDA_HOME in activate script

`scripts/activate_upsampling_env.sh` previously set:

```
CUDA_HOME=/apps/cuda/11.8.0   # no bin/nvcc on this cluster
```

Fixed by `scripts/setup_cuda_toolkit_env.sh` — derives `CUDA_HOME` from `module load cuda/11.8.0` + `which nvcc`.

## Login-node check (no GPU — expected)

```bash
conda activate ~/.conda/envs/pdans_x4
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 2.0.1+cu118 11.8 False
```

## Fix applied

```bash
pip install --force-reinstall --no-deps torch==2.0.1+cu118 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

**Not** a full env rebuild — single-package reinstall only.

## Smoke sbatch diagnostics added

`jobs/run_upsampling_x4_smoke.sbatch` now runs `scripts/print_gpu_cuda_diagnostics.sh` after env activation:

- `HOST`, `CUDA_VISIBLE_DEVICES`, `which python`, `nvidia-smi`
- torch version / `cuda.is_available()` / device name

## GPU validation pending

Job **1717964** (`x4_env_validate`) will confirm `torch.cuda.is_available()` on a GPU node and run PDANS one-sample smoke.

## Recommendation for next smoke rerun

After **1717964** PASS:

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling
export SMOKE_TAG=envfix
bash scripts/submit_upsampling_x4_method.sh pdans   # or per-method
```

Use new `SMOKE_TAG` and new `afterok` dependencies — do **not** reuse failed smoke job IDs 1717035–1717045.
