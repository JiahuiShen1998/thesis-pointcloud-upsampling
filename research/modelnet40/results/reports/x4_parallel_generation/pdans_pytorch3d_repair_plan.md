# PDANS pytorch3d Repair Plan

- Generated: 2026-06-29
- Env: `~/.conda/envs/pdans_x4` (Python 3.8)
- Status: **pytorch3d uninstalled** (mismatched build removed); repair **not executed** yet

---

## Current environment (post-uninstall)

| Package | Version | Status |
| --- | --- | --- |
| torch | **2.0.1+cu118** | OK — `torch.version.cuda=11.8` |
| pytorch3d | — | **removed** (was 0.7.7, ABI mismatch) |
| torchvision | 0.15.2+cu118 | installed (unchanged) |

### Failure that triggered repair

Job **1717964** on GPU node (`tg066`):

```
ImportError: pytorch3d/_C...so: undefined symbol: _ZN2at4_ops15to_dtype_layout4callE...
```

Stack: `pointnet2_ops` → `pytorch3d.ops.knn` → `pytorch3d._C`

**Root cause:** `pytorch3d==0.7.7` wheel/binary was **not** built against `torch==2.0.1+cu118` ABI (likely residual from earlier corrupted torch 2.4.0 metadata era).

Login-node test after torch reinstall:

- `from pytorch3d import _C` — sometimes appeared OK in isolation
- `from pytorch3d.ops import knn` — **FAIL** (same undefined symbol)

---

## Action taken (this pass)

```bash
conda activate ~/.conda/envs/pdans_x4
pip uninstall -y pytorch3d   # removed 0.7.7
```

**Not done:** reinstall, source compile, smoke, full generation, PointNet++ training.

---

## Repair options (ranked)

### Option A — Official prebuilt wheel (recommended first try)

PyTorch3D publishes Colab wheels at:

`https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html`

For current stack:

| Variable | Value |
| --- | --- |
| Python | 3.8 → `py38` |
| CUDA | 11.8 → `cu118` |
| PyTorch | 2.0.1 → `pyt201` |
| **version_str** | **`py38_cu118_pyt201`** |

**Available wheel (verified 2026-06-29):**

```
pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl
```

There is **no** official `0.7.7` wheel for `py38_cu118_pyt201` (only 0.7.4).

**Proposed install (GPU/login node, ~2 min):**

```bash
conda activate ~/.conda/envs/pdans_x4
pip install iopath fvcore
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html
```

**Validation gate:**

```bash
python -c "from pytorch3d.ops import knn; print('pytorch3d ops ok')"
python -c "
import sys
sys.path[:0] = [
  '.../PDANS/pointnet2_ops_lib',
  '.../PDANS',
]
from pointnet2_ops import pointnet2_utils
print('pointnet2_ops ok')
"
# On GPU node:
python -c "import torch; assert torch.cuda.is_available()"
```

Then re-run `jobs/run_upsampling_x4_env_validate.sbatch` **PDANS section only** or dedicated import job.

**Risk:** PDANS may expect 0.7.7 APIs; 0.7.4 is older but same major line. PyTorch3D 0.7.4 release notes support torch 2.0.x. **Low risk** for `knn` op used by PDANS.

---

### Option B — Source compile pytorch3d 0.7.7 (fallback)

If Option A fails PDANS import or runtime:

```bash
conda activate ~/.conda/envs/pdans_x4
module load gcc/11.5.0 cuda/11.8.0
export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"

pip install iopath fvcore ninja
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
```

**Requirements:**

- GPU or CPU node with `nvcc` (module `cuda/11.8.0`)
- ~15–45 min compile time
- `gcc/11.5.0` (cluster module)

**When to use:** Option A wheel missing or PDANS needs 0.7.7 specifically.

---

### Option C — New conda env (last resort)

Create `pdans_x4_v2` with matched conda-forge/pytorch3d stack (e.g. torch 2.3.1 + pytorch3d 0.7.8 via conda). **Not recommended first** — would require retesting entire PDANS checkpoint pipeline and may conflict with existing `pdans_x4` torch 2.0.1+cu118 pin.

---

## Recommended execution sequence

| Step | Action | Submit job? |
| ---: | --- | --- |
| 1 | Option A wheel install | No — interactive on login/GPU node |
| 2 | Import test + `torch.cuda.is_available()` on GPU | `x4_env_validate` (PDANS-only subset) or manual |
| 3 | If PASS → PDANS smoke `SMOKE_TAG=rerun2` | Yes — smoke only |
| 4 | If smoke PASS → PDANS full arrays with new `afterok` | Yes — after smoke |
| 5 | If Option A fails → Option B source compile | Interactive / short GPU job |

**Do not** submit full generation or PointNet++ until smoke PASS.

---

## PDANS dependencies to preserve

| Package | Notes |
| --- | --- |
| torch 2.0.1+cu118 | **keep** — do not upgrade |
| torchvision 0.15.2+cu118 | keep |
| iopath, fvcore | install before pytorch3d if missing |
| pointnet2_ops_lib | local PDANS path — unchanged |

---

## References

- Env validate failure: `reports/x4_parallel_generation/x4_env_validate_1717964_report.md`
- CUDA diagnosis: `reports/x4_parallel_generation/pdans_cuda_diagnosis.md`
- PyTorch3D install docs: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
