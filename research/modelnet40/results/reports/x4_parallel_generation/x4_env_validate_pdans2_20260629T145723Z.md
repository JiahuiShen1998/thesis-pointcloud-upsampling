# PDANS GPU Environment Validate (pdans2)

- Started: 2026-06-29T14:57:23Z
- Job purpose: PASS_PDANS_READY_FOR_SMOKE gate

## Diagnostics
Activated pdans upsampling env (CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb)
HOST=tg083
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb
TF_LIB_DIR=
LD_LIBRARY_PATH=/home/hpc/iwnt/iwnt189h/.conda/envs/pdans_x4/lib/python3.8/site-packages/torch/lib:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64
/home/hpc/iwnt/iwnt189h/.conda/envs/pdans_x4/bin/python
Python 3.8.20
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Mon Jun 29 16:57:26 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080        On  |   00000000:3E:00.0 Off |                  N/A |
| 30%   37C    P8             25W /  300W |       1MiB /  10240MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
torch: 2.0.1+cu118
torch cuda: 11.8
cuda available: True
device count: 1
device name: NVIDIA GeForce RTX 3080

## torch.cuda
PASS: torch.cuda.is_available()
## pytorch3d GPU knn_points
knn gpu ok: torch.Size([1, 16, 1]) torch.Size([1, 16, 1]) torch.Size([1, 16, 1, 3])
PASS: pytorch3d knn_points on GPU
## PDANS import
/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS/pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py:18: UserWarning: Unable to load pointnet2_ops cpp extension. JIT Compiling.
  warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")
/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS/pointops/functions/pointops.py:13: UserWarning: Unable to load pointops_cuda cpp extension.
  warnings.warn("Unable to load pointops_cuda cpp extension.")
Traceback (most recent call last):
  File "<stdin>", line 13, in <module>
  File "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/scripts/upsampling_x4_factory.py", line 20, in load_upsampler
    return load_pdans_upsampler(**kwargs)
  File "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/scripts/pdans_modelnet40_utils.py", line 114, in load_pdans_upsampler
    return PDANSUpsampler(**kwargs)
  File "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/scripts/pdans_modelnet40_utils.py", line 45, in __init__
    from util import calc_diffusion_hyperparams, sampling_ddim
  File "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS/pointnet2/util.py", line 600, in <module>
    import open3d
ModuleNotFoundError: No module named 'open3d'
