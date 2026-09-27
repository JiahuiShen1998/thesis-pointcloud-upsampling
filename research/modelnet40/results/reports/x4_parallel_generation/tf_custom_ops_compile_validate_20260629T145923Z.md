# TF Custom Ops Compile Validation

- Started: 2026-06-29T14:59:23Z
- Mode: compile-only (no smoke, no full generation)

## PUNET
Activated punet upsampling env (CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb)
HOST=tg080
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb
TF_LIB_DIR=/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core
LD_LIBRARY_PATH=/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/bin/python
Python 3.7.12
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Mon Jun 29 16:59:28 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080        On  |   00000000:1A:00.0 Off |                  N/A |
| 30%   39C    P8             18W /  300W |       1MiB /  10240MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'
