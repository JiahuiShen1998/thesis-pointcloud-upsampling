# PDANS ×4 GPU Validate Report

- Started: 2026-06-30T07:49:19Z
- Host: tg06a
- SLURM_JOB_ID: 1723850

## Environment
Tue Jun 30 09:49:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:86:00.0 Off |                  N/A |
| 27%   27C    P8             14W /  250W |       1MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

- python: /home/hpc/iwnt/iwnt189h/.conda/envs/pdans_x4/bin/python
- python version: Python 3.8.20
- torch: 2.0.1+cu118
- torch.version.cuda: 11.8
- torch.cuda.is_available(): True
- GPU: NVIDIA GeForce RTX 2080 Ti
- pytorch3d: 0.7.4

## torch.cuda
PASS: torch.cuda.is_available()
## pytorch3d GPU knn_points
PASS: pytorch3d knn_points on GPU
## PDANS checkpoints
PASS: checkpoint exists — /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS/checkpoints/PU1K_PDANS.pkl
PASS: checkpoint exists — /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS/checkpoints/PUGAN_PDANS.pkl
## PDANS import + load_upsampler
PASS: PDANS import + load_upsampler
## Single-sample smoke Line A
PASS: smoke generation Line A
PASS: Line A 1024→4096 shape + finite

## Single-sample smoke Line B
PASS: smoke generation Line B
PASS: Line B 512→2048 shape + finite

## Output directories
- Line A: `datasets/modelnet40_original_up/pdans_x4_smoke_validate`
- Line B: `datasets/modelnet40_downsampled50_up/pdans_x4_smoke_validate`

## Conclusion: **PASS**

PDANS GPU validate complete. Ready for smoke generation (80 samples/line).
