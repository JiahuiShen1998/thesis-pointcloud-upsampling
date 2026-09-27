# ModelNet40 PU-GCN Capability Audit

- Host: `lms41-24`
- Resolved project root: `/home/ra87racy/projects/modelnet40_pointnet2_upsampling`
- Woody PROJECT_ROOT exists: `False`
- Status: **ready**

## Summary

- code_path: `/home/ra87racy/projects/baseline_detectors/PointRCNN/external/PU-GCN`
- checkpoint_path: `/home/ra87racy/projects/baseline_detectors/PointRCNN/external/PU-GCN/pretrained/pu1k-pugcn`
- environment: `/home/ra87racy/miniconda3/envs/pugcn/bin/python`
- supports_x4: true
- supports_lineA_1024_to_4096: true
- supports_lineB_256_to_1024: true
- existing_outputs: lineA raw=8 strict=8; lineB raw=8 strict=8

## Note

PROJECT_ROOT woody path not mounted on this host (lms41-24); using fallback /home/ra87racy/projects/modelnet40_pointnet2_upsampling | existing outputs lineA strict=8 raw=8; lineB strict=8 raw=8 | gpu_check: gpu=True 2026-07-05 20:35:52.270443: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA 2026-07-05 20:35:52.433315: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero 2026-07-05 20:35:52.434221: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x557dd

## PU-EdgeFormer / TULIP (not in this run)

- PU-EdgeFormer: `pending_checkpoint` (checkpoint not transferred yet)
- TULIP: `pending / not included in this run`
