# TF Custom Ops Compile Validation

- Started: 2026-07-01T14:29:17Z
- Mode: compile-only (no smoke, no full generation)

## PUNET
TF cuda compat: symlink libcudart.so.10.0 -> /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64/libcudart.so.11.0
Activated punet upsampling env (CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb)
HOST=tg06a
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb
TF_LIB_DIR=/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core
LD_LIBRARY_PATH=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/.cuda_runtime_compat:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/.cuda_runtime_compat
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/bin/python
Python 3.7.12
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Wed Jul  1 16:29:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:86:00.0 Off |                  N/A |
| 27%   27C    P8             19W /  250W |       1MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
torch: not installed (ok for TF-only methods punet/pugcn)
tf: 1.15.0
tf sysconfig lib: /home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core
framework libs: ['/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so', '/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1']
Recompiling TF ops for punet...
TF cuda compat: symlink libcudart.so.10.0 -> /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64/libcudart.so.11.0
Activated punet upsampling env (CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb)
tf_auctionmatch_g.cu(220): warning #1444-D: function "__shfl_down(float, unsigned int, int)"
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp(207): here was declared deprecated ("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")

tf_auctionmatch_g.cu(221): warning #1444-D: function "__shfl_down(float, unsigned int, int)"
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp(207): here was declared deprecated ("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")

tf_auctionmatch_g.cu(222): warning #1444-D: function "__shfl_down(int, unsigned int, int)"
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp(169): here was declared deprecated ("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")

tf_auctionmatch_g.cu(243): warning #1444-D: function "__shfl_down(float, unsigned int, int)"
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp(207): here was declared deprecated ("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")

tf_auctionmatch_g.cu(244): warning #1444-D: function "__shfl_down(float, unsigned int, int)"
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp(207): here was declared deprecated ("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")

tf_auctionmatch_g.cu(245): warning #1444-D: function "__shfl_down(int, unsigned int, int)"
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp(169): here was declared deprecated ("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")

ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 764; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 767; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 770; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 789; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 792; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 795; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 814; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 817; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 820; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 839; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 842; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 845; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 864; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 867; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 870; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 911; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 914; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725557.tinygpu/tmpxft_00368b7e_00000000-6_tf_auctionmatch_g.ptx, line 917; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
tf_auctionmatch.cpp: In lambda function:
tf_auctionmatch.cpp:16:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   16 |         c->WithRank(c->input(0), 3, &dims1);
      |                                            ^
In file included from tf_auctionmatch.cpp:3:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_auctionmatch.cpp:18:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   18 |         c->WithRank(c->input(1), 3, &dims2);
      |                                            ^
In file included from tf_auctionmatch.cpp:3:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_grouping.cpp: In lambda function:
tf_grouping.cpp:23:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   23 |         c->WithRank(c->input(1), 3, &dims2);
      |                                            ^
In file included from tf_grouping.cpp:9:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_grouping.cpp: In lambda function:
tf_grouping.cpp:48:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   48 |         c->WithRank(c->input(0), 3, &dims1);
      |                                            ^
In file included from tf_grouping.cpp:9:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_grouping.cpp:50:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   50 |         c->WithRank(c->input(1), 3, &dims2);
      |                                            ^
In file included from tf_grouping.cpp:9:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_interpolate.cpp: In lambda function:
tf_interpolate.cpp:29:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   29 |         c->WithRank(c->input(0), 3, &dims1);
      |                                            ^
In file included from tf_interpolate.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_interpolate.cpp:31:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   31 |         c->WithRank(c->input(1), 3, &dims2);
      |                                            ^
In file included from tf_interpolate.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp: In lambda function:
tf_sampling.cpp:20:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   20 |     c->WithRank(c->input(0), 2, &dims1);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp:22:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   22 |     c->WithRank(c->input(1), 2, &dims2);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp: In lambda function:
tf_sampling.cpp:34:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   34 |     c->WithRank(c->input(0), 3, &dims1);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp: In lambda function:
tf_sampling.cpp:47:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   47 |     c->WithRank(c->input(0), 3, &dims1);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp:49:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   49 |     c->WithRank(c->input(1), 2, &dims2);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
TF ops compile finished for punet
ldd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-Net/code/tf_ops/sampling/tf_sampling_so.so:
	linux-vdso.so.1 (0x00007ffc731d6000)
	libtensorflow_framework.so.1 => /home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1 (0x000073bff6a00000)
	libcudart.so.11.0 => /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64/libcudart.so.11.0 (0x000073bff6600000)
	libstdc++.so.6 => /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-13.3.0/gcc-11.5.0-ecfwth4lnz6pnfrgh6kcs6igy7e4yqzz/lib64/libstdc++.so.6 (0x000073bff63e5000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x000073bff6917000)
	libgcc_s.so.1 => /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-13.3.0/gcc-11.5.0-ecfwth4lnz6pnfrgh6kcs6igy7e4yqzz/lib64/libgcc_s.so.1 (0x000073bff87bc000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x000073bff6000000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x000073bff87b7000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x000073bff87b2000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x000073bff87ad000)
	/lib64/ld-linux-x86-64.so.2 (0x000073bff8806000)
2026-07-01 16:30:36.369347: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
punet import ok
PASS: punet TF ops import

## PUGCN
TF cuda compat: symlink libcudart.so.10.0 -> /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64/libcudart.so.11.0
Activated pugcn upsampling env (CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb)
HOST=tg06a
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb
TF_LIB_DIR=/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core
LD_LIBRARY_PATH=/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/.cuda_runtime_compat:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/.cuda_runtime_compat:/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/.cuda_runtime_compat
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/bin/python
Python 3.7.12
/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Wed Jul  1 16:30:41 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:86:00.0 Off |                  N/A |
| 27%   27C    P8             20W /  250W |       1MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
torch: not installed (ok for TF-only methods punet/pugcn)
tf: 1.15.0
tf sysconfig lib: /home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core
framework libs: ['/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so', '/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1']
Recompiling TF ops for pugcn...
TF cuda compat: symlink libcudart.so.10.0 -> /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64/libcudart.so.11.0
Activated pugcn upsampling env (CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb)
linux
===> tf_lib is located at: /home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core
===> tf_inc is located at: /home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include
===> cuda_dir is located at: /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb
===> cuda_lib is located at: /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64
===> cuda_inc is located at: /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/include
===> tf_compile_flags: -I/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0
===> tf_link_flags: -L/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core -l:libtensorflow_framework.so.1
===> change the location of them if wrong
current working dir is: /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN/tf_ops
tf_grouping.cpp: In lambda function:
tf_grouping.cpp:23:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   23 |         c->WithRank(c->input(1), 3, &dims2);
      |                                            ^
In file included from tf_grouping.cpp:9:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_grouping.cpp: In lambda function:
tf_grouping.cpp:48:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   48 |         c->WithRank(c->input(0), 3, &dims1);
      |                                            ^
In file included from tf_grouping.cpp:9:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_grouping.cpp:50:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   50 |         c->WithRank(c->input(1), 3, &dims2);
      |                                            ^
In file included from tf_grouping.cpp:9:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_interpolate.cpp: In lambda function:
tf_interpolate.cpp:29:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   29 |         c->WithRank(c->input(0), 3, &dims1);
      |                                            ^
In file included from tf_interpolate.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_interpolate.cpp:31:44: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   31 |         c->WithRank(c->input(1), 3, &dims2);
      |                                            ^
In file included from tf_interpolate.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp: In lambda function:
tf_sampling.cpp:20:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   20 |     c->WithRank(c->input(0), 2, &dims1);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp:22:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   22 |     c->WithRank(c->input(1), 2, &dims2);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp: In lambda function:
tf_sampling.cpp:34:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   34 |     c->WithRank(c->input(0), 3, &dims1);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp: In lambda function:
tf_sampling.cpp:47:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   47 |     c->WithRank(c->input(0), 3, &dims1);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
tf_sampling.cpp:49:40: warning: ignoring return value of 'tensorflow::Status tensorflow::shape_inference::InferenceContext::WithRank(tensorflow::shape_inference::ShapeHandle, tensorflow::int64, tensorflow::shape_inference::ShapeHandle*)', declared with attribute 'warn_unused_result' [-Wunused-result]
   49 |     c->WithRank(c->input(1), 2, &dims2);
      |                                        ^
In file included from tf_sampling.cpp:8:
/home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/include/tensorflow/core/framework/shape_inference.h:394:10: note: declared here
  394 |   Status WithRank(ShapeHandle shape, int64 rank,
      |          ^~~~~~~~
TF ops compile finished for pugcn
ldd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN/tf_ops/grouping/tf_grouping_so.so:
	linux-vdso.so.1 (0x00007ffe80153000)
	libcudart.so.11.0 => /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64/libcudart.so.11.0 (0x0000778840000000)
	libtensorflow_framework.so.1 => /home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1 (0x000077883e200000)
	libstdc++.so.6 => /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-13.3.0/gcc-11.5.0-ecfwth4lnz6pnfrgh6kcs6igy7e4yqzz/lib64/libstdc++.so.6 (0x000077883dfe5000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x000077883ff17000)
	libgcc_s.so.1 => /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-13.3.0/gcc-11.5.0-ecfwth4lnz6pnfrgh6kcs6igy7e4yqzz/lib64/libgcc_s.so.1 (0x000077884030e000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x000077883dc00000)
	/lib64/ld-linux-x86-64.so.2 (0x0000778840353000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x0000778840309000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x0000778840304000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007788402ff000)
pugcn import ok
PASS: pugcn TF ops import

## Summary
ALL COMPILE VALIDATIONS PASS
