# TF Custom Ops Compile Validation

- Started: 2026-07-01T14:22:17Z
- Mode: compile-only (no smoke, no full generation)

## PUNET
Activated punet upsampling env (CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb)
HOST=tg06a
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
Wed Jul  1 16:22:22 2026       
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
Recompiling TF ops for punet...
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

ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 764; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 767; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 770; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 789; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 792; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 795; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 814; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 817; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 820; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 839; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 842; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 845; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 864; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 867; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 870; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 911; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 914; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
ptxas /tmp/1725547.tinygpu/tmpxft_00368420_00000000-6_tf_auctionmatch_g.ptx, line 917; warning : Instruction 'shfl' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version
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
2026-07-01 16:23:38.025073: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.0'; dlerror: libcudart.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/hpc/iwnt/iwnt189h/.conda/envs/tf15_upsampling/lib/python3.7/site-packages/tensorflow_core:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64:/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb/lib64
2026-07-01 16:23:38.025103: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
scripts/validate_tf_custom_ops_gpu.sh: line 23: 3573001 Segmentation fault      python -c "${import_expr}"
FAIL: punet TF ops import

## PUGCN
Activated pugcn upsampling env (CUDA_HOME=/apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-11.5.0/cuda-11.8.0-nebruz5tppzmf2sld4yy7b2dq3zld7qb)
HOST=tg06a
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
Wed Jul  1 16:23:43 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:86:00.0 Off |                  N/A |
| 27%   27C    P8             18W /  250W |       1MiB /  11264MiB |      0%      Default |
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
pugcn import ok
PASS: pugcn TF ops import

## Summary
SOME COMPILE VALIDATIONS FAILED
