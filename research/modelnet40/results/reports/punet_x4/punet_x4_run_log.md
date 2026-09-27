Wed Jul  1 04:18:07 PM CEST 2026

## TF ops validate rerun (fix torch diagnostic)

- Fix: `print_gpu_cuda_diagnostics.sh` — torch optional for TF env
- Submitted: job **1725540** (`validate_tf_custom_ops_gpu.sbatch`)
- Reason: job 1722423 failed with `ModuleNotFoundError: No module named 'torch'` before ops compile
- Rerun validate: job **1725547;tinygpu**
- CUDA10 compat validate: job **1725557**
- Smoke A: job 1725559
- Smoke B: job 1725560
- Smoke rerun A: 1725563 (opencv fix)
- Smoke rerun B: 1725564 (opencv fix)
- Smoke GPU fix A: 1725569
- Smoke GPU fix B: 1725570
- Smoke checkpoint fix A: 1725573
- Smoke cudnn fix A: 1725581 B: 1725582
- **Smoke PASS** (40+40/line, jobs 1725581/1725582)
- Full generation submitted: A=**1725588**, B=**1725589**
