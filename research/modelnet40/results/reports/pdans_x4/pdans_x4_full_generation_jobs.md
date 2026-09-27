# PDANS ×4 Full Generation Jobs

- Submitted: 2026-06-30 09:57 CEST
- Method: **PDANS** | Upsampling factor: **×4**
- Array: 16 chunks per line (`--array=0-15`)

## Line A — Original 1024 → 4096

| Field | Value |
| --- | --- |
| Job ID | **1723852** (`pdans_x4_a`) |
| Array | 0–15 |
| Input | `datasets/modelnet40_original` (1024×3) |
| Output | `datasets/modelnet40_original_up/pdans_x4` |
| Expected | train=9843, test=2468, total=12311 |
| Target points | 4096×3 |
| Command | `sbatch --job-name=pdans_x4_a --export=ALL,METHOD=pdans,LINE=A jobs/run_upsampling_x4_full_array.sbatch` |
| Sbatch | `jobs/run_upsampling_x4_full_array.sbatch` |
| Chunk script | `scripts/run_upsampling_x4_chunk.py --method pdans --line A --chunk-id N --num-chunks 16 --skip-existing` |
| Logs | `logs/pdans_x4_a_<jobid>_<array>.out` / `.err` |

## Line B — Downsampled50 512 → 2048

| Field | Value |
| --- | --- |
| Job ID | **1723853** (`pdans_x4_b`) |
| Array | 0–15 |
| Input | `datasets/modelnet40_downsampled50` (512×3) |
| Output | `datasets/modelnet40_downsampled50_up/pdans_x4` |
| Expected | train=9843, test=2468, total=12311 |
| Target points | 2048×3 |
| Command | `sbatch --job-name=pdans_x4_b --export=ALL,METHOD=pdans,LINE=B jobs/run_upsampling_x4_full_array.sbatch` |
| Sbatch | `jobs/run_upsampling_x4_full_array.sbatch` |
| Chunk script | `scripts/run_upsampling_x4_chunk.py --method pdans --line B --chunk-id N --num-chunks 16 --skip-existing` |
| Logs | `logs/pdans_x4_b_<jobid>_<array>.out` / `.err` |

## Prerequisites (met)

- GPU validate **PASS** (job 1723850)
- Smoke **PASS** (job 1723851, 40+40 per line)

## Monitor

```bash
squeue.tinygpu -u $USER
find datasets/modelnet40_original_up/pdans_x4 -name '*.npy' | wc -l
find datasets/modelnet40_downsampled50_up/pdans_x4 -name '*.npy' | wc -l
```

## Post-completion

```bash
python3 scripts/audit_upsampling_x4_full.py --method pdans
# copy reports to reports/pdans_x4/ as needed
```
