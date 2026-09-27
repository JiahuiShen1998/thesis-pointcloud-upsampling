# PU-Net ×4 Full Generation Jobs

- Submitted: 2026-07-01
- Array: 16 chunks per line (`--array=0-15`)

| Line | Job ID | Input root | Output root | Target points | Expected count | Log pattern |
| --- | ---: | --- | --- | ---: | ---: | --- |
| A | **1725588** | `datasets/modelnet40_original` | `datasets/modelnet40_original_up/punet_x4` | 4096 | 12311 (9843+2468) | `logs/punet_x4_a_1725588_%a.out` |
| B | **1725589** | `datasets/modelnet40_downsampled50` | `datasets/modelnet40_downsampled50_up/punet_x4` | 2048 | 12311 (9843+2468) | `logs/punet_x4_b_1725589_%a.out` |

## Command

```bash
sbatch.tinygpu --job-name=punet_x4_a --export=ALL,METHOD=punet,LINE=A jobs/run_upsampling_x4_full_array.sbatch
sbatch.tinygpu --job-name=punet_x4_b --export=ALL,METHOD=punet,LINE=B jobs/run_upsampling_x4_full_array.sbatch
```

## Script

- `jobs/run_upsampling_x4_full_array.sbatch`
- `scripts/run_upsampling_x4_chunk.py --method punet --line {A|B}`

## Gate

PointNet++ training blocked until full generation audit + provenance PASS.
