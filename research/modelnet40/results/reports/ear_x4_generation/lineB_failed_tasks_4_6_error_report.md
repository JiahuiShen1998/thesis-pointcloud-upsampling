# EAR ×4 Line B — Failed Tasks 4 & 6 Error Report

- Job array: **1716175** (`ear_x4_b`)
- Failed tasks: **4**, **6**
- Generated: 2026-06-26
- Output directory (preserved): `datasets/modelnet40_downsampled50_up/ear_x4`

## Summary

| Task | State | Exit | Elapsed | Root cause |
| ---: | --- | ---: | --- | --- |
| 4 | FAILED | 1:0 | ~2s | `FileNotFoundError` on metadata symlink race |
| 6 | FAILED | 1:0 | ~1s | Same |

**Conclusion:** Not a data-directory corruption issue. Not a systemic EAR/model failure. Two array tasks crashed at startup in `ensure_metadata_link()` when parallel workers raced on `metadata/class_to_idx.json` symlink creation.

**Remediation:** Retry only tasks 4 and 6 with fixed `ensure_metadata_link()` (catches `FileNotFoundError` on unlink). Do **not** delete existing Line B outputs or rerun the full 1716175 array.

## sacct snapshot

```
1716175_4   ear_x4_b   FAILED   1:0   00:00:02
1716175_6   ear_x4_b   FAILED   1:0   00:00:01
(other tasks 0–3,5,7–15 COMPLETED)
```

## Log files

| Task | stdout | stderr |
| ---: | --- | --- |
| 4 | `logs/ear_x4_generation/ear_x4_b_1716175_4.out` | `logs/ear_x4_generation/ear_x4_b_1716175_4.err` |
| 6 | `logs/ear_x4_generation/ear_x4_b_1716175_6.out` | `logs/ear_x4_generation/ear_x4_b_1716175_6.err` |

## Stack trace (both tasks)

```
File ".../scripts/run_ear_x4_chunk.py", line 180, in main
  ensure_metadata_link(cfg["output_root"], CLASS_TO_IDX)
File ".../scripts/ear_modelnet40_utils.py", line 117, in ensure_metadata_link
  dst.unlink()
FileNotFoundError: .../datasets/modelnet40_downsampled50_up/ear_x4/metadata/class_to_idx.json
```

## Output impact at failure time

| Split | Count at report | Expected |
| --- | ---: | ---: |
| train | 8303 | 9843 |
| test | 2468 | 2468 |

Missing train samples (~1540) correspond to chunks 4 and 6 (and any partial writes from those chunk ranges). Test split was already complete from other tasks.

## Code fix applied

`scripts/ear_modelnet40_utils.py` — `ensure_metadata_link()`:

- Wrap `dst.unlink()` in `try/except FileNotFoundError`
- Only create symlink when `dst` does not exist after unlink attempt

## Retry plan

```bash
sbatch.tinygpu --job-name=ear_x4_b_retry --array=4,6 \
  jobs/run_ear_x4_lineB_cpu_array.sbatch
```

- Do not `scancel` 1716175 (already finished except failed tasks).
- Do not remove `datasets/modelnet40_downsampled50_up/ear_x4` existing `.npy` files.
