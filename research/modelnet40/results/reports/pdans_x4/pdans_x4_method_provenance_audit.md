# PDANS ×4 Method Provenance Audit

- Audit date: 2026-06-30
- Scope: Confirm PDANS ×4 outputs use PDANS diffusion inference (not generic resample)

## Executive summary

| Line | Input → Output | Status |
| --- | --- | --- |
| A (Original) | 1024 → 4096 (×4) | **method_confirmed** |
| B (Downsampled50) | 512 → 2048 (×4) | **method_confirmed** |

PDANS ×4 generation calls `PDANSUpsampler.upsample_xyz()` → `sampling_ddim()` with checkpoint `PU1K_PDANS.pkl`. Wrapper `finalize_output` / `resample_to_target` exists but smoke + full generation show `method_raw_points == final_points` (PDANS outputs exact target count).

---

## Call chain

```
scripts/run_upsampling_x4_chunk.py
  → load_upsampler("pdans")  [upsampling_x4_factory.py]
  → PDANSUpsampler.upsample_xyz()  [pdans_modelnet40_utils.py]
      → PointNet2CloudCondition + sampling_ddim (PDANS diffusion)
      → checkpoint: external/.../PDANS/checkpoints/PU1K_PDANS.pkl
  → finalize_output(raw_xyz, target_n, seed)  # postprocess hook
  → np.save(...)
```

## Artifacts

| Item | Path |
| --- | --- |
| Wrapper | `scripts/pdans_modelnet40_utils.py` |
| Factory | `scripts/upsampling_x4_factory.py` |
| Chunk script | `scripts/run_upsampling_x4_chunk.py` |
| Smoke script | `scripts/run_upsampling_x4_smoke.py` |
| Full sbatch | `jobs/run_upsampling_x4_full_array.sbatch` |
| Validate sbatch | `jobs/validate_pdans_x4_gpu.sbatch` |
| Checkpoint | `.../PDANS/checkpoints/PU1K_PDANS.pkl` |
| Config | `.../PDANS/pointnet2/exp_configs/PU1K.json` |

## Slurm jobs

| Line | Job ID | Notes |
| --- | ---: | --- |
| A full | 1723852 → rerun 1723887 | 12311 samples PASS |
| B full | 1723853 → 1723888 → 1723925 | CUDA arch fix + rerun |

## Provenance checks

| Check | Line A | Line B |
| --- | --- | --- |
| method_specific_call_detected | true | true |
| checkpoint loaded | PU1K_PDANS.pkl | PU1K_PDANS.pkl |
| generic_resample_detected (primary) | false | false |
| np.repeat / random duplicate only | false | false |
| postprocess_detected | yes (`resample_to_target`) | yes |
| postprocess triggered | no (raw==target) | no (raw==target) |
| **status** | method_confirmed | method_confirmed |

## Log evidence

Smoke audit (`reports/pdans_x4_smoke_audit.csv`): all 80 samples `method_raw_points` equals `final_points` (4096/2048), `status=success`.

Full generation audit: **PASS** — 9843 train + 2468 test per line, correct shapes, no NaN/Inf.

---

## Related

- `reports/pdans_x4/pdans_x4_full_generation_audit.md`
- `reports/pdans_x4/pdans_x4_output_similarity_sanity.md`
