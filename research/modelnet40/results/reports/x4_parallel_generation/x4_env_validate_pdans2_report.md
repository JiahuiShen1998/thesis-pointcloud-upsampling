# x4_env_validate_pdans2 — Report (pending)

- Job ID: **1722422**
- Job name: `x4_env_validate_pdans2`
- Submitted: 2026-06-29
- Status: **queued / running** — fill in after job completes

## Purpose

GPU gate before PDANS production smoke (`rerun3`):

1. `torch.cuda.is_available() == True`
2. `pytorch3d.ops.knn_points` on GPU
3. PDANS `load_upsampler("pdans")` import
4. Single-sample smoke Line A (4096×3) and Line B (2048×3), tag `validate2`

## Logs

- `logs/x4_env_validate_pdans2_1722422.out`
- `logs/x4_env_validate_pdans2_1722422.err`

## Expected conclusion

| Outcome | Next step |
| --- | --- |
| **PASS_PDANS_READY_FOR_SMOKE** | Smoke 1722424/1722425 run automatically (afterok) |
| **FAIL** | Do not run smoke; scancel 1722424/1722425 if still pending; try Option B compile |

## Downstream smoke (dependency)

| Job | Line | SMOKE_TAG |
| ---: | --- | --- |
| 1722424 | A | rerun3 |
| 1722425 | B | rerun3 |

Update this file when job 1722422 completes.
