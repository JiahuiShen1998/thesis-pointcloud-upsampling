# EAR ×4 Preflight Result Check

- Checked: 2026-06-30 09:40 CEST
- Purpose: Confirm EAR ×4 PointNet++ completed before PDANS work (no EAR re-run)

## Final reports

| Experiment | Status | Best test acc | Baseline | Delta | Report |
| --- | --- | ---: | ---: | ---: | --- |
| original_ear_x4_pointnet2 | **completed** (200/200) | **91.48%** (ep 61) | original_baseline 91.95% | -0.47 pp | `reports/original_ear_x4_pointnet2/final_report.md` |
| downsampled50_ear_x4_pointnet2 | **completed** (200/200) | **90.61%** (ep 74) | downsampled50_native512_pointnet2 91.26% | -0.65 pp | `reports/downsampled50_ear_x4_pointnet2/final_report.md` |

## Main results table

- Updated via `scripts/update_modelnet40_main_results_table.py --refresh`
- Path: `reports/modelnet40_pointnet2_main_results_status.md`

## EAR ×4 data generation (unchanged)

- Generation audit: **PASS** (`reports/ear_x4_full_generation_audit.md`)
- Method provenance: **PASS** (`reports/ear_x4_method_provenance_audit.md`)

## Action

- EAR collection: **DONE** — proceed to PDANS validate/smoke/full generation.
- Do **not** re-submit EAR generation or training.
