# Line B EAR / PDANS Provenance Audit

- Generated at: 2026-07-03 10:59:28 UTC
- Downsampled x4: **PASS** — reused existing modelnet40_downsampled_x4; no rebuild needed

## Method summaries

### EAR
- provenance_status: **PASS**
- strict_count: 12311 / 12311
- raw_count: 12311
- raw_shape_distribution: {"1024": 12311}
- sample_pass: 20/20
- warnings: none
- inference_evidence: chunk_audit_csv+run_lineB_downsampled_x4_upsampling_chunk.py

### PDANS
- provenance_status: **PASS**
- strict_count: 12311 / 12311
- raw_count: 12311
- raw_shape_distribution: {"1024": 12311}
- sample_pass: 20/20
- warnings: none
- inference_evidence: chunk_audit_csv+run_lineB_downsampled_x4_upsampling_chunk.py

## Pipeline

`modelnet40_downsampled_x4` (256) → method inference → `lineB_downsampled_x4_up/raw/METHOD` → strict normalize → `strict_N/METHOD` (1024)

No evidence of Line A 4096 / old 512→2048 / original copy in spot checks.