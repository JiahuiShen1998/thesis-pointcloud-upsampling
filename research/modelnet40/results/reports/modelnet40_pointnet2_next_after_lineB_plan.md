# ModelNet40 PointNet++ Next Steps After Line B

- Generated at: 2026-07-07 19:50:00 UTC
- Prerequisite: **Line B full training 5/5 PASS**

## Line B status

- Full training completed for all 5 branches.
- Classification summary: `reports/modelnet40_pointnet2_lineB_classification_summary.md`
- Geometry vs classification: `reports/modelnet40_lineB_geometry_vs_classification_summary.md`

## Recommended next steps

### 1. Submit Line A smoke jobs

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling

sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineA_original_baseline_1024.sbatch
sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineA_ear_4096.sbatch
sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineA_pdans_4096.sbatch
sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineA_punet_4096.sbatch
sbatch.tinygpu jobs/pointnet2_smoke/smoke_lineA_pugcn_4096.sbatch
```

Branches:
- lineA_original_baseline_1024
- lineA_ear_4096
- lineA_pdans_4096
- lineA_punet_4096
- lineA_pugcn_4096

**Note:** Line A upsampling branches may need the same metadata fix as Line B (label maps only, no baseline manifests).

### 2. After Line A smoke ALL PASS → submit Line A full training

Use `jobs/pointnet2_full/run_lineA_*.sbatch` (5 jobs).

### 3. Final integration

- Line A classification results
- Line B classification results (completed)
- Geometry metrics (completed)
- Cross-line discussion and thesis tables

## Do not

- Re-run Line B full training unless a specific branch fails on re-audit.
- Mix smoke metrics with formal classification results.
- Start Line A full training before Line A smoke passes.
