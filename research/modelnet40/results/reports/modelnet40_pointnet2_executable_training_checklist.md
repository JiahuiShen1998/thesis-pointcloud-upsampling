# ModelNet40 PointNet++ Executable Training Checklist

- Generated at: 2026-07-07 07:29:07 UTC

## Recommended order

First batch Line B:
- downsampled baseline
- EAR
- PDANS
- PU-Net
- PU-GCN

Second batch Line A:
- original baseline
- EAR
- PDANS
- PU-Net
- PU-GCN

## Smoke commands

`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_downsampled_x4_baseline_256.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_ear_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_pdans_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_punet_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineB_pugcn_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_original_baseline_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_ear_4096.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_pdans_4096.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_punet_4096.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_smoke/smoke_lineA_pugcn_4096.sbatch`

## Full training commands

`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineB_downsampled_x4_baseline_256.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineB_ear_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineB_pdans_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineB_punet_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineB_pugcn_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineA_original_baseline_1024.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineA_ear_4096.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineA_pdans_4096.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineA_punet_4096.sbatch`
`sbatch /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/jobs/pointnet2_full/run_lineA_pugcn_4096.sbatch`

## Monitoring commands

```bash
watch -n 30 "squeue -u $USER | rg 'pn2|pointnet2|lineA|lineB' || true"
```

```bash
rg -n "First train batch tensor shape|First train batch loss|Training finished|Saved checkpoint" /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_smoke /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final
```

```bash
ls /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/* /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/lineA_original_up/* /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/*
```

## Expected output files

- `checkpoints/best_model.pth`
- `train.log`
- `metrics.json`
- `<variant>_result.csv`
- `<variant>_result.md`
- SLURM stdout/stderr under `logs/pointnet2_smoke/` or `logs/pointnet2_full/`

## Result summary target

- Consolidate final branch metrics into `reports/modelnet40_pointnet2_main_results_status.csv` after all 10 full runs complete.
- Keep the protocol-level narrative in `reports/modelnet40_x4_final_protocol_report.md`.
