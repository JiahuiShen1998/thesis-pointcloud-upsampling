# PDANS ×4 PointNet++ Training Jobs

- Submitted: 2026-06-30

| Experiment | Line | Job ID | `num_point` | Log |
| --- | --- | ---: | ---: | --- |
| original_pdans_x4_pointnet2 | A | **1723957** (`p2_pdans4_a`) | 4096 | `logs/original_pdans_x4_pointnet2/train_1723957.log` |
| downsampled50_pdans_x4_pointnet2 | B | **1723958** (`p2_pdans4_b`) | 2048 | `logs/downsampled50_pdans_x4_pointnet2/train_1723958.log` |

## Monitor

```bash
squeue.tinygpu -u $USER
tail -f logs/original_pdans_x4_pointnet2/train.log
tail -f logs/downsampled50_pdans_x4_pointnet2/train.log
```

## Baseline comparison targets

| Experiment | Compare against | Target |
| --- | --- | ---: |
| original_pdans_x4_pointnet2 | original_baseline | 91.95% |
| downsampled50_pdans_x4_pointnet2 | downsampled50_native512_pointnet2 | 91.26% |
