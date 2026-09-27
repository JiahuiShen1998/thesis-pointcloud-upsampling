# ModelNet40 PointNet++ Next Training Plan

- Generated at: 2026-07-07 07:03:13 UTC
- This file is a planning artifact only. No training is launched in this pass.

## Goal

Train/evaluate PointNet++ on all protocol branches so that downstream classification accuracy can be compared against the completed geometric quality metrics.

## Line B branches

1. Downsampled x4 baseline, 256 pts
2. Downsampled x4 + EAR, 1024 pts
3. Downsampled x4 + PDANS, 1024 pts
4. Downsampled x4 + PU-Net, 1024 pts
5. Downsampled x4 + PU-GCN, 1024 pts

## Line A branches

1. Original baseline, 1024 pts
2. Original + EAR, 4096 pts
3. Original + PDANS, 4096 pts
4. Original + PU-Net, 4096 pts
5. Original + PU-GCN, 4096 pts

## Recommended execution order

1. Sanity-check PointNet++ input links/manifests for all 10 branches.
2. Run the two baselines first: Original baseline and Downsampled x4 baseline.
3. Run the four Line B upsampling branches.
4. Run the four Line A upsampling branches.
5. Aggregate train/test accuracy into a single comparison summary.
6. Jointly analyze classification accuracy with CD/HD/exact P2F/NUC.

## Analysis focus

- Whether smaller delta vs Original baseline on CD/HD/P2F corresponds to stronger classification accuracy.
- Whether uniformity changes (NUC) help or hurt PointNet++.
- Whether Line A 4096-point upsampling improves recognition compared with Original baseline.
