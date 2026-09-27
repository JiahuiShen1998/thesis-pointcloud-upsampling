# PU-GCN CenterPoint adaptation convergence

Overall status: **CONVERGED**

PU-GCN remains fixed at the released PU1K model-100 checkpoint. The epochs below are detector-adaptation epochs.

Stopping evidence uses full 3,769-frame KITTI validation Car 3D Moderate AP_R40, min_delta=0.200 AP, patience=3 epochs.

| Arm | Terminal plateau | Best epoch | Best AP | Previous e3 | Change vs e3 | Baseline AP | Gap to baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| line_a_pugcn_observed_first | True | 12 | 75.9661 | 74.7012 | +1.2649 | 79.2773 | -3.3112 |
| line_b_pugcn_observed_first | True | 12 | 62.9007 | 61.6639 | +1.2368 | 68.0490 | -5.1483 |
