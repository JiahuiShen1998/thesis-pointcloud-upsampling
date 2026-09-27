# PointRCNN convergence results

Metric: Car 3D Moderate AP_R40, all 3,769 KITTI validation frames.

| Arm | Selected RPN epoch | RCNN schedule | Best RCNN epoch | AP | Gain vs unadapted | Change vs old 3-epoch | Gap to baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| line_a_pugcn_observed_first | 14 | 18 | 2 | 71.5920 | 7.3172 | 0.5845 | -10.3608 |
| line_b_pugcn_observed_first | 14 | 24 | 19 | 57.0485 | 21.5197 | 1.8340 | -11.2827 |

Line A baseline: official detector on original-N input. Line B baseline: existing 3-epoch adapted sparse input.

The earlier 18-epoch Line B RCNN best remains 56.88760049103507 (epoch 16), but that schedule did not meet terminal patience.

Longer RCNN schedules are separate experiments from the same initialization, not appended epochs. Full curves and source paths are in final_comparison.json.
