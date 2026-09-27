# Three-epoch detector-adaptation convergence audit

The three epochs are detector adaptation, not PU-GCN training. PU-GCN uses the fixed PU1K model-100 checkpoint.

## PointRCNN training losses

| Arm | Stage | Epoch 1 median | Epoch 2 median | Epoch 3 median | E1→E3 | Outliers (>100×median) |
|---|---|---:|---:|---:|---:|---:|
| line_a_pugcn | rpn | 1.624055 | 1.444884 | 1.347459 | -17.03% | 0 |
| line_a_pugcn | rcnn | 1.043027 | 0.996256 | 0.959636 | -8.00% | 7 |
| line_b_pugcn | rpn | 2.240354 | 1.815689 | 1.621827 | -27.61% | 1 |
| line_b_pugcn | rcnn | 1.304891 | 1.242044 | 1.205131 | -7.65% | 12 |
| line_b_baseline | rpn | 1.696398 | 1.433417 | 1.290139 | -23.95% | 2 |
| line_b_baseline | rcnn | 0.955630 | 0.945073 | 0.915855 | -4.16% | 13 |
| line_a_pugcn_observed_first | rpn | 1.549086 | 1.419624 | 1.294114 | -16.46% | 0 |
| line_a_pugcn_observed_first | rcnn | 1.012804 | 1.000988 | 0.968045 | -4.42% | 12 |
| line_b_pugcn_observed_first | rpn | 2.087424 | 1.753703 | 1.569361 | -24.82% | 0 |
| line_b_pugcn_observed_first | rcnn | 1.254476 | 1.204835 | 1.168438 | -6.86% | 0 |

## CenterPoint training losses

| Arm | Epoch 1 mean | Epoch 2 mean | Epoch 3 mean | Final LR | E1→E3 |
|---|---:|---:|---:|---:|---:|
| line_a_pugcn | 2.330000 | 2.210000 | 2.110000 | 3.066e-09 | -9.44% |
| line_b_pugcn | 3.690000 | 3.290000 | 3.100000 | 3.066e-09 | -15.99% |
| line_b_baseline | 3.000000 | 2.810000 | 2.650000 | 3.066e-09 | -11.67% |

## Interpretation

A completed schedule or a decreasing training loss is not proof of validation convergence. Only epoch 3 was checkpointed and there was no per-epoch validation AP, so the present records cannot establish that epoch 3 is the optimum or that AP had plateaued. The supported conclusion is: the three-epoch optimization schedule completed and training losses decreased, but validation convergence was not demonstrated. A convergence claim requires saving/evaluating every epoch, an explicitly declared stopping rule and separation of checkpoint selection from the final evaluation.
