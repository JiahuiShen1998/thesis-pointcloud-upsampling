# PointRCNN 64-frame retraining screen

## Protocol

- Detector: PointRCNN, with both RPN and RCNN retrained.
- Training subset: 64 fixed frames (63 effective after filtering), seed `20260823`.
- Schedule: 3 RPN epochs followed by 3 RCNN epochs, 93 steps per stage.
- Evaluation: the same 256-frame `patch_causal_pilot256` split for every condition.
- Metric below: Car 3D AP@0.70 from the repository's legacy KITTI Python evaluator (Easy / Moderate / Hard).
- `Pretrained` uses the original PointRCNN checkpoint; `Finetuned` uses the epoch-3 checkpoint trained on the matching input distribution.

## Results

| Line | Input | Pretrained 3D AP | Finetuned 3D AP | Moderate change after retraining | Finetuned vs matching baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Baseline (original KITTI) | 89.7006 / 79.1910 / 77.8818 | 89.7154 / 78.1811 / 76.5808 | -1.0099 | 0.0000 |
| A | PU-GCN x4 | 85.7645 / 60.2662 / 58.7664 | 84.8716 / 64.4408 / 59.0670 | **+4.1746** | -13.7403 |
| A | PDANS x4 | 88.3118 / 68.5835 / 66.3019 | 87.4731 / 67.0161 / 64.1091 | -1.5674 | -11.1650 |
| B | Baseline (floor N/4) | 84.9560 / 66.6188 / 59.4582 | 83.3933 / 62.9139 / 57.1490 | -3.7049 | 0.0000 |
| B | PU-GCN x4 | 50.0097 / 32.9201 / 26.4822 | 55.9809 / 37.5150 / 32.5904 | **+4.5949** | -25.3989 |
| B | PDANS x4 | 65.1380 / 43.0067 / 39.8897 | 71.1787 / 47.4327 / 41.5808 | **+4.4260** | -15.4812 |

The difference-in-differences for Moderate 3D AP, after subtracting the matching baseline's retraining change, is +5.1845 for Line A PU-GCN, -0.5575 for Line A PDANS, +8.2998 for Line B PU-GCN, and +8.1309 for Line B PDANS.

## Interpretation

This screen supports the hypothesis that a frozen detector suffers from input-distribution mismatch: retraining recovers more than 4 AP for Line A PU-GCN and for both Line B upsampling inputs. However, no upsampled and retrained condition beats its corresponding retrained baseline. The strongest absolute upsampling result is Line A PDANS at 67.0161 Moderate 3D AP, still 11.1650 AP below the Line A baseline.

The baseline itself drops after this very small, single-seed retraining (-1.0099 AP on Line A and -3.7049 AP on Line B), so the run is a screening experiment rather than final evidence. The most useful next confirmation is to repeat the positive adaptation cases with multiple seeds while retaining their paired baselines; only then should a larger-frame run be selected.

## Completion checks

- Six RPN epoch-3 checkpoints and six RCNN epoch-3 checkpoints exist.
- All 12 evaluation conditions are marked `PASS`.
- Every condition contains predictions for all 256 evaluation frames.

