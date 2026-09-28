# Experimental progress brief — 2026-09-17

## Reporting summary

Work progressed from engineering and data-pipeline preparation in February and March to integration of multiple point-cloud upsamplers on KITTI, evaluation with PointRCNN and CenterPoint, strict point-count control, failure analysis, detector adaptation, and per-epoch convergence checks. Environment compatibility, normalization, patch locality, sampling, and voxel issues have corresponding investigation records. Exact dates in February and March lack complete contemporaneous logs; this period is therefore recorded only as retrospectively supported engineering preparation, without claiming completion of the main classification or detection experiments.

The central finding is that increasing point count does not automatically improve detection accuracy. Preserving real observations and adapting the detector recover substantial performance. After extended training, both input lines for both detectors meet the validation-metric plateau criterion used here, but remain below their respective reference baselines.

## Current results

Metric: Car 3D Moderate AP_R40 on the KITTI 3,769-frame validation set. Differences are AP points. Every upsampled row uses observed-first input.

| Detector | Line | Same input, unadapted | Original 3-epoch adaptation | Best after extended training | Reference baseline | Difference from baseline |
|---|---|---:|---:|---:|---:|---:|
| PointRCNN | A | 64.2748 | 71.0075 | **71.5920** | 81.9528 | -10.3608 |
| PointRCNN | B | 35.5289 | 55.2145 | **57.0485** | 68.3312 | -11.2827 |
| CenterPoint | A | 63.2064 | 74.7012 | **75.9661** | 79.2773 | -3.3112 |
| CenterPoint | B | 44.0929 | 61.6639 | **62.9007** | 68.0490 | -5.1483 |

- PointRCNN: Line A completed 18 epochs each for RPN/RCNN and selected RPN e14 + RCNN e2. Line B completed 18 RPN epochs and 24 epochs under the new RCNN schedule, selecting RPN e14 + RCNN e19.
- CenterPoint: each line completed 12 epochs; e12 was the best checkpoint for both.
- Plateau criterion: an improvement must exceed the previous significant best by 0.2 AP points, followed by at least 3 consecutive final epochs without significant improvement. This is not mathematical global convergence. CenterPoint reached its highest numerical AP at e12, but the increase did not exceed the threshold.
- The Line A baseline uses original N-point input and official weights. The Line B baseline uses sparse M-point input and existing 3-epoch adapted weights. No new control trained every baseline to convergence, so this is not a fully equal-budget comparison.
- The detectors were retrained. PU-GCN still uses the fixed PU1K model-100 checkpoint; its network was not retrained on KITTI.
- Work recorded as unfinished in this source brief, including full ModelNet40 classification, new strict full-set EAR runs, and full-set SPU-PMD runs, was not counted as completed here. This dated brief does not override later records from the separate ModelNet40 project.

## Answers to the two central questions

1. **Do more points improve detection?** The current full-validation results do not support that claim. Geometry, the detector input distribution, and preservation of original observations all matter.
2. **Is the baseline gap caused only by the earlier 3-epoch training budget?** Longer training helps, but a gap remains after the defined plateau. Insufficient epoch count alone does not explain the result; it also does not establish that every alternative training scheme would fail.

Evidence paths on the original research host:

- PointRCNN final summary: `/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/FINAL_RESULTS.md`.
- PointRCNN stage evidence: `/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/final_comparison.json`.
- CenterPoint convergence summary: `/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_20260914/reports/centerpoint_convergence_summary.md`.

[Complete chronology, repairs, code, and historical results](EXPERIMENT_WORK_RECORD_202602_TO_20260917_EN.md).
