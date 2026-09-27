# Detector-separated Easy/Moderate/Hard analysis

This directory is a structural rearrangement of the previous dual-detector evidence package. The two detectors no longer appear in the same performance figure or the same root-cause report.

Throughout, `diff/difficult` is written as `Hard` to match the official KITTI metric naming.

Improved experiment design:

- `DETECTOR_AWARE_CONTROLLED_UPSAMPLING_PROPOSAL_EN.md`

## PointRCNN

- `pointrcnn/POINT_RCNN_SEPARATE_ANALYSIS_EN.md`
- Car only, because the frozen model configuration is Car-only.
- Analyses the formal 3,769-frame results together with the earlier 256-frame PDANS 2.5% positive control.
- Easy, Moderate, and Hard are kept separate, as are BBox, BEV, and 3D.

## CenterPoint

- `centerpoint/CENTERPOINT_SEPARATE_ANALYSIS_EN.md`
- Car, Pedestrian, and Cyclist each get their own figures and sections.
- Includes a dedicated analysis of the positive exceptions in Line B: PDANS/Pedestrian, and PDANS and PU-GCN on Cyclist.

## Original three-frame box-level visualisation

The full point clouds, boxes, BEV views, and object crops are still located at:

`../dual_detector_three_frame_root_cause_20260730/frames/`

This directory neither copies nor modifies the old experiment results; it only adds the separated statistics, figures, and reports.
