# Detector-separated Easy/Moderate/Hard analysis

This catalogue is a structural reordering of the previous version of the double detector evidence package.Two detectors no longer appear in the same performance map or in the same root report.

of which `diff/difficult` Unified writing by name of KITTI official indicator `Hard`.

Improvement of experimental design:

- `DETECTOR_AWARE_CONTROLLED_UPSAMPLING_PROPOSAL_EN.md`

## PointRCNN

- `pointrcnn/POINT_RCNN_SEPARATE_ANALYSIS_EN.md`
- only Car because frozen model configuration is Car-only.
- Official 3,769 frame results and previous 256 frame PDANS 2.5% are being analysed at the same time.
- Easy, Moderate, Hard and BBox, BEV and 3D are retained.

## CenterPoint

- `centerpoint/CENTERPOINT_SEPARATE_ANALYSIS_EN.md`
- Car, Pedestrian and Cyclist are graphs.
- Special analysis of the positive exceptions for PDANS/Pedestrian and PDANS and PU-GCN/Cyclist in Line B.

##  Original 3 frame frame visualization

The complete point cloud, frame, BEV and the target cropping are still in:

`../dual_detector_three_frame_root_cause_20260730/frames/`

This Catalogue no reproduces or modifies the results of old experiments, adding only statistics, charts and reports after separation.
