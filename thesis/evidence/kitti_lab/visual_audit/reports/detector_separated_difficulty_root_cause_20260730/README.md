# Detector-separated Easy/Moderate/Hard analysis

本目录是对上一版双检测器证据包的结构化重排。两个检测器不再出现在同一张性能图或同一份根因报告中。

其中 `diff/difficult` 按KITTI官方指标命名统一写作 `Hard`。

改进实验设计：

- `DETECTOR_AWARE_CONTROLLED_UPSAMPLING_PROPOSAL_ZH.md`

## PointRCNN

- `pointrcnn/POINT_RCNN_SEPARATE_ANALYSIS_ZH.md`
- 仅Car，因为冻结模型配置就是Car-only。
- 正式3,769帧结果和此前256帧PDANS 2.5%正对照同时分析。
- Easy、Moderate、Hard以及BBox、BEV、3D分别保留。

## CenterPoint

- `centerpoint/CENTERPOINT_SEPARATE_ANALYSIS_ZH.md`
- Car、Pedestrian、Cyclist分别成图成节。
- 特别分析Line B中PDANS/Pedestrian与PDANS、PU-GCN/Cyclist的正向例外。

## 原三帧框级可视化

完整点云、框、BEV和目标裁剪仍位于：

`../dual_detector_three_frame_root_cause_20260730/frames/`

本目录没有复制或修改旧实验结果，只新增分离后的统计、图表和报告。
