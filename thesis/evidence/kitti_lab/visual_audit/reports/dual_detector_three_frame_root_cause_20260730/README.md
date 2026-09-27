# Dual-detector three-frame evidence package

主报告：`THESIS_ROOT_CAUSE_REPORT_ZH.md`

最终三帧：`000590`、`005625`、`006682`。

## 快速入口

- 48 个 case：`case_index.csv`
- 所有框：`analysis/all_selected_frame_boxes.csv`
- 所有逐 GT 转移：`analysis/selected_frame_gt_transitions.csv`
- BBox/BEV/3D 全量 AP：`analysis/all_detector_ap_r40_bbox_bev_3d.csv`
- 全量 transition 统计：`analysis/transition_summary_full_val.csv`
- 三帧目录：`frames/<frame>/line_<a|b>/<method>/<detector>/`

## 复现

```bash
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python

$PY scripts/analyze_dual_detector_frame_transitions.py

$PY scripts/generate_dual_detector_three_frame_evidence.py \
  --frames 000590 005625 006682

$PY scripts/summarize_dual_detector_evidence.py
```

以上命令只读取已有实验结果，并写入本目录；不会运行上采样、检测器训练或检测器推理。

## 非 HTML 交互 3D

```bash
$PY scripts/view_dual_detector_evidence_open3d.py \
  --frame 005625 --line B --method pdans --detector centerpoint \
  --state upsampled --effective --gt-index 4 --point-size 1.0
```

去掉 `--gt-index` 查看完整 frame；去掉 `--effective` 查看完整 observed/E1 点云；`--state overlay` 使用灰色 observed 与蓝色 generated 叠加。
