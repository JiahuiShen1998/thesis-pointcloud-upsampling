# 实验进展简报（截至 2026-09-17）

## 可直接用于汇报的版本

从二、三月的工程与数据流程搭建开始，工作逐步推进到 KITTI 上多种点云上采样方法的接入、PointRCNN/CenterPoint 双检测器评估、严格点数控制、失败原因分析，以及检测器适配训练和逐 epoch 收敛验证。早期出现的环境兼容、归一化、patch 局部性、采样和体素问题均有相应排查记录。二、三月的具体日期缺少同期完整日志，因此该阶段仅列为回溯可确认的工程准备，不宣称已完成分类或检测主实验。

核心结果是：上采样增加点数，并不自动提高检测精度。保留真实观测点并进行检测器适配能明显恢复性能；延长训练后，两个检测器的 Line A/B 都满足本次设定的验证指标平台期判据，但仍低于各自参照基线。

## 当前结果

指标为 KITTI 3,769 帧验证集 Car 3D Moderate AP_R40；差值单位为 AP 点。所有上采样行均为 observed-first 输入。

| 检测器 | Line | 同输入未适配 | 原 3-epoch | 延长训练后最佳 | 参照基线 | 与基线差值 |
|---|---|---:|---:|---:|---:|---:|
| PointRCNN | A | 64.2748 | 71.0075 | **71.5920** | 81.9528 | -10.3608 |
| PointRCNN | B | 35.5289 | 55.2145 | **57.0485** | 68.3312 | -11.2827 |
| CenterPoint | A | 63.2064 | 74.7012 | **75.9661** | 79.2773 | -3.3112 |
| CenterPoint | B | 44.0929 | 61.6639 | **62.9007** | 68.0490 | -5.1483 |

- PointRCNN：A 的 RPN/RCNN 各完成 18 epochs，选 RPN e14 + RCNN e2；B 的 RPN 完成 18、RCNN 新 schedule 完成 24，选 RPN e14 + RCNN e19。
- CenterPoint：A/B 各完成 12 epochs，最佳 checkpoint 均为 e12。
- 判据：超过显著最佳值 0.2 AP 点才算显著改善，末尾连续至少 3 epochs 未显著改善。不是数学意义的全局收敛；CenterPoint 虽然 e12 取得数值最高 AP，但增幅未超过该阈值。
- A 基线是原始 N 点、官方权重；B 基线是稀疏 M 点、已有 3-epoch adapted 权重。没有新增“全部 baseline 同样训练到收敛”的对照，不能称为完全等预算比较。
- 重训的是检测器。PU-GCN 仍使用固定 PU1K model-100，没有在 KITTI 重训 PU-GCN 网络。
- ModelNet40 完整分类、EAR 新 strict 全量、SPU-PMD 全量等未完成工作，未被计作已完成实验。

## 两个核心问题的答案

1. **点更多是否让检测更好？** 当前完整验证结果不支持；几何与检测器输入分布、原始观测保留都重要。
2. **一直低于 baseline 是否只因原先仅训练 3 epochs？** 延长训练确实有改善，但平台期后的差距仍存在，不能仅用训练轮数不足解释；也不能据此认定所有训练方案都无效。

证据：[PointRCNN 最终汇总](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/FINAL_RESULTS.md)、[PointRCNN 逐阶段证据](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/final_comparison.json)、[CenterPoint 收敛汇总](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_20260914/reports/centerpoint_convergence_summary.md)。

[完整时间线、问题处理、代码和历史结果记录](/home/ra87racy/reports/EXPERIMENT_WORK_RECORD_202602_TO_20260917_ZH.md)

