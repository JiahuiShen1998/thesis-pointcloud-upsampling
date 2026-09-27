# 点云上采样—下游检测实验总账（论文写作版）

原始总账整理时间：2026-08-04  
最新增补时间：2026-09-13（见第 18—25 节）  
主项目：KITTI 点云上采样 × PointRCNN / CenterPoint 下游 3D 检测  
用途：统一实验口径、结果、工作记录、代码修改、论文依据和可写结论

> **版本说明：** 第 0—17 节保留 2026-08-04 时点的完整历史快照，便于追踪当时尚未完成的 surface-c32、PDANS 和 detector-adaptation 状态；第 18—25 节加入 2026-08 至 2026-09 的后续实验，并作为“当前最新状态”。若旧状态与增补状态不一致，以第 18—25 节及其直接结果文件为准。

---

## 0. 先看这一页：目前可以写进论文的核心结论

### 0.1 研究问题

本项目真正研究的不是“能不能把点数机械地变成 4 倍”，而是：

> 在严格控制输出点数、观测点保留方式和检测器不变的条件下，学习式点云上采样能否恢复稀疏 LiDAR 的有效表面证据，并改善下游 3D 检测？

当前证据给出的回答是：

1. 严格 4× 点数不等于有效信息增加。  
   点数满足 4N 只是计数约束；决定检测效果的是新增点是否贴近真实表面、是否覆盖有用空间、是否避免重复和伪体素。

2. 在当前统一全量协议下，四种方法均不能恢复原始点云的 PointRCNN 检测性能。  
   Line A 中 PDANS 最稳健，但 Car Moderate 3D AP_R40 仍由原始点云的 82.2554 降至 67.2235。Line B 中所有方法也低于下采样基线。

3. 方法总体排序在两个检测器上具有一致性：  
   PDANS > PU-GCN > PU-EdgeFormer > 旧 PU-Net 管线。  
   但旧 PU-Net 结果包含明确的归一化/反归一化适配缺陷，不能当作 PU-Net 方法本身的公平上限。

4. 下游检测器存在有限输入预算，且会放大错误点的影响。  
   PointRCNN 固定抽取 16384 点；CenterPoint 每体素最多 5 点、测试最多 40000 体素。重复点、伪体素或异常空间分布会与真实观测竞争预算。

5. 旧 patch 提取器是主要工程性根因之一。  
   它使用粗空间分箱后连续切分 2048 点，并不保证是真正局部的表面 patch。Line B patch 的 XY 对角线中位数约 30.46 m、p90 约 124.10 m，严重违背通用 patch 上采样网络的训练假设。

6. “局部化”本身也不够。  
   第一版局部 cover-kNN patch 虽明显改善几何局部性，却通过重复填充凑满 2048 点；这使 PointRCNN 的有效体素覆盖骤降，而 CenterPoint 因体素化会吸收重复，反而明显改善。因此，局部、唯一支持和检测器预算必须同时控制。

7. 2026-08-04 时 surface-c32 仍只是几何候选；后续 PU-GCN 专项已推进到完整检测器适配矩阵。  
   多方法 surface-c32 的统一双检测器全量结论仍未形成；但 PU-GCN 后续使用改进的局部 patch 协议，已经完成 3712 帧 detector adaptation 与 3769 帧、20/20 arms 的 PointRCNN/CenterPoint 全量验证。最新结果见第 18—20 节。

### 0.2 论文最适合的主线

建议论文主线写成：

1. 建立严格 exact-4N、双输入线、冻结检测器的公平协议；
2. 证明“点数恢复”并不等于“检测恢复”；
3. 用点/体素预算、patch 局部性、重复填充和生成点表面一致性解释失败；
4. 通过 E3、观测点优先排序、剂量实验、匹配真实点控制和 patch/归一化 2×2 消融确认因果；
5. 提出“面向下游检测的上采样”需要满足的条件：局部、唯一、表面一致、预算感知和任务一致。

---

## 1. 项目范围与实验分支

| 分支 | 目标 | 当前状态 | 是否可作为论文主结果 |
|---|---|---:|---:|
| KITTI × PointRCNN | 检查 exact-4N 上采样对 Car 3D 检测的影响 | 全量主实验完成 | 是 |
| KITTI × CenterPoint | 用不同检测机制验证结论是否可泛化 | 全量主实验完成 | 是 |
| 比例/剂量实验 | 分离少量生成点增益与高比例生成点伤害 | 256 帧细粒度完成；全量 g10 完成 | 是，需标注规模 |
| patch/PU-Net 因果消融 | 定位 patch 非局部、重复填充和归一化缺陷 | 256 帧 2×2 完成 | 是，作为机制实验 |
| surface-c32 新候选 | 构造局部、唯一支持、零重复 patch | 几何与部分 256 生成完成；检测未跑 | 只能写为进行中 |
| ModelNet40 × PointNet++ | 检查分类任务上的 4× 上采样 | 仅 smoke；HPC 正式实验未提交 | 不能写结果 |
| TULIP / SPU-PMD / EAR | 方法可行性、早期工程探索 | 非统一协议或未进入最终方法集 | 只写背景/探索 |

---

## 2. 正式实验协议

### 2.1 数据集与划分

- 数据集：KITTI 3D Object Detection。
- 验证集：3769 帧。
- 正式对比以离线 KITTI AP_R40 为准。
- PointRCNN 正式主任务为 Car；CenterPoint 同时报告 Car、Pedestrian、Cyclist。
- AP 类型：BBox、BEV、3D；难度：Easy、Moderate、Hard。
- IoU 阈值：Car 为 0.7；Pedestrian/Cyclist 为 0.5。

KITTI 的原始论文和基准出处见第 11 节。

### 2.2 两条输入线

#### Line A：原始点云增密

- 输入：原始扫描 N 点。
- 输出：保留 N 个观测点，再生成 3N 个点。
- 最终点数：严格 4N。
- 研究问题：当原始传感器证据已经完整时，继续加入大量生成点是否帮助检测。

#### Line B：稀疏点云恢复

- 输入：确定性 4× 下采样，M = floor(N/4)。
- 输出：保留 M 个观测点，再生成 3M 个点。
- 最终点数：严格 4M，约等于原始扫描点数。
- 研究问题：学习式上采样能否把低分辨率输入恢复到原始传感器水平。

### 2.3 正式方法

1. PDANS；
2. PU-GCN；
3. PU-EdgeFormer；
4. PU-Net。

重要限定：

- 正式主表中的旧 PU-Net 使用了后来被确认有缺陷的适配器：没有按训练时单位球归一化，也没有正确逆变换。因此应标为“PU-Net（旧适配器/缺陷管线）”。
- TULIP、SPU-PMD、EAR 没有进入最终 exact-4N 四方法统一主表。

### 2.4 三层检测输入

#### E1：原始 exact-4N 输入

严格检查每帧点数、有限值和观测/生成组成，不做为检测器特制的再采样。

#### E2：PointRCNN 规范化 16384 输入

这是 PointRCNN 正式主结果：

1. 严格 FOV/距离过滤；
2. 0.1 m 体素代表点；
3. 深度区间为 [0,20)、[20,40)、[40,60)、[60,70.4] m；
4. 根据候选数量用 largest-remainder 分配配额；
5. 全过程确定性、无标签、无 AP 选择；
6. 输出固定为 16384 点。

#### E3：观测点优先的 32768 敏感性实验

- 先保留真实观测点，再用生成点填至 32768；
- 用于判断 E2 的 16384 点上限是否是主要损失来源；
- E3 是敏感性实验，不应写成新的主配置。

协议文件：

- [E1/E2/E3 根因报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_e3_final_root_cause_report.md)
- [exact-4N protocol.json](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/protocol.json)
- [E1/E2 AP 汇总](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_live_ap_summary.csv)
- [E3 AP 汇总](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e3_real_first_32768_live_ap_summary.csv)

### 2.5 PointRCNN 设置

- 配置：[tools/cfgs/default.yaml](/home/ra87racy/projects/baseline_detectors/PointRCNN/tools/cfgs/default.yaml)
- 权重：[tools/PointRCNN.pth](/home/ra87racy/projects/baseline_detectors/PointRCNN/tools/PointRCNN.pth)
- 权重冻结，只更换输入，不重新训练。
- RPN 输入点数：16384。
- PointNet++ MSG 半径：
  - [0.1, 0.5]；
  - [0.5, 1.0]；
  - [1.0, 2.0]；
  - [2.0, 4.0]。
- 每尺度邻域采样数：16/32。
- RPN 与 RCNN score 阈值：0.3。
- RCNN NMS 阈值：0.1。
- rect camera 坐标范围：
  - x：[-40, 40]；
  - y：[-1, 3]；
  - z：[0, 70.4]。
- 不使用反射强度作为网络输入特征。

### 2.6 CenterPoint 设置

- 配置：[centerpoint.yaml](/home/ra87racy/projects/baseline_detectors/PointRCNN/external/OpenPCDet/tools/cfgs/kitti_models/centerpoint.yaml)
- 数据配置：[kitti_dataset.yaml](/home/ra87racy/projects/baseline_detectors/PointRCNN/external/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml)
- 权重：[checkpoint_epoch_80.pth](/home/ra87racy/projects/baseline_detectors/PointRCNN/external/centerpoint_hpc_bundle_20260717/outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth)
- 在原始 KITTI 上训练 80 epoch；所有输入复用同一冻结权重。
- 结构：MeanVFE → VoxelResBackBone8x → HeightCompression → BaseBEVBackbone → CenterHead。
- 类别：Car、Pedestrian、Cyclist。
- 点云范围：[0,-40,-3,70.4,40,1]。
- 体素大小：0.05 × 0.05 × 0.1 m。
- 每体素最多 5 点。
- 测试最多 40000 体素。
- 测试不 shuffle。

---

## 3. 正式主结果

### 3.1 PointRCNN：全量 3769 帧 E2 AP_R40

下表是论文最重要的主结果。数值顺序为 Easy / Moderate / Hard。

| 输入 | BBox AP_R40 | BEV AP_R40 | 3D AP_R40 | Moderate 3D 相对基线 |
|---|---:|---:|---:|---:|
| 原始基线 | 99.3170 / 94.0842 / 90.1018 | 95.9809 / 88.9807 / 86.5942 | 92.2731 / 82.2554 / 77.9454 | — |
| 下采样基线 | 95.2406 / 79.9487 / 75.5846 | 91.0369 / 76.3979 / 72.0313 | 85.1766 / 65.7460 / 61.3818 | — |
| Line A PDANS | 93.3196 / 81.7462 / 76.8537 | 90.0035 / 78.0160 / 73.2078 | 84.6118 / 67.2235 / 62.3981 | -15.0319 |
| Line A PU-GCN | 92.4442 / 71.7663 / 67.0732 | 86.9259 / 66.7872 / 62.0159 | 80.1727 / 58.2759 / 53.4719 | -23.9796 |
| Line A PU-EdgeFormer | 81.9530 / 58.7956 / 52.2988 | 75.9674 / 53.8311 / 48.9408 | 65.6774 / 44.9596 / 40.1919 | -37.2958 |
| Line A PU-Net（旧缺陷管线） | 38.4606 / 23.8945 / 20.6926 | 28.9543 / 18.7383 / 16.8468 | 11.7639 / 8.3802 / 7.4407 | -73.8753 |
| Line B PDANS | 83.7454 / 60.3295 / 53.7948 | 76.6291 / 54.5678 / 48.1389 | 65.0123 / 44.0752 / 39.1729 | -21.6708 |
| Line B PU-GCN | 60.6102 / 38.6101 / 32.3995 | 57.2512 / 36.1932 / 31.4262 | 45.9303 / 28.6803 / 24.1310 | -37.0657 |
| Line B PU-EdgeFormer | 52.7497 / 33.4898 / 28.7087 | 43.6454 / 27.7381 / 23.3813 | 33.7635 / 20.5693 / 17.6153 | -45.1767 |
| Line B PU-Net（旧缺陷管线） | 33.6163 / 20.5057 / 18.4031 | 23.7717 / 15.5219 / 13.6207 | 12.4097 / 8.7807 / 7.7637 | -56.9653 |

可写结论：

- Line A 的所有方法都显著低于原始点云；添加 3N 生成点会伤害冻结检测器。
- Line B 的所有方法都低于下采样基线；这说明当前生成点不仅没有补回丢失证据，还干扰了已有稀疏观测。
- PDANS 在两条线上最好，但“相对最好”不等于“恢复成功”。
- 排名在两条线均大体一致，支持几何质量与检测性能之间存在稳定关系。

### 3.2 PointRCNN：E3 32768 观测点优先敏感性实验

| 输入 | 3D AP_R40 Easy / Moderate / Hard | Moderate：E3 - E2 |
|---|---:|---:|
| 原始基线 | 92.03 / 80.38 / 77.52 | -1.88 |
| 下采样基线 | 84.59 / 64.14 / 57.50 | -1.61 |
| Line A PDANS | 86.27 / 70.06 / 65.27 | +2.84 |
| Line A PU-GCN | 85.19 / 64.95 / 58.66 | +6.68 |
| Line A PU-EdgeFormer | 71.54 / 51.63 / 45.65 | +6.67 |
| Line A PU-Net（旧缺陷管线） | 27.92 / 18.93 / 17.88 | +10.55 |
| Line B PDANS | 64.92 / 44.04 / 39.16 | -0.04 |
| Line B PU-GCN | 45.09 / 28.28 / 23.83 | -0.40 |
| Line B PU-EdgeFormer | 32.50 / 20.03 / 16.20 | -0.54 |
| Line B PU-Net（旧缺陷管线） | 10.99 / 7.48 / 7.19 | -1.30 |

解释：

- Line A 在扩大预算并优先真实点后有所恢复，说明 16384 点竞争确实造成损失。
- 但即使 E3，所有方法仍远低于原始基线，因此输入上限不是唯一或主导根因。
- Line B 几乎不改善，说明其失败主要来自生成几何，而不是 16384 cap。

### 3.3 CenterPoint：全量主结果（Moderate 3D AP_R40）

| 输入 | Car | Pedestrian | Cyclist |
|---|---:|---:|---:|
| 原始基线 | 79.2773 | 50.6533 | 64.6054 |
| Line A PDANS | 64.3574 | 45.1910 | 45.8953 |
| Line A PU-GCN | 58.8955 | 44.1470 | 44.3082 |
| Line A PU-EdgeFormer | 45.7342 | 36.8205 | 35.2235 |
| Line A PU-Net（旧缺陷管线） | 10.7512 | 19.4930 | 13.1928 |
| 下采样基线 | 64.5979 | 24.5690 | 15.4577 |
| Line B PDANS | 46.7512 | 28.1790 | 15.5186 |
| Line B PU-GCN | 39.0714 | 24.9302 | 16.8458 |
| Line B PU-EdgeFormer | 24.7476 | 15.5896 | 7.0374 |
| Line B PU-Net（旧缺陷管线） | 11.7178 | 11.9174 | 5.0172 |

解释：

- Car 上重现 PointRCNN 的总体排序和失败趋势，说明结论并非单一检测器偶然。
- Line B 的 PDANS/PU-GCN 对 Pedestrian/Cyclist 有极小局部恢复，但 Car 明显下降；不能据此宣称总体恢复成功。
- 跨类别的差异提示：细小目标可能从少量局部填充获益，但大量不可靠点仍会破坏主任务。

完整 CenterPoint 报告：

- [Observed-first 全量报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_exact4n_reconstructed_order_safe_20260729/CENTERPOINT_OBSERVED_FIRST_FULL_REPORT.md)
- [CenterPoint 全 AP 表](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_exact4n_reconstructed_order_safe_20260729/full_ap_summary.csv)

---

## 4. 根因与机制证据

### 4.1 旧 patch 提取器不是局部表面 patch

旧流程的核心是：

1. 对场景做粗空间分箱；
2. 在分箱结果中按顺序切分 2048 点；
3. 把这些点作为通用物体级上采样网络的输入 patch。

问题在于，连续 2048 行不等于空间上紧凑、同一表面或同一对象。测得：

- Line B patch XY 对角线中位数约 30.46 m；
- p90 约 124.10 m。

因此，一个 patch 可能混入地面、车辆、建筑、植被和远距离结构。预训练在规范物体表面 patch 上的网络会把这种混合输入当成一个局部形状进行生成，产生跨表面插值和无支撑点。

主根因报告：

- [论文级根因报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/dual_detector_three_frame_root_cause_20260730/THESIS_ROOT_CAUSE_REPORT_ZH.md)
- [双检测器完整 AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/dual_detector_three_frame_root_cause_20260730/analysis/all_detector_ap_r40_bbox_bev_3d.csv)
- [便携分析包说明](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/current_task_visualization_analysis_portable_20260731/README_ZH.md)

### 4.2 真正有效的是新增“表面证据”，而非新增点数

几何分析显示，当前方法虽然生成大量点，但新增的真实支持体素有限，同时产生大量参考点云中不存在的额外体素。可以把机制概括为：

> 小量正确的表面恢复 + 大量冗余/错位体素 + 检测器预算竞争 → AP 下降。

建议论文中重点报告以下几类指标：

- generated-to-real 最近邻距离；
- reference-to-output recall；
- generated voxel precision；
- extra voxel fraction；
- occupied voxel 数；
- 局部 PCA 平面距离；
- patch XY 直径和对象纯度；
- 每点/每体素重复密度。

### 4.3 PointRCNN 与 CenterPoint 为什么反应不同

#### PointRCNN

- 直接在点集上做 PointNet++ 局部聚合；
- 固定 16384 输入槽位；
- 高度重复或簇状点会占用原始点级采样预算；
- 近重复点不一定增加邻域信息，反而减少覆盖的独立空间位置。

#### CenterPoint

- 先把点云量化成体素；
- 同一体素中的重复点会被 MeanVFE 聚合；
- 重复点的危害可部分被体素化吸收；
- 但大量新体素会触发 40000 体素上限，并与真实体素竞争。

这解释了为什么同一版局部 patch 可能改善 CenterPoint，却显著伤害 PointRCNN。

### 4.4 体素预算证据

在 Line A 的 32 帧统计中，进入 CenterPoint 测试 cap 前的体素数中位数与 40000 cap 命中情况如下：

| 输入 | 体素数中位数 | 超过/命中 40000 cap |
|---|---:|---:|
| 原始点云 | 14944 | 0 / 32 |
| PDANS | 36913 | 6 / 32 |
| PU-GCN | 45108 | 29 / 32 |
| PU-EdgeFormer | 46182 | 29 / 32 |
| PU-Net（旧缺陷管线） | 55384 | 32 / 32 |

Line B 未命中 40000 cap，因此：

- Line A 中体素预算是明确的放大器；
- Line B 的失败不能归因于 CenterPoint cap，仍指向生成几何质量。

### 4.5 观测点优先排序实验

该实验保持每帧点的多重集合完全相同，只把 N 个观测点放在 3N 个生成点之前。8 个方法/输入组合 × 3769 帧全部通过同集合审计。

相对原重构顺序的 Moderate 3D AP_R40 平均变化：

- Line A PDANS：+0.0869；
- Line A PU-GCN：+0.8939；
- Line A PU-EdgeFormer：+0.7919；
- Line A PU-Net（旧缺陷管线）：+5.8776；
- Line B：仅 -0.0018 到 +0.0311。

结论：

- 顺序和预算竞争是真实存在的；
- 但除极端失败的 PU-Net 外，收益较小；
- 它是次要机制，不足以解释主 AP 降幅。

### 4.6 GT 转移分析

相对于对应基线，基线 TP 变为 FN 的 Car 比例：

| 检测器/输入线 | PDANS | PU-GCN | PU-EdgeFormer | PU-Net 旧管线 |
|---|---:|---:|---:|---:|
| PointRCNN Line A | 23.1% | 32.7% | 47.9% | 80.9% |
| PointRCNN Line B | 35.4% | 55.4% | 67.0% | 76.5% |
| CenterPoint Line A | 20.6% | 25.9% | 39.3% | 70.8% |
| CenterPoint Line B | 27.6% | 37.4% | 55.3% | 67.0% |

代表帧：

- 000590；
- 005625；
- 006682。

已完成 3 帧 × 2 条线 × 4 方法 × 2 检测器 = 48 个案例。确有少量恢复案例，但损失明显更多。GT 转移和可视化应作为 AP 的机制支撑，不能替代正式 AP。

---

## 5. 剂量、控制实验与消融

### 5.1 生成点比例实验

报告：

- [联合剂量分析](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_x4_detector_recovery_combined_analysis_20260725/combined_analysis_report.md)
- [完整 BEV/3D 对照表](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_x4_detector_recovery_combined_analysis_20260725/complete_bev_3d_comparison.md)

#### 256 帧细粒度替换比例

比例：g2.5、g5、g7.5、g10；所有 32/32 组合通过审计。

Line A Car Moderate 3D AP_R40：

| 方法 | g2.5 | g5 | g7.5 | g10 | 最佳相对原始基线 82.66 |
|---|---:|---:|---:|---:|---:|
| PDANS | 85.10 | 82.37 | 82.25 | 82.42 | +2.43 |
| PU-GCN | 82.69 | 82.17 | 81.88 | 79.65 | +0.03 |
| PU-EdgeFormer | 81.97 | 82.08 | 79.39 | 77.04 | -0.58 |
| PU-Net 旧管线 | 79.10 | 77.82 | 75.64 | 71.67 | -3.56 |

Line B Car Moderate 3D AP_R40：

| 方法 | g2.5 | g5 | g7.5 | g10 | 下采样基线 |
|---|---:|---:|---:|---:|---:|
| PDANS | 66.40 | 63.26 | 61.34 | 60.86 | 68.29 |
| PU-GCN | 65.54 | 61.82 | 60.87 | 56.16 | 68.29 |
| PU-EdgeFormer | 63.23 | 60.32 | 53.96 | 53.20 | 68.29 |
| PU-Net 旧管线 | 62.60 | 57.35 | 51.43 | 47.59 | 68.29 |

匹配真实点控制：

- 原始点云 real-control c2.5 = 84.42；
- PDANS g2.5 = 85.10；
- 因此 PDANS 相对相同点数的真实填充控制只高约 0.68 AP；
- 且该结果来自 256 帧，必须全量确认，不能直接写成稳定提升。

#### 256 帧粗比例

g10、g15、g25、g35、g40、g50 共 48/48 组合通过。总体趋势为：生成点比例越高，检测性能越差。

#### 全量 3769 帧 g10

Car Moderate 3D AP_R40：

| 输入线 | PDANS | PU-GCN | PU-EdgeFormer | PU-Net 旧管线 |
|---|---:|---:|---:|---:|
| Line A | 79.951 | 79.478 | 76.433 | 72.371 |
| Line B | 59.048 | 55.954 | 52.415 | 45.700 |

全量 g25/g50 尚未完成。

可写结论：

- 生成点对检测呈明显剂量效应；
- 少量高质量生成点在 Line A 可能帮助或保持性能；
- 当生成点比例提高时，冗余、错位和预算竞争累积，性能快速下降；
- Line B 更难，因为稀疏输入本身缺少可供网络可靠重建的局部证据。

### 5.2 sampler-safe 修复

旧 PointRCNN 采样逻辑在 far points ≥ 16384 时会产生负采样数，并在 000378 等帧崩溃。

修复：

- 当远点已经达到 16384 时，从全部候选中无放回抽取 16384 点；
- 当需要补点但目标补点数超过可用点数时，允许有放回采样；
- 记录 fallback 触发情况。

结果：

- 所有此前失败的 Line A 组合都完成 3769/3769；
- fallback 每方法仅约 3–12/3769 帧，小于 0.4%；
- 因此该 bug 解释的是“为什么运行崩溃”，不是“为什么 AP 下降”。

报告：

- [sampler-safe 最终报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_pointrcnn_eval_line_a_sampler_safe_v2_20260717_235037/reports/final_line_a_sampler_safe_v2_report.md)

### 5.3 第一版局部 patch：几何改善，但重复填充伤害 PointRCNN

候选：fps_ball_cover_knn_v3。

设置：

- Line A 主半径 2 m；
- cover 半径 6 m；
- cover_min = 32；
- 唯一 kNN floor = 256；
- patch overlap ratio = 3；
- 不足 2048 点时确定性重复填充。

几何 20 帧：

- patch XY p50/p90：旧 9.439/84.097 m → 新 5.334/16.443 m；
- 源点覆盖率接近 1。

但 frame 000001：

- 277 个 patch；
- 覆盖率 99.9867%；
- 每源点平均进入 4.72 个 patch；
- patch 唯一源点中位数仅 464/2048；
- p10 为 256/2048；
- 重复行约 59.4%。

256 帧检测变化：

| 检测器 | 指标 | 旧 patch | 局部重复 patch | 变化 |
|---|---|---:|---:|---:|
| PointRCNN | Car Moderate BEV | 70.3912 | 51.6618 | -18.7293 |
| PointRCNN | Car Moderate 3D | 57.1829 | 43.6976 | -13.4853 |
| CenterPoint | Car Moderate BEV | 70.6567 | 86.4220 | +15.7653 |
| CenterPoint | Car Moderate 3D | 61.1172 | 77.4150 | +16.2978 |
| CenterPoint | Pedestrian Moderate 3D | — | — | +1.3431 |
| CenterPoint | Cyclist Moderate 3D | — | — | +23.4861 |

PointRCNN 机制指标：

- occupied voxels 中位数：11584.5 → 2124.5，下降 81.7%；
- 最密 10% 体素的点占比：27.23% → 87.85%。

结论：

- patch 更局部并不自动带来下游收益；
- 重复凑数会让点级检测器的有限槽位被近重复点占据；
- CenterPoint 的体素化吸收了一部分重复，因此表现方向相反；
- 该候选不应扩展到全量。

报告：

- [PDANS 双检测器分歧根因](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/analysis/pdans_old_vs_cover_knn_v3/PDANS_DETECTOR_DISAGREEMENT_ROOT_CAUSE.md)
- [patch/PU-Net 因果协议](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/PROTOCOL.md)

### 5.4 PU-Net 归一化 × patch 的 2×2 消融

四个条件：

- A：错误归一化 + 旧 patch；
- B：单位球归一化/逆变换 + 旧 patch；
- C：错误归一化 + 局部重复 patch；
- D：单位球归一化/逆变换 + 局部重复 patch。

256 帧 Car Moderate 3D AP_R40：

| 检测器 | 基线 | A | B | C | D | D - A | D - 基线 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PointRCNN | 81.3033 | 7.9823 | 18.2742 | 6.3008 | 43.5424 | +35.5602 | -37.7609 |
| CenterPoint | 79.8368 | 15.9788 | 41.2227 | 16.4119 | 46.2047 | +30.2259 | -33.6321 |

结论：

- 正确归一化/逆变换是 PU-Net 的首要修复；
- 正确归一化和局部 patch 存在正交互；
- 即使 D 大幅优于 A，仍远低于检测基线；
- 旧 PU-Net 主结果应被解释为管线失败证据，而不是方法本身性能；
- D 也不值得直接扩展到 3769 帧，需先解决重复和源点支持问题。

报告：

- [PU-Net 2×2 最终决策](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/punet256_2x2/reports/FINAL_DECISION_CN.md)
- [PU-Net 2×2 双检测器效果](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/punet256_2x2/reports/PUNET_2X2_DETECTOR_EFFECTS.md)

### 5.5 最新 surface-c32：唯一支持、零重复

选择的候选：surface_cover_exact_pr1_c32。

设置：

- patch 大小：2048；
- 每个 patch 必须有 2048 个唯一源点；
- 零重复填充；
- primary patch_num_ratio = 1；
- 主半径 2 m；
- cover 半径 6 m；
- cover_min = 32。

20 帧 patch 指标：

| 指标 | 结果 |
|---|---:|
| patch 总数 | 2025 |
| 覆盖率 p50 | 0.999975 |
| 覆盖率最小值 | 0.999638 |
| membership mean p50 | 1.739 |
| unique source p10 | 2048 |
| 重复比例 | 0 |
| XY 直径 p50 | 5.332 m |
| XY 直径 p90 | 24.180 m |
| 多对象 patch 比例中位数 | 0.00476 |

20 帧方法输出的 generated-to-real 最近邻 p90 中位数：

| 方法 | 旧 patch | surface-c32 |
|---|---:|---:|
| PU-GCN | 0.1557 | 0.1127 |
| PU-EdgeFormer | 0.2236 | 0.0991 |
| PDANS | 0.1399 | 0.0974 |
| PU-Net（修复归一化） | 0.7920 | 0.2759 |

所有方法的 1 mm 近似唯一率约 99.97%–100%，不再存在第一版局部 patch 的重复塌缩。

当前状态（2026-08-04）：

| 方法 | 256 帧生成 | exact-4N 检查 | PointRCNN AP | CenterPoint AP |
|---|---:|---:|---:|---:|
| PU-GCN | 完成 | 通过 | 未运行 | 未运行 |
| PU-EdgeFormer | 完成 | 通过 | 未运行 | 未运行 |
| PU-Net 修复版 | 完成 | 通过 | 未运行 | 未运行 |
| PDANS | 中断：231/256 merged raw；232 个日志 | 无最终 bin | 未运行 | 未运行 |

PDANS 状态文件仍显示 RUNNING，但当前没有匹配进程/tmux；最后一帧 006812 的日志停在 DDIM 过程中。因此这是陈旧状态，不是仍在运行。

几何报告：

- [surface 搜索 summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/geometry20_surface_search_v1/reports/summary.json)
- [方法输出 summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/geometry20_surface_search_v1/reports/method_outputs/summary.json)
- [256 生成状态](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/surface_candidate256_v1/sequence_state.json)

---

## 6. 历史实验：保留作工作记录，但不要混入正式主表

### 6.1 早期 PointRCNN / EAR / PU-Net

历史报告：

- [WORKLOG_2026-05-04](/home/ra87racy/projects/baseline_detectors/PointRCNN/WORKLOG_2026-05-04.md)
- [PointRCNN baseline report](/home/ra87racy/projects/baseline_detectors/PointRCNN/POINTRCNN_BASELINE_REPORT.md)
- [PointRCNN EAR report](/home/ra87racy/projects/baseline_detectors/PointRCNN/POINTRCNN_EAR_REPORT.md)
- [PU-Net integration report](/home/ra87racy/projects/baseline_detectors/PointRCNN/PU_NET_INTEGRATION_REPORT.md)
- [KITTI upsampling comparison](/home/ra87racy/projects/baseline_detectors/PointRCNN/KITTI_UPSAMPLING_COMPARISON.md)

早期代表结果：

- baseline 3D AP 约 89.19 / 78.85 / 77.91；
- EAR 约 88.56 / 77.96 / 76.78；
- 早期 PU-Net 约 0.20 / 1.14 / 1.14。

这些实验存在 x2、整场输入、旧适配器、旧采样或其他非统一设置，只能作为工程探索记录。

### 6.2 2026-05-16 周报

[weekly_progress_report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/weekly_report_20260516_full_comparison/weekly_progress_report.md)

其中 Moderate 3D AP：

- original baseline：77.93；
- 50% downsample：76.99；
- EAR：73.89；
- PU-Net：35.26；
- PU-GCN：50.65。

但 PU-GCN Line B 使用 RPN 26000 和 safe sampling，与后来的统一 E2 16384 不兼容。不能与第 3 节主表直接比较。

### 6.3 旧输出实际倍率审计

[upsampling_ratio_audit](/home/ra87racy/reports/upsampling_ratio_audit.md)

发现：

- EAR 实际约 1.008×；
- 旧 PU-Net 倍率不一致；
- PU-GCN、PDANS 部分输出被 100k cap 截断；
- TULIP 是 range-image 垂直 4×，但投影回 xyz 后点数可能少于输入。

这直接推动了后续 strict exact-4N 协议。论文可把它写成“协议设计动机”，但不要把这些旧结果视作公平方法对比。

### 6.4 TULIP、SPU-PMD、PU-EdgeFormer、PDANS 可行性记录

- [TULIP/PDANS/SPU-PMD feasibility](/home/ra87racy/projects/experiments/method_feasibility_check/TULIP_PDANS_SPUPMD_FEASIBILITY_REPORT.md)
- [TULIP/PDANS/SPU-PMD reproduction](/home/ra87racy/projects/experiments/method_repro_check/TULIP_PDANS_SPUPMD_REPRO_REPORT.md)
- [PU-EdgeFormer point-cloud-only report](/home/ra87racy/projects/experiments/puef_pointcloud_only/PU_EDGEFORMER_POINTCLOUD_ONLY_REPORT.md)

历史过程：

- TULIP 作为原生 LiDAR range-image 方法曾被列为优先候选；
- PDANS 的 PUGAN/PU1K checkpoint 后来确认可用，每个约 1.7 GB；
- SPU-PMD 完成可行性/smoke，但未进入最终四方法正式表；
- PU-EdgeFormer 初期受 checkpoint 和 CUDA10 自定义算子阻塞，后来通过复用 PU-GCN ops/预训练环境实现正式运行。

EAR 是自定义 EAR-style 工程方法，本地没有可追溯论文来源。论文中不要把它伪装成已有正式文献方法。

---

## 7. ModelNet40 / PointNet++ 分支

主报告：

- [ModelNet40 x4 final protocol](/home/ra87racy/projects/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_final_protocol_report.md)

已完成：

- PU-GCN 本地能力准备完成；
- Line A：1024 → 4096；
- Line B：256 → 1024；
- 每条线各 8 个 smoke 样本通过；
- 点数和 NaN/Inf 检查通过。

未完成：

- 12311 个样本/线的全量生成没有提交到 HPC；
- PU-EdgeFormer checkpoint 尚未完成该分支接入；
- PointNet++ Line B 的 5 个 smoke 分支没有提交；
- 没有分类准确率、几何质量指标或完整训练结果。

阻塞原因：

- 当前 agent 主机没有 HPC 挂载；
- 没有可用 SLURM 环境；
- 所需 job 文件不在当前主机可执行上下文。

论文边界：

> ModelNet40 目前只能写成“协议和 smoke 验证已准备”，不能写成已有分类结果。

---

## 8. 已进行的代码和管线修改

本节区分“当前工作树可核实修改”和“历史报告记录的修改”。

### 8.1 PointRCNN 当前工作树可核实修改

当前 tracked diff 约 50 insertions / 9 deletions。

1. [lib/config.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/lib/config.py)  
   yaml.load 改为 yaml.safe_load，适配新 PyYAML 并降低不安全反序列化风险。

2. [lib/datasets/kitti_dataset.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/lib/datasets/kitti_dataset.py)  
   增加 NFS 稳健读取：
   - 最多 20 次重试；
   - 检查字节数和 short read；
   - 每次等待 0.1 s。

3. [lib/datasets/kitti_rcnn_dataset.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/lib/datasets/kitti_rcnn_dataset.py)  
   增加 sampler-safe：
   - far points ≥ 16384 时安全无放回采样；
   - 补点不足时允许 replacement；
   - 避免负采样数和崩溃。

4. [tools/eval_rcnn.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/tools/eval_rcnn.py)  
   args.test 模式下即使 split=val 也跳过内置 AP，允许之后用统一离线评估恢复 AP。

### 8.2 新增关键脚本

核心 wrapper：

- [kitti_patch_extractor.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/kitti_patch_extractor.py)
- [tf_pugcn_family_patch_infer.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/tf_pugcn_family_patch_infer.py)
- [tf_punet_patch_infer.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/tf_punet_patch_infer.py)
- [pdans_patch_infer.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/pdans_patch_infer.py)
- [strict_x4_from_merged_raw.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/strict_x4_from_merged_raw.py)

实验编排：

- [run_patch_causal_upsampling.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_patch_causal_upsampling.py)
- [run_surface_candidate_256_sequence.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_surface_candidate_256_sequence.py)
- [prepare_pointrcnn_e1_e2_inputs.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/prepare_pointrcnn_e1_e2_inputs.py)
- [run_pointrcnn_e1_e2_evals.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pointrcnn_e1_e2_evals.py)
- [run_centerpoint_exact4n_observed_first.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_centerpoint_exact4n_observed_first.py)
- [run_centerpoint_exact4n_local_staged.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_centerpoint_exact4n_local_staged.py)

关键功能：

- exact-4N 点数强校验；
- 保留观测点与生成点 provenance；
- 最近邻强度赋值；
- 多种 patch 模式；
- 局部 cover 补充；
- 单位球归一化/逆变换；
- 方法特定推理环境；
- E1/E2/E3 和双检测器自动评估；
- 几何、体素、GT 转移和案例可视化分析。

### 8.3 PU-Net 关键修复

历史集成报告：

- [PU_NET_INTEGRATION_REPORT](/home/ra87racy/projects/baseline_detectors/PointRCNN/PU_NET_INTEGRATION_REPORT.md)

历史修改包括：

- Python 2 → Python 3；
- 增加 data_folder 参数；
- 修复整数除法；
- 增加 CPU fallback；
- 更新自定义 op 编译脚本。

当前最关键的因果修复位于：

- [tf_punet_patch_infer.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/tf_punet_patch_infer.py)

unit_sphere_v1 流程：

1. patch 减去质心；
2. 按最大半径缩放到单位球；
3. 在训练分布对应的尺度中推理；
4. 输出乘回尺度并加回质心。

### 8.4 PDANS 修改

- pointnet2/util.py：设备感知的 torch.normal/tensor 创建，移除硬编码 .cuda()；
- pointnet2_utils.py：用 setdefault 设置 TORCH_CUDA_ARCH_LIST=7.5；
- checkpoint 和编译产物为本地未跟踪资产。

### 8.5 PU-GCN 修改

tf_ops/compile.sh：

- 移除硬编码 CUDA 10；
- 支持 CUDA_HOME/CUDA_PATH；
- 支持 conda libcudart；
- 使用 TensorFlow 实际 compile/link flags；
- 调整 PATH/LD_LIBRARY_PATH。

### 8.6 OpenPCDet / CenterPoint 修改

- KITTI-only 环境下将 Argo2 数据集注册改为可选，避免强制依赖 av2；
- 对受信任 checkpoint metadata 显式使用 weights_only=False，适配 PyTorch 2.6+；
- 新增 create_kitti_val_infos_only.py。

### 8.7 patch 提取器迭代

已实现/尝试：

- deterministic_spatial_chunk；
- FPS + kNN；
- FPS + ball query；
- cover supplement；
- fps_ball_cover_knn_v3；
- surface-c32（通过唯一 2048 支持和零重复约束）。

设计演化：

> 粗分箱连续切片 → 局部覆盖但重复严重 → 局部、全覆盖、唯一支持、零重复。

---

## 9. 工作记录时间线

### 2026-05：基线复现与早期方法接入

- 复现 PointRCNN baseline；
- 建立 EAR-style、PU-Net、PU-GCN 等早期输入；
- 修复 Python/CUDA/TensorFlow 自定义 op 兼容性；
- 完成早期全量或周报级比较；
- 发现 PU-Net 输出异常和采样崩溃问题。

### 2026-06：方法可行性与 HPC 迁移

- 检查 TULIP、PDANS、SPU-PMD、PU-EdgeFormer；
- 梳理 checkpoint、算子和环境阻塞；
- 统计 10 套旧 final variant，每套 3769 帧，总量约 42 GB；
- 制定实验室到 HPC 迁移计划；
- 2026-06-15 SSH dry-run 因认证失败，未真正传输。

迁移报告：

- [lab_to_hpc_transfer_plan](/home/ra87racy/projects/kitti_upsampling_lab_to_hpc_migration/reports/lab_to_hpc_transfer_plan.md)

### 2026-07 上旬：倍率审计与统一协议

- 发现旧输出并非严格 4×；
- 定义 Line A / Line B；
- 将输出统一为 observed + generated 的 exact-4N；
- 引入 deterministic、label-free、AP-free 检测输入选择；
- 完成 sampler-safe 修复。

### 2026-07 中旬：PointRCNN E1/E2/E3 全量

- 完成四方法 × 两条线 × 3769 帧；
- 生成 E1/E2 正式主表；
- 运行 E3 32768 观测点优先敏感性实验；
- 确认 PointRCNN cap 是放大因素而非唯一根因。

### 2026-07 下旬：CenterPoint 与跨检测器验证

- 接入 OpenPCDet CenterPoint；
- 固定同一 80 epoch checkpoint；
- 完成 exact-4N 双线全量；
- 进行 observed-first 同集合排序实验；
- 统计体素 cap、GT 转移和代表案例。

### 2026-07-25 起：剂量与匹配控制

- 运行 256 帧 g2.5–g50；
- 运行 matched observed-fill 控制；
- 完成全量 g10；
- 发现少量 PDANS 在 Line A 有局部收益，但高比例稳定恶化。

### 2026-07-30 至 2026-08-04：根因、patch 与 PU-Net 因果实验

- 定位旧 patch 非局部；
- 构造 fps_ball_cover_knn_v3；
- 发现重复填充造成 PointRCNN/CenterPoint 分歧；
- 完成 PU-Net 归一化 × patch 2×2；
- 设计 surface-c32 唯一支持候选；
- 完成 20 帧几何和方法输出验证；
- 完成三方法 256 帧生成；
- PDANS 256 帧在 231 个 merged raw 后中断；
- 尚未运行 surface-c32 的双检测器 AP。

---

## 10. 报告、结果和日志索引

### 10.1 写论文优先阅读

1. [论文级根因报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/dual_detector_three_frame_root_cause_20260730/THESIS_ROOT_CAUSE_REPORT_ZH.md)
2. [E1/E2/E3 根因报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_e3_final_root_cause_report.md)
3. [联合剂量分析](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_x4_detector_recovery_combined_analysis_20260725/combined_analysis_report.md)
4. [CenterPoint observed-first 全量报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_exact4n_reconstructed_order_safe_20260729/CENTERPOINT_OBSERVED_FIRST_FULL_REPORT.md)
5. [patch/PU-Net 因果协议](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/PROTOCOL.md)
6. [PDANS 检测器分歧根因](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/analysis/pdans_old_vs_cover_knn_v3/PDANS_DETECTOR_DISAGREEMENT_ROOT_CAUSE.md)
7. [PU-Net 2×2 最终决策](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/punet256_2x2/reports/FINAL_DECISION_CN.md)

### 10.2 原始数字

- [PointRCNN E1/E2 AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_live_ap_summary.csv)
- [PointRCNN E3 AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e3_real_first_32768_live_ap_summary.csv)
- [双检测器所有 AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/dual_detector_three_frame_root_cause_20260730/analysis/all_detector_ap_r40_bbox_bev_3d.csv)
- [CenterPoint full AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_exact4n_reconstructed_order_safe_20260729/full_ap_summary.csv)
- [完整剂量 BEV/3D 表](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_x4_detector_recovery_combined_analysis_20260725/complete_bev_3d_comparison.md)

### 10.3 历史和工程记录

- [2026-05-04 工作日志](/home/ra87racy/projects/baseline_detectors/PointRCNN/WORKLOG_2026-05-04.md)
- [2026-05-16 周报](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/weekly_report_20260516_full_comparison/weekly_progress_report.md)
- [PU-Net 集成报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/PU_NET_INTEGRATION_REPORT.md)
- [倍率审计](/home/ra87racy/reports/upsampling_ratio_audit.md)
- [HPC 迁移计划](/home/ra87racy/projects/kitti_upsampling_lab_to_hpc_migration/reports/lab_to_hpc_transfer_plan.md)
- [ModelNet40 协议状态](/home/ra87racy/projects/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_final_protocol_report.md)

---

## 11. 使用的论文、方法原理与本项目中的作用

### 11.1 数据集与检测器

1. Andreas Geiger, Philip Lenz, Raquel Urtasun.  
   “Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.” CVPR 2012.  
   作用：KITTI 数据集和 3D 检测基准来源。  
   原文：[KITTI/CVPR PDF](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Geiger2012CVPR.pdf)

2. Shaoshuai Shi, Xiaogang Wang, Hongsheng Li.  
   “PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud.” CVPR 2019.  
   原理：第一阶段直接在原始点上做前景分割和 bottom-up proposal；第二阶段在 canonical 坐标中细化 3D box。  
   本项目作用：主要点式下游检测器；其 16384 点预算和 PointNet++ 局部聚合用于分析生成点竞争。  
   原文：[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.html)

3. Tianwei Yin, Xingyi Zhou, Philipp Krähenbühl.  
   “Center-Based 3D Object Detection and Tracking.” CVPR 2021.  
   原理：把 3D 目标表示为中心点，用 BEV keypoint head 检测中心并回归尺寸、朝向等属性。  
   本项目作用：体素/BEV 检测器，用来验证根因是否跨检测架构成立。  
   原文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html)

4. Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.  
   “PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.” NeurIPS 2017.  
   原理：在嵌套局部邻域上递归应用 PointNet，以多尺度方式学习点集局部结构。  
   本项目作用：PointRCNN backbone 的局部聚合基础，也是 ModelNet40 分类分支的下游网络。  
   原文：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html)

### 11.2 正式上采样方法

5. Lequan Yu, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, Pheng-Ann Heng.  
   “PU-Net: Point Cloud Upsampling Network.” CVPR 2018.  
   原理：学习多层逐点特征，用多分支卷积在特征空间扩张点数，并以表面贴合与均匀分布联合损失训练 patch-level 网络。  
   本项目作用：最早的学习式上采样基线；其 patch-level 和归一化假设直接暴露了当前整场景适配问题。  
   原文：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.html)

6. Guocheng Qian, Abdulellah Abualshour, Guohao Li, Ali Thabet, Bernard Ghanem.  
   “PU-GCN: Point Cloud Upsampling Using Graph Convolutional Networks.” CVPR 2021.  
   原理：Inception DenseGCN 做多尺度特征提取，NodeShuffle 用图卷积邻域信息扩张点。  
   本项目作用：图卷积型正式方法，性能通常仅次于 PDANS。  
   原文：[CVF Open Access PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_PU-GCN_Point_Cloud_Upsampling_Using_Graph_Convolutional_Networks_CVPR_2021_paper.pdf)

7. Dohoon Kim, Minwoo Shin, Joonki Paik.  
   “PU-EdgeFormer: Edge Transformer for Dense Prediction in Point Cloud Upsampling.” ICASSP 2023 / arXiv:2305.01148.  
   原理：结合 EdgeConv 图卷积与多头自注意力，同时建模局部几何和全局结构。  
   本项目作用：Transformer/graph 混合型正式方法。  
   原文：[arXiv](https://arxiv.org/abs/2305.01148)

8. Boqian Zhang, Shen Yang, Hao Chen, Chao Yang, Jing Jia, Guang Jiang.  
   “Point Cloud Upsampling Using Conditional Diffusion Module with Adaptive Noise Suppression.” CVPR 2025.  
   原理：条件扩散生成密集点；ANS 根据点与邻域关系进行自适应噪声抑制；TreeTrans 融合跨层特征。  
   本项目作用：正式方法中最稳健，尤其在噪声和分布偏移下总体最好。  
   原文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Point_Cloud_Upsampling_Using_Conditional_Diffusion_Module_with_Adaptive_Noise_CVPR_2025_paper.html)

### 11.3 可行性/背景方法

9. Bin Yang, Patrick Pfreundschuh, Roland Siegwart, Marco Hutter, Peyman Moghadam, Vaishakh Patil.  
   “TULIP: Transformer for Upsampling of LiDAR Point Clouds.” CVPR 2024.  
   原理：把 LiDAR 投影为 range image，并修改 Swin Transformer 的 patch/window 几何以匹配 LiDAR 范围图特性。  
   本项目作用：原生 LiDAR 方法候选；旧输出存在 xyz 点数不等于严格 4× 的协议问题，未进入最终四方法主表。  
   原文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_TULIP_Transformer_for_Upsampling_of_LiDAR_Point_Clouds_CVPR_2024_paper.html)

10. Yanzhe Liu, Rong Chen, Yushi Li, Yixi Li, Xuehou Tan.  
    “SPU-PMD: Self-Supervised Point Cloud Upsampling via Progressive Mesh Deformation.” CVPR 2024.  
    原理：把上采样视作可变形拓扑上的密化，通过粗网格插值和多阶段 mesh deformation 逐步优化结构。  
    本项目作用：完成可行性/smoke，未进入正式主表。  
    原文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_SPU-PMD_Self-Supervised_Point_Cloud_Upsampling_via_Progressive_Mesh_Deformation_CVPR_2024_paper.html)

---

## 12. 本项目使用的核心原理

### 12.1 公平比较原则

1. 输出点数必须严格相同；
2. 观测点必须保留并记录 provenance；
3. 只改变输入点云，冻结检测器与权重；
4. 所有采样确定性；
5. 不用 GT label 或 AP 选择输入；
6. 先做小规模机制筛选，再做全量验证；
7. 区分工程 bug、协议影响和方法本身能力。

### 12.2 patch 局部几何

通用点云上采样网络通常假设输入是规范尺度的局部表面 patch。因此需要：

- FPS 保证中心覆盖；
- kNN 保证固定邻域数；
- ball query 保证空间局部性；
- cover supplement 保证未覆盖源点被纳入；
- 唯一源点约束避免重复凑数；
- overlap 控制 patch 边界稳定性；
- 归一化/逆变换把 KITTI 米制场景映射回训练尺度。

### 12.3 检测预算竞争

- PointRCNN：有限点槽位；重复点直接占用点级预算。
- CenterPoint：有限体素槽位；同体素重复可能被吸收，但新增伪体素会占用体素预算。
- 因此上采样应优化“独立、受支持的空间证据”，不是总点数。

### 12.4 表面一致性

新增点应：

- 靠近已有/参考表面；
- 沿局部切平面扩展；
- 避免跨越深度断裂和对象边界；
- 不在空中形成 unsupported voxels；
- 对远距离和稀疏目标保持尺度适应。

局部 PCA 可把邻域分解为切平面方向与法向方向；理想新增点在切向扩展，但法向偏移小。

### 12.5 剂量—反应与匹配控制

若生成点比例越高、AP 越低，说明伤害具有剂量效应。  
matched real-fill 控制用于回答：

> 性能变化来自“更多点”，还是来自“生成点的质量”？

PDANS g2.5 与 real-control 的差距仅约 +0.68，说明少量收益需要谨慎解释。

### 12.6 AP 与机制指标的关系

- AP_R40 是最终任务指标；
- 几何、体素和 GT 转移是解释机制的中间证据；
- 好的几何平均值不保证检测提升；
- 检测器最敏感的可能是少量关键目标附近的错误点、空体素和预算竞争。

---

## 13. 论文中可以写与不能写的内容

### 13.1 可以写

- 建立了严格 exact-4N、双线、冻结双检测器的评估协议；
- 证明当前通用 patch 上采样方法在 KITTI 场景上不能自动恢复下游检测；
- PDANS 在正式四方法中最稳健；
- PointRCNN 与 CenterPoint 总体排序一致；
- 输入点/体素预算是误差放大器；
- 旧非局部 patch 和重复填充是可测量的关键管线缺陷；
- PU-Net 的单位球归一化/逆变换是必要适配；
- 少量生成点可能在 Line A 有局部收益，但高比例呈稳定退化；
- surface-c32 在 20 帧几何上明显改善。

### 13.2 不能写

- “4× 上采样恢复了原始检测性能”；
- “PU-Net 方法本身只有 8 AP”；
- “surface-c32 已提高检测 AP”；
- “ModelNet40 分类结果完成”；
- “PDANS g2.5 在全量数据稳定超过原始基线”；
- “TULIP、SPU-PMD 与四正式方法已在相同 exact-4N 协议公平比较”；
- 把早期 x2、100k cap、RPN 26000 或旧适配器结果混入统一主表；
- 把 3 帧或 256 帧机制实验写成 3769 帧正式结论。

---

## 14. 推荐论文结构

### 第 1 章：引言

- LiDAR 稀疏性与下游检测；
- 点云上采样通常优化几何指标，但是否帮助检测仍不明确；
- 本文关注严格控制下的任务有效性和失败机制。

### 第 2 章：相关工作

- 点云上采样：PU-Net、PU-GCN、PU-EdgeFormer、PDANS；
- 原生 LiDAR 上采样：TULIP；
- 自监督拓扑上采样：SPU-PMD；
- 点式检测：PointRCNN/PointNet++；
- 体素中心检测：CenterPoint。

### 第 3 章：方法与实验协议

- exact-4N；
- Line A / Line B；
- observed/generated provenance；
- E1/E2/E3；
- 冻结双检测器；
- 几何、体素与任务指标。

### 第 4 章：主实验结果

- PointRCNN 全量主表；
- CenterPoint 跨检测器结果；
- 方法排序与 Line A/Line B 差异。

### 第 5 章：根因与消融

- patch 非局部；
- 点/体素预算；
- observed-first；
- E3；
- 剂量与 matched real control；
- PU-Net 归一化 × patch；
- 局部重复 patch 的检测器分歧。

### 第 6 章：改进方向

- surface-c32；
- 局部、唯一、表面一致、预算感知；
- 需要任务感知训练或 KITTI 场景微调。

### 第 7 章：局限与结论

- 域差异；
- 旧 PU-Net 主表的适配缺陷；
- surface-c32/ModelNet40 尚未完成；
- 结论限定在当前冻结检测器和输入协议。

---

## 15. 下一步实验优先级

### P0：完成当前 surface-c32 因果闭环

1. 从 231/256 继续或安全重跑 PDANS；
2. 生成 PDANS strict final bins；
3. 对四方法重新审计 exact-4N；
4. 先跑 256 帧 PointRCNN 和 CenterPoint；
5. 与旧 patch、第一版局部重复 patch、原始/下采样基线对比；
6. 只有双检测器至少不恶化时才考虑 3769 帧。

### P1：补充 surface-c32 的关键机制表

- patch unique support；
- patch 直径；
- multi-object fraction；
- generated-to-real NN；
- occupied voxels；
- top10% voxel concentration；
- CenterPoint cap hits；
- PointRCNN 16384 后真实点保留率；
- GT TP→FN / FN→TP。

### P2：把 PU-Net 主结果重新分层

- 旧适配器结果保留为故障对照；
- 正确归一化 + surface-c32 作为修复版；
- 不要用修复版替换旧表而不说明协议变化。

### P3：全量确认少量剂量

- 优先全量 PDANS g2.5；
- 配套 full-val real-control c2.5；
- 预注册同一随机种子、同一 16384 规则；
- 检查提升是否仍存在。

### P4：任务感知改进

- 在 KITTI 局部表面 patch 上微调；
- 对对象边界、深度断裂和地面采用不同 patch 规则；
- 加入法向/平面一致性；
- 对生成点设置 detector-aware confidence；
- 控制新增体素数量，而非仅控制点数；
- 研究把生成点作为可选候选而非强制占满 3N。

### P5：ModelNet40 分支

- 只有在 HPC 环境恢复后继续；
- 先完成 PU-GCN 全量和 PointNet++ baseline；
- 再决定是否值得接入 PU-EdgeFormer；
- 分类结果应独立成章，不与 KITTI 检测结果混称。

---

## 16. 复现实验检查清单

- [ ] 数据 split 是否为 3769 帧 val；
- [ ] Line A/Line B 定义是否一致；
- [ ] 每帧是否严格 4N/4M；
- [ ] 观测点是否全部保留；
- [ ] 生成点 provenance 是否可追踪；
- [ ] intensity 是否按统一最近邻规则赋值；
- [ ] 是否存在 NaN/Inf；
- [ ] PointRCNN 是否固定同一 checkpoint/config；
- [ ] CenterPoint 是否固定同一 checkpoint/config；
- [ ] E2 16384 规则是否完全一致；
- [ ] 是否使用确定性 seed；
- [ ] 是否使用 label/AP 选择输入；
- [ ] PU-Net 是否明确标注旧/新归一化；
- [ ] patch 是否统计局部性、唯一支持和重复率；
- [ ] 是否检查 CenterPoint 40000 体素 cap；
- [ ] 256 帧和 3769 帧结果是否清楚区分；
- [ ] 历史非统一结果是否与正式表隔离；
- [ ] 未完成实验是否明确标为 pending。

---

## 17. 最终一句话

当前项目最有价值的论文贡献，不是证明某个通用上采样网络能在 KITTI 上“造出更多点”，而是通过严格计数、双输入线、双检测器和因果消融证明：

> 对下游 3D 检测而言，决定性能的是新增点能否形成局部、唯一、真实表面支持，并在有限点/体素预算内补充而不是挤占传感器证据。

---

## 18. 2026-09-13 当前状态总览（增补）

### 18.1 结果版本必须分成三层

截至 2026-09-13，KITTI 工作已经形成三类不能互相替代的证据：

| 层级 | 主要目的 | 数据规模 | 检测器状态 | 当前用途 |
|---|---|---:|---|---|
| 四方法统一主实验 | 公平比较 PDANS、PU-GCN、PU-EdgeFormer、PU-Net* | 3769 帧 × Line A/B | 冻结官方/既定权重 | 方法横向主比较 |
| 机制与消融实验 | 定位 patch、归一化、剂量、点/体素预算、对象迁移 | 20/32/64/256 帧 | 冻结或小规模适配 | 解释原因，不能替代全量 AP |
| PU-GCN detector-adaptation 专项 | 判断检测器输入域适配能否追回损失 | 3712 训练帧；3769 验证帧；20 arms | official 与 3-epoch adapted 权重并列 | 最新、最强的输入域适配证据 |

因此，论文或汇报中应同时保留两句话：

1. 四方法冻结检测器的公平排序仍由第 3 节主表给出，PDANS 最稳健，四种方法总体未恢复基线。
2. 后续 PU-GCN 专项证明 detector adaptation 和 observed-first 可以显著追回部分 AP；20/20 arm 矩阵已经完整结束，但其中所有 PU-GCN 输入条件仍没有超过相应 baseline。

### 18.2 当前完成度

- 四方法 strict-4N 全量主矩阵：完成。
- PointRCNN/CenterPoint 跨检测器全量主矩阵：完成。
- E1/E2/E3、observed-first、剂量、体素、对象转移、距离分层：完成。
- PU-Net 归一化 × patch 的 256 帧 2×2：完成；修复版尚未完成 3769 帧双检测器全量主表。
- PU-GCN detector adaptation：3712 帧训练、3769 帧验证、20/20 arms 完成。
- 多方法 surface-c32 全量双检测器重跑：尚未形成统一最终矩阵。
- ModelNet40 分支：仍只有 smoke/协议准备，不能写成完整分类结果。

最新直接证据：

- [PU-GCN 完整 3769 帧检测矩阵](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.md)
- [PU-GCN 完整协议与代码说明](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/protocol_and_code_zh.md)
- [3-epoch 收敛审计](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/convergence_audit.md)

---

## 19. PU-GCN 检测器适配：最新 3769 帧完整结果

### 19.1 协议

- 上采样器：固定 PU-GCN PU1K `model-100`，没有在 KITTI 上按 3 epochs 重训。
- detector adaptation 训练集：KITTI train 3712 帧。
- 最终评价集：KITTI val 3769 帧。
- PointRCNN：RPN 3 epochs，再做 offline RCNN 3 epochs；batch size 1，workers 0，Adam one-cycle，LR 0.0002，seed 20260823。
- CenterPoint：3 epochs；batch size 2，workers 0，Adam one-cycle，LR 0.0003，weight decay 0.01，seed 666。
- 只保存 epoch 3；没有 epoch 1/2 的验证 AP，所以“loss 下降”不等于“已经证明收敛”。
- 最新 full-val evaluator 显式采用 AP_R40；不得把旧 PointRCNN legacy evaluator 的 AP_R11 数字混进本表。

输入类型：

- `baseline`：对应输入线的真实观测；Line A 为原始 N 点，Line B 为 M=floor(N/4) 点。
- `direct`：PU-GCN strict 4N/4M 直接输出。
- `observed-first`：完整保留 N/M 个观测点，再从 strict 预测池确定性抽 3N/3M；总数仍为 4N/4M。
- `official`：既定检测器权重。
- `adapted`：对相应输入域执行 3-epoch detector adaptation 后的权重。

### 19.2 20/20 arms 完整矩阵

以下均为 Car 3D AP_R40 Easy / Moderate / Hard；每个 arm 均为 3769/3769、状态 PASS。

| Detector | Line | Input | Weights | AP_R40 E/M/H | Moderate 差值 |
|---|---|---|---|---:|---:|
| PointRCNN | A | baseline N | official | 92.4325 / 81.9528 / 77.8468 | 0.0000 |
| PointRCNN | A | direct 4N | official | 84.7443 / 61.9294 / 57.7869 | -20.0235 |
| PointRCNN | A | direct 4N | adapted | 85.1671 / 68.8800 / 64.6697 | -13.0728* |
| PointRCNN | A | observed N + predicted 3N | official | 85.0374 / 64.2748 / 60.0911 | -17.6780 |
| PointRCNN | A | observed N + predicted 3N | adapted | 86.7068 / 71.0075 / 66.7939 | -10.9453* |
| PointRCNN | B | baseline M | official | 85.2016 / 65.5757 / 61.2831 | 0.0000 |
| PointRCNN | B | baseline M | adapted | 84.3920 / 68.3312 / 64.2613 | 0.0000 |
| PointRCNN | B | direct 4M | official | 46.4657 / 30.0909 / 25.8000 | -35.4848 |
| PointRCNN | B | direct 4M | adapted | 68.2592 / 46.6150 / 40.2487 | -21.7163 |
| PointRCNN | B | observed M + predicted 3M | official | 55.0177 / 35.5289 / 30.9359 | -30.0469 |
| PointRCNN | B | observed M + predicted 3M | adapted | 74.1970 / 55.2145 / 49.0583 | -13.1167 |
| CenterPoint | A | baseline N | official | 88.3907 / 79.2773 / 76.7371 | 0.0000 |
| CenterPoint | A | direct 4N | official | 81.8888 / 60.6081 / 57.9061 | -18.6692 |
| CenterPoint | A | observed N + predicted 3N | official | 83.3121 / 63.2064 / 61.0418 | -16.0709 |
| CenterPoint | A | observed N + predicted 3N | adapted | 86.8154 / 74.7012 / 72.6570 | -4.5761* |
| CenterPoint | B | baseline M | official | 81.4003 / 64.5979 / 59.9701 | 0.0000 |
| CenterPoint | B | baseline M | adapted | 83.7514 / 68.0490 / 64.6309 | 0.0000 |
| CenterPoint | B | direct 4M | official | 56.4211 / 35.3530 / 31.2043 | -29.2449 |
| CenterPoint | B | observed M + predicted 3M | official | 66.9217 / 44.0929 / 40.0268 | -20.5050 |
| CenterPoint | B | observed M + predicted 3M | adapted | 80.4109 / 61.6639 / 57.5252 | -6.3851 |

`*` Line A 没有单独训练“adapted original-N baseline”；这些差值引用 official Line A baseline，因此同时改变了输入和权重，不能当作纯粹的输入差值。Line B 有配对 adapted baseline，因果口径更干净。

### 19.3 从矩阵直接计算出的作用量

#### detector adaptation 的恢复量

| Detector/Line/Input | Adapted − Official Moderate AP |
|---|---:|
| PointRCNN A direct | +6.9506 |
| PointRCNN A observed-first | +6.7327 |
| PointRCNN B baseline | +2.7555 |
| PointRCNN B direct | +16.5241 |
| PointRCNN B observed-first | +19.6856 |
| CenterPoint A observed-first | +11.4948 |
| CenterPoint B baseline | +3.4511 |
| CenterPoint B observed-first | +17.5710 |

#### observed-first 的恢复量

| Detector/Line/Weights | Observed-first − Direct Moderate AP |
|---|---:|
| PointRCNN A official | +2.3454 |
| PointRCNN A adapted | +2.1275 |
| PointRCNN B official | +5.4380 |
| PointRCNN B adapted | +8.5995 |
| CenterPoint A official | +2.5983 |
| CenterPoint B official | +8.7399 |

CenterPoint 没有 `direct adapted` arm，因此不能从现有矩阵计算 adapted 权重下的纯 observed-first−direct 差值。

### 19.4 最新结论

1. detector adaptation 是有效的缓解手段，尤其对 Line B；例如 PointRCNN observed-first 提升 19.6856 AP，CenterPoint observed-first 提升 17.5710 AP。
2. observed-first 也是有效的输入保护手段，Line B 的收益大于 Line A，说明稀疏输入下真实观测更容易在生成点中被稀释。
3. 两种手段叠加仍没有超过配对 baseline。最接近基线的是 CenterPoint Line A observed-first adapted，仍低 4.5761 AP；具有配对 adapted baseline 的 CenterPoint Line B 仍低 6.3851 AP，PointRCNN Line B 仍低 13.1167 AP。
4. 因此“冻结检测器造成全部失败”不成立；域偏移解释了较大一部分，但生成几何、patch 适配和输入预算仍留下不可忽略的残差。
5. 三轮训练是否最优尚未被证明。只有 epoch-3 checkpoint 和下降的训练 loss，没有逐 epoch full-val 曲线；不能写“3 epochs 已收敛”。

### 19.5 训练 loss 审计

PointRCNN 的各 arm 从 epoch 1 到 3 的 median loss 均下降：

- Line A direct：RPN -17.03%，RCNN -8.00%；
- Line B direct：RPN -27.61%，RCNN -7.65%；
- Line B baseline：RPN -23.95%，RCNN -4.16%；
- Line A observed-first：RPN -16.46%，RCNN -4.42%；
- Line B observed-first：RPN -24.82%，RCNN -6.86%。

CenterPoint mean loss：

- Line A PU-GCN：2.33 → 2.11，下降 9.44%；
- Line B PU-GCN：3.69 → 3.10，下降 15.99%；
- Line B baseline：3.00 → 2.65，下降 11.67%。

这些数字证明优化流程正常执行，不证明 validation AP 已达到平台期。

---

## 20. 2026-08 至 2026-09 新增工作时间线

### 2026-08-05：detector-aware 低剂量 PDANS 路线

- 构造 V1—V5 低剂量、体素感知和 proposal-gated 候选。
- V4 在 pilot256 上把生成点锚定到已有测量体素；CenterPoint 范围内每帧生成点中位数约 423.5，新激活体素中位数为 0。
- V4 的 PointRCNN Car Moderate 3D/BEV 相对 baseline 为 +0.3093/+0.2093；CenterPoint Car 3D 为 -0.0192，Pedestrian/Cyclist 3D 分别为 -0.4366/-0.3439。
- V5 使用无 GT 的 CenterPoint Car 预 NMS 候选进行类别门控，使 Pedestrian/Cyclist 基本回到 baseline，但 pilot256 只用于开发，不能当最终提升。
- 后续独立 holdout 与多 seed 审计未支持把小幅正值扩写成稳定收益。

证据：

- [V4 决策报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_pdans_v4_20260805/reports/v4_decision_report.md)
- [V5 pilot 决策报告](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_pdans_v5_20260805/reports/v5_pilot256_decision_report.md)
- [跨检测器诊断](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_cross_detector_diagnosis256_v1_20260805/summary/diagnosis_report.md)

### 2026-08-08：OpenPCDet 三类别 PointRCNN pilot256

- 使用同一三类别 OpenPCDet PointRCNN checkpoint，对 Car、Pedestrian、Cyclist 并行评价。
- Line A/B 的四种传统 exact-4N 方法在 Moderate 3D 与 BEV 上均未超过对应 baseline。
- PDANS 总体最稳健，PU-GCN 通常第二；失败不局限于 Car。
- detector-aware V4 是唯一接近 baseline 的候选，但它是约 2.4% 低剂量插入，不属于 exact-4N 横向主表。

[三类别完整对比](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/openpcdet_pointrcnn_three_class_all_methods_pilot256_20260808/ALL_METHODS_THREE_CLASS_COMPARISON_ZH.md)

### 2026-08-10 至 2026-08-12：采样噪声与实验规模审计

- split-region 路径在 16 seeds 下，V1−baseline 的 Car Moderate 3D AP 均值为 -0.438，标准差 1.769；最初单 seed 的 +1.2227 被判定为幸运抽样。
- 16 个配对差值中 9 个为负；MDE95=3.47 AP。
- 20 帧重采样实验的配对 delta 标准差为 6.02 AP，MDE95 约 11.8 AP；因此 20 帧只适合发现灾难性回退，不能判断 1—3 AP 的方法收益。
- 该噪声结论只适用于需要随机重复填充的 split-region 路径；不能直接套到固定 16384 点的 E2 路径。

证据：

- [16-seed 判决](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pointrcnn_sampling_variance_probe_20260810/SIGMA_16SEED_VERDICT_ZH.md)
- [候选方法与 20 帧判读能力分诊](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detection_oriented_upsampling_triage_20260812/DETECTION_ORIENTED_UPSAMPLING_TRIAGE_ZH.md)

### 2026-08-24 至 2026-08-27：检测器微调与全训练适配

- 先以固定 64 个训练帧完成 6 arms 的 PointRCNN RPN+RCNN 3-epoch screening，并在同一 256 帧上评价。
- PU-GCN Line A/B 分别因微调恢复 +4.1746/+4.5949 AP；PDANS Line B 恢复 +4.4260 AP；但所有适配输入仍低于相应适配 baseline。
- 小样本微调还使 baseline 本身下降，因此只能作为筛选证据。
- 随后使用全部 3712 个训练帧完成 PU-GCN 的 PointRCNN/CenterPoint 输入域适配，并补做 observed-first 分支。

[64-frame 微调筛查](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pointrcnn_finetune64_six_arms_20260824/reports/finetune64_screen_report.md)

### 2026-09-08 至 2026-09-10：全量 20-arm 验证闭环

- 重新生成/审计 Line A、Line B 的 PU-GCN 输入。
- 完成 direct、observed-first、official、adapted 的 PointRCNN 11 arms 与 CenterPoint 9 arms。
- 汇总器只接收 `frame_count=3769` 且 `status=PASS` 的结果；最终 20/20 通过。
- 显式修正 AP 口径为 R40，避免 repository legacy R11 混用。

### 2026-09-11 至 2026-09-13：论文与交付整理

- 形成第 3—4 章 KITTI 实验配置/方法正文、第 5—6 章结果/讨论正文。
- 形成 37 张 KITTI 学术图、总 contact sheet、Markdown/LaTeX/PDF 多格式材料。
- 整理 PU-GCN detector-adaptation 中英文汇报与代码证据。

主交付：

- [第 3—4 章 KITTI 详细正文](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_3_4_KITTI_DETAILED_ZH.md)
- [第 5—6 章 KITTI 详细正文](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH.md)
- [第 5—6 章 PDF](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH.pdf)
- [统一学术图包](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/CHAPTERS_5_6_UNIFIED_ACADEMIC_FIGURES_20260911)

---

## 21. 代码使用与复现入口（当前版）

### 21.1 一键运行 PU-GCN 全量 detector-adaptation 矩阵

```bash
cd /home/ra87racy/projects/baseline_detectors/PointRCNN
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh all
```

可恢复阶段：

```bash
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh generate_a
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh generate_b
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh prepare
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh pointrcnn
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh centerpoint
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh aggregate
```

`all` 的真实数据流为：

1. `run_patch_causal_upsampling.py`：提取 patch、调用 PU-GCN、合并 raw candidate、生成 strict 4×。
2. `verify_patch_causal_strict_x4.py`：检查帧覆盖、点数、有限值与 strict 规则。
3. `prepare_centerpoint_observed_first_train.py`：构造 observed N/M + predicted 3N/3M。
4. `run_pointrcnn_checkpoint_split_eval.py`：以显式 split、显式 LiDAR 目录、显式 checkpoint 评价 PointRCNN。
5. `run_patch_causal_centerpoint_eval.py`：评价 CenterPoint。
6. `summarize_pugcn_detector_adaptation_full_val_20260908.py`：拒绝任何少于 3769 帧或非 PASS 的 arm，再生成最终矩阵。

### 21.2 Line B 下采样

核心规则：

```text
M = floor(N/4)
seed(frame) = 20260702 + int(frame_id)
choice(N, M, replace=False)
sort(indices)
```

入口：

```bash
/home/ra87racy/projects/baseline_detectors/PointRCNN/venv_pointrcnn/bin/python \
  scripts/prepare_kitti_downsampled_x4_val.py
```

### 21.3 patch 与 PU-GCN 关键参数

| 参数 | Line A | Line B |
|---|---:|---:|
| patch 大小 | 2048 | 2048 |
| primary patch budget | ceil(N/2048) × 1 | ceil(M/2048) × 1 |
| primary ball radius | 2 m | 4 m |
| primary center 最低支持 | 2048 | 2048 |
| cover radius | 6 m | 6 m |
| cover eligibility | 半径内至少 32 点 | 半径内至少 32 点 |
| supplemental kNN floor | 2048 unique | 2048 unique |
| seed | 20260702 + frame_id | 同左 |

PU-GCN 参数：PU1K `model-100`、PUGCN、Inception DenseGCN、`n_blocks=2`、`channels=32`、`k=20`、`d=2`、NodeShuffle、4×。每个 2048 patch 先按质心与最远距离做单位球归一化，输出 8192 点后逆变换回 LiDAR 米制坐标。

### 21.4 strict 4× 与 observed-first 的选择规则

- patch raw 合并后若候选数大于 4N/4M，以固定 seed 均匀无放回抽到目标数；候选不足则失败，不复制点凑数。
- intensity 不由 PU-GCN 预测；对生成 XYZ 使用同帧 observed XYZ 的 1-NN intensity。
- observed-first 的 3N/3M 不是 PU-GCN 的置信度输出，也不是几何最优子集；它使用 SHA-256 派生逐帧 seed 后均匀无放回抽样。
- 当前没有使用 confidence、曲率、FPS、到 observed 的距离或 voxel 去重来选 3N/3M。

### 21.5 运行前后必须保存的证据

- `protocol.json`：协议与路径；
- per-frame manifest：N/M、4N/4M、seed、状态；
- provenance：entrypoint、checkpoint、是否真实调用模型、是否 fallback；
- 输入和输出有限值、shape、点数、观测前缀/多重集合审计；
- detector checkpoint/config；
- 3769 个预测文件及空文件统计；
- AP 原始输出、解析 CSV、run_complete 状态；
- 源码快照、文件大小与 SHA-256。

---

## 22. 代码修改总表（当前工作区可核实）

### 22.1 PointRCNN 主仓库 tracked 修改

截至本次复核，tracked diff 为 4 个文件、72 insertions / 8 deletions：

1. `lib/config.py`
   - `yaml.load` 改为 `yaml.safe_load`；适配新 PyYAML，并避免不安全反序列化。
2. `lib/datasets/kitti_dataset.py`
   - NFS 稳健读取；最多 20 次，间隔 0.1 s；校验字节数、16-byte/point 对齐与 short read。
3. `lib/datasets/kitti_rcnn_dataset.py`
   - far points 已达到 16384 时从全部候选安全无放回抽样；
   - 点数不足且补点数超过已有点数时允许 replacement；
   - 记录 sampler-safe fallback 帧和次数。
4. `lib/rpn/proposal_layer.py`
   - spatial-region/far-only 输入的 near proposal bucket 为空时，不再触发旧断言；
   - 把完整 proposal budget 分配给唯一非空的 far bucket，并走既定 NMS。

### 22.2 OpenPCDet tracked 修改

1. `pcdet/datasets/__init__.py`
   - Argoverse 2 注册改为可选导入；KITTI-only 环境不再强制安装 `av2/pyarrow`。
2. `pcdet/datasets/processor/data_processor.py`
   - 点数不足时，仅在补点数超过已有点数时启用 `replace=True`，修复 Line B 边界错误。
3. `pcdet/models/detectors/detector3d_template.py`
   - 对受信任的项目 checkpoint 显式 `weights_only=False`，兼容 PyTorch 2.6+ 默认行为。
4. `tools/create_kitti_val_infos_only.py`
   - 新增 KITTI val-only info 生成入口。

### 22.3 PU-GCN 修改

`tf_ops/compile.sh`：

- 删除 `/usr/local/cuda-10.0` 硬编码；
- 支持 `CUDA_HOME`/`CUDA_PATH`；
- 优先使用 conda 内 `libcudart.so`；
- 读取 TensorFlow 实际 compile/link flags；
- 补齐 CUDA PATH 与 LD_LIBRARY_PATH。

这些是编译兼容修改，没有改变 Inception DenseGCN、NodeShuffle 或网络损失。

### 22.4 PDANS 修改

- `pointnet2/util.py`：DDIM 初始噪声、label、diffusion step tensor 跟随 condition/net device，去除硬编码 `.cuda()`。
- `pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py`：用 `setdefault` 设置 `TORCH_CUDA_ARCH_LIST=7.5`，允许外部环境覆盖。
- 正式采样使用 float32；半精度试验出现 NaN，未用于结果。

### 22.5 PU-Net 修改与版本边界

- Python 2 → Python 3 兼容、整数除法、数据路径参数、CPU fallback、自定义 op ABI/编译脚本修复。
- `tf_punet_patch_infer.py` 加入 `unit_sphere_v1`：中心化、最大半径归一化、网络推理、逆尺度与逆平移。
- 全量主表仍是 `legacy_none` 旧适配器；修复版只在 256 帧 2×2 中出现。不得静默替换或混写。

### 22.6 修改的学术边界

- “兼容性修复”使代码可运行，不等于提出新网络。
- sampler-safe 与 far-only 修复解决崩溃，不自动解释 AP 变化。
- observed-first、patch 规则和 detector adaptation 属于实验协议/输入适配变化，必须单独命名 arm。
- 当前主仓库和外部仓库均有未提交工作区修改；只记录 commit hash 不足以完整复现，必须同时保存 diff/source snapshot。

---

## 23. 环境、配置、权重与哈希

### 23.1 实际可导入环境（2026-09-02 复核）

| 模块 | Python | 主要框架 | 关键依赖 |
|---|---:|---|---|
| PointRCNN | 3.9.25 | PyTorch 2.8.0+cu128 | NumPy 2.0.2，SciPy 1.13.1 |
| CenterPoint/OpenPCDet | 3.10.20 | PyTorch 2.7.1+cu128 | NumPy 1.26.4，spconv 2.3.6 |
| PU-Net/PU-GCN | 3.6.8 | TensorFlow 1.13.1 | NumPy 1.19.5 |
| PU-EdgeFormer | 3.6.8 | TensorFlow 1.13.1 | NumPy 1.16.6 |
| PDANS full run | 3.11.15 | PyTorch 2.11.0+cu130 | NumPy 2.4.4，ninja 1.13.0 |
| TULIP | 3.8.20 | PyTorch 1.12.0+cu113 | NumPy 1.24.3，timm 1.0.27，einops 0.8.1 |

运行日志记录硬件为 CentOS Stream 9、Intel Core i5-10505、约 46 GiB RAM、NVIDIA Quadro RTX 4000 8 GiB。该项是历史运行日志，不表示写作时 GPU 驱动一定在线。

### 23.2 检测器固定配置

PointRCNN：Car-only 主任务、RPN 16384 点、PointNet++ SA centers `[4096,1024,256,64]`、半径 `[0.1,0.5] / [0.5,1.0] / [1.0,2.0] / [2.0,4.0]` m、每尺度 16/32 邻域、RPN/RCNN score 0.3、RCNN NMS 0.1、RPN test pre/post NMS 9000/100、RPN NMS 0.8、不使用 intensity。

CenterPoint：Car/Pedestrian/Cyclist、MeanVFE、VoxelResBackBone8x、HeightCompression、BaseBEVBackbone、CenterHead；范围 `[0,-40,-3,70.4,40,1]`；体素 `0.05×0.05×0.1 m`；5 points/voxel；40000 test voxels；test 不 shuffle；原始权重训练 80 epochs。

### 23.3 关键哈希

| 文件 | SHA-256 |
|---|---|
| `tools/cfgs/default.yaml` | `ce6473a5b3451106c9866701701cbce95e71c8daf35b2c0b31e4876ad55a653e` |
| `tools/PointRCNN.pth` | `4631beaa311d7b17b0a934e96e3eacbb4944b4bec2d217801541324a83f8ecff` |
| `centerpoint.yaml` | `445d12f0355ac01232b8837ab001cc08e0742061ff0618200a909af9383c501f` |
| `kitti_dataset.yaml` | `a6c44a7b0f15b94e135946cecbd8b5541906a69d37811705e9884a411e7b1797` |
| `checkpoint_epoch_80.pth` | `c3e68c693ad98606f6c790be9e9aaa8bc723df2b67282f21a02350ec50fa69bd` |

仓库提交：主项目 `1d0dee91262b970f460135252049112d80259ca0`，OpenPCDet `233f849829b6ac19afb8af8837a0246890908755`，PU-GCN `0d29ee7c819f3cf80c4535c581d701f4e9bbb8b8`，PU-EdgeFormer 复现目录 `03b8118fa86e32b57a5fb56aea285e17b70bda99`，PDANS `15355e200e8e86658c602951a2751cd9125b2e51`。

---

## 24. 结果解释纪律与已发现的文档冲突

### 24.1 AP 口径

- 正式主结果使用 KITTI AP_R40。
- 最新 full-val runner 显式取 40 个 recall samples。
- 部分旧 PointRCNN repository evaluator 默认产生 legacy AP_R11；旧 pilot 若没有显式 R40 证据，必须保留原始 evaluator 标签，不能与 R40 裸数值直接混合。

### 24.2 数据规模

- 3769：正式 val 全量。
- 3712：detector adaptation train。
- 256：固定 pilot/对象审计；可做强补充证据，不替代 full-val。
- 64/32/20：机制、holdout、体素或几何分析。
- 20 帧 MDE95 约 11.8 AP，因此不能用它确认小幅收益。

### 24.3 TULIP 状态更正

旧正文中曾写“尚未发现 TULIP CenterPoint 全量评价”，但当前目录存在两个完整 PASS 结果：

- Line A CenterPoint Car/Pedestrian/Cyclist Moderate 3D AP_R40：31.7697 / 15.9778 / 4.5920；
- Line B：17.6847 / 2.9893 / 0.0798。

直接证据为 [TULIP CenterPoint full AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_tulip_native_extended_20260729/full_ap_summary.csv)。因此，当前状态应写为“CenterPoint Line A/B 全量已完成，但 TULIP 输出为 range-image native 点数，不属于四方法 exact-4N 主表”。PointRCNN TULIP 使用 4096 点和关闭 distance-based proposal 的专用兼容配置，也不能直接与默认 16384 点主表排序。

### 24.4 可写与不可写的最新边界

可以写：

- detector adaptation 与 observed-first 分别能显著恢复部分 PU-GCN 检测性能；
- 20/20 full-val arm 均已完成；其中所有 PU-GCN 输入 arm 仍低于相应 baseline；
- 冻结检测器域偏移是重要因素，但不是唯一根因；
- Line B 中 paired adapted baseline 提供了最干净的残差证据；
- 三轮训练完成且 loss 下降。

不能写：

- “3 epochs 已经收敛”或“epoch 3 是最优 checkpoint”；
- “适配后 PU-GCN 已恢复或超过 baseline”；
- “Line A adapted 差值是纯输入效应”；
- 把 PU-GCN 专项适配结果推广为所有四种方法的适配上限；
- 把 PU-Net* 旧适配器的极低 AP 当作 PU-Net 架构能力；
- 把 detector-aware V4/V5 单 seed/pilot 的小幅正值写成稳定全量收益。

---

## 25. 最终交付索引与推荐阅读顺序

若只需要一次性掌握整个 KITTI 工作，按以下顺序阅读：

1. 本总账：配置、历史、代码修改、主表、消融、时间线和最新增补。
2. [第 3—4 章详细正文](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_3_4_KITTI_DETAILED_ZH.md)：研究设计、环境、数据、方法接入、格式转换、检测器和指标公式。
3. [第 5—6 章详细正文](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH.md)：全量结果、对象/距离/几何/体素证据、讨论、限制和展望。
4. [最新 PU-GCN 3769 帧矩阵](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.md)：2026-09 的最终 detector-adaptation 数字。
5. [最新协议与代码](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/protocol_and_code_zh.md)：3N/3M 如何选择、patch 如何提取、3 epochs 到底训练了什么、代码入口在哪里。

一句话总结果：

> 在严格 exact-4N、公平双线和双检测器条件下，传统通用上采样并未自动恢复 KITTI 3D 检测；改进 patch、保护观测并对检测器做输入域适配可以追回显著性能，但截至 20/20 全量验证，PU-GCN 仍低于相应基线，说明任务有效增密必须同时解决真实表面支持、观测保护、点/体素预算和检测器分布适配。
