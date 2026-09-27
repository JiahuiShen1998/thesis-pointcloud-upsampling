# KITTI 点云上采样导致 3D 检测性能下降：双实验线、双检测器证据链报告

日期：2026-07-30

## 1. 核心结论

本实验中的性能下降不是单一检测器 bug，也不能用“生成点有噪声”一句话概括。现有结果支持一条三层串联的失效链：

1. **上游局部几何假设被破坏。** 当前公共 patch extractor 对点做粗空间排序后切连续 2,048 点块；实测 Line B patch 的 XY 对角线中位数约 30.46 m、p90 约 124.10 m。一个 patch 可能同时包含道路、多个物体和远背景，不是 PU-Net、PU-GCN、PU-EdgeFormer、PDANS 训练时假设的局部表面。
2. **新增点的“数量”远大于新增的有效传感器证据。** Line B 的参考体素召回只从约 45.8% 提高到 54.8%–59.0%，但最终云中有 45.5%–62.1% 的额外 0.2 m 体素不受完整原始扫描支持；生成体素参考精度只有 23.8%–46.0%。
3. **错误/冗余几何经过检测器的有限输入机制被放大。** PointRCNN 的 RPN 固定接收 16,384 点；CenterPoint 每体素最多 5 点、测试最多 40,000 体素。生成点会改变 PointRCNN 的局部邻域和前景分数，也会改变 CenterPoint 的体素占据、体素均值、BEV heatmap 及框的中心/尺寸/旋转回归。

因此，当前管线的真实效果不是“稀疏点云变稠密”，而是：

`少量正确表面覆盖恢复 + 大量冗余/错位体素 + detector-specific 有限预算竞争`

这解释了为什么：

- Line A 原始扫描已经完整，上采样只有新增估计，没有可恢复的真实测量，所以四种方法在两检测器上全部下降。
- Line B 虽有少量恢复，但错误几何的代价更大；Car 的 0.7 IoU 定位指标全部下降。
- CenterPoint 的 Pedestrian/Cyclist 在 Line B 有少数小幅正增益，因为小目标基线极稀疏且 IoU 门槛为 0.5；这不否定整体几何污染。
- 方法排序在两检测器上基本稳定为 `PDANS > PU-GCN > PU-EdgeFormer > 当前 PU-Net 管线`，并与生成体素参考精度排序一致。

## 2. 冻结协议与证据边界

| 项目 | PointRCNN | CenterPoint |
|---|---|---|
| 正式输入 | E1 上采样云经统一 E2 适配 | exact-4N E1 |
| 检测器实际输入 | FOV/range + 0.1 m voxel representative + 分层确定性采样，恰好 16,384 点 | FOV/range + 0.05×0.05×0.1 m voxel，每 voxel 最多 5 点，最多 40,000 voxels |
| 类别 | 当前 checkpoint/config 仅 Car | Car、Pedestrian、Cyclist |
| 全量指标 | 3,769 帧 KITTI val，BBox/BEV/3D AP_R40 | 3,769 帧 KITTI val，BBox/BEV/3D AP_R40 |
| 三帧审计 | 同类 oriented 3D IoU，Car 0.70 | Car 0.70，Pedestrian/Cyclist 0.50 |

三帧逐框匹配用于解释“哪个物体发生了什么变化”，不替代 KITTI evaluator 的 difficulty、DontCare 和全阈值积分。所有论文主表数值必须取自 `analysis/all_detector_ap_r40_bbox_bev_3d.csv`。

PointRCNN 只显示 Car 不是可视化漏类，而是当前冻结配置 `CLASSES: Car`。CenterPoint 的冻结配置明确包含 Car、Pedestrian、Cyclist。

## 3. 全量 AP 结果

### 3.1 PointRCNN：Car Moderate AP_R40

| Line | 输入 | BBox | BEV | 3D | Δ3D vs 本线 baseline |
|---|---|---:|---:|---:|---:|
| A | Original baseline | 94.08 | 88.98 | 82.26 | — |
| A | PDANS | 81.75 | 78.02 | 67.22 | -15.03 |
| A | PU-GCN | 71.77 | 66.79 | 58.28 | -23.98 |
| A | PU-EdgeFormer | 58.80 | 53.83 | 44.96 | -37.30 |
| A | PU-Net* | 23.89 | 18.74 | 8.38 | -73.88 |
| B | Downsampled baseline | 79.95 | 76.40 | 65.75 | — |
| B | PDANS | 60.33 | 54.57 | 44.08 | -21.67 |
| B | PU-GCN | 38.61 | 36.19 | 28.68 | -37.07 |
| B | PU-EdgeFormer | 33.49 | 27.74 | 20.57 | -45.18 |
| B | PU-Net* | 20.51 | 15.52 | 8.78 | -56.97 |

### 3.2 CenterPoint：Moderate 3D AP_R40

| Line | 输入 | Car | Pedestrian | Cyclist |
|---|---|---:|---:|---:|
| A | Original baseline | 79.28 | 50.65 | 64.61 |
| A | PDANS | 64.36 (-14.92) | 45.19 (-5.46) | 45.90 (-18.71) |
| A | PU-GCN | 58.90 (-20.38) | 44.15 (-6.51) | 44.31 (-20.30) |
| A | PU-EdgeFormer | 45.73 (-33.54) | 36.82 (-13.83) | 35.22 (-29.38) |
| A | PU-Net* | 10.75 (-68.53) | 19.49 (-31.16) | 13.19 (-51.41) |
| B | Downsampled baseline | 64.60 | 24.57 | 15.46 |
| B | PDANS | 46.75 (-17.85) | 28.18 (**+3.61**) | 15.52 (+0.06) |
| B | PU-GCN | 39.07 (-25.53) | 24.93 (+0.36) | 16.85 (**+1.39**) |
| B | PU-EdgeFormer | 24.75 (-39.85) | 15.59 (-8.98) | 7.04 (-8.42) |
| B | PU-Net* | 11.72 (-52.88) | 11.92 (-12.65) | 5.02 (-10.44) |

`*` 当前 PU-Net 是已确认的 wrapper-defect 管线结果，不能作为 PU-Net 方法能力上限。

完整 Easy/Moderate/Hard、BBox/BEV/3D 的 138 行长表见：

`analysis/all_detector_ap_r40_bbox_bev_3d.csv`

## 4. 逐 GT 全量转移统计

对 3,769 帧中每个可评测 GT，分别判断 baseline 和 upsampled 输入是否达到相应 3D IoU 门槛。以下是 Car 中“baseline TP 在上采样后变成 FN”的比例：

| Detector | Line | PDANS | PU-GCN | PU-EdgeFormer | PU-Net* |
|---|---|---:|---:|---:|---:|
| PointRCNN | A | 23.1% | 32.7% | 47.9% | 80.9% |
| PointRCNN | B | 35.4% | 55.4% | 67.0% | 76.5% |
| CenterPoint | A | 20.6% | 25.9% | 39.3% | 70.8% |
| CenterPoint | B | 27.6% | 37.4% | 55.3% | 67.0% |

这组比例有三个作用：

1. 它证明 AP 下降对应真实的逐物体漏检增长，不是 AP 文本解析错误。
2. 两个结构不同的 detector 呈现一致方法排序，排除了“只对某一个 detector 架构不兼容”的解释。
3. Line B 中 PointRCNN 和 CenterPoint 的丢失比例普遍高于 Line A，说明“从 1/4 稀疏点恢复到原点数”并没有恢复原来的空间证据。

完整按 detector / line / method / class 的保持、置信度下降、定位下降、丢失、恢复和双漏检统计见：

- `analysis/transition_summary_full_val.csv`
- `analysis/transition_summary_by_distance.csv`
- `figures/full_val_lost_vs_recovered_gt.png`
- `figures/full_val_car_loss_by_distance.png`

## 5. 三帧为什么这样选

三帧先由全部 3,769 帧的跨方法、跨实验线、跨 detector 逐框事件排名，再补充“必须出现恢复框”和“小目标多类别”的约束：

| Frame | GT 构成 | 主要用途 |
|---|---|---|
| 000590 | 17 Car + 1 Cyclist | 密集道路中连续 Car 从高 IoU baseline 框变成漏检；观察随方法变差的级联效应 |
| 005625 | 15 Car + 1 Cyclist | 同一帧同时包含 lost 与 recovered；PointRCNN 和 CenterPoint 均有正负变化 |
| 006682 | 3 Car + 7 Pedestrian + 5 Cyclist | CenterPoint 多类别、小目标、0.5 IoU 情况；验证少量恢复与大量损失并存 |

### 5.1 Frame 000590：恢复不是主导，连续漏检是主导

- PointRCNN Line A / PDANS：4 个 baseline TP 丢失，另有 2 个定位下降、6 个置信度下降。
- 其中 GT 2（x≈11.35 m）baseline 3D IoU 0.777，GT 4（x≈16.25 m）0.815，GT 10（x≈19.27 m）0.738，GT 15（x≈39.76 m）0.807；上采样后均无达到 0.7 的 Car 框。
- PointRCNN Line A / PU-EdgeFormer 与 PU-Net 均丢失 9 个 baseline TP。
- CenterPoint Line A / PDANS 丢失 7 个 Car；PU-GCN 10 个；PU-EdgeFormer 与 PU-Net 各 12 个。该帧呈现与全量 AP 相同的排序。

### 5.2 Frame 005625：恢复框真实存在，但数量远少于丢失框

- PointRCNN Line A / PDANS：GT 4（x≈5.38 m）由 baseline FN 变成 upsampled TP，3D IoU=0.792；同一比较中仍有 3 个 Car 从 TP 变成 FN。
- PointRCNN Line B / PDANS：GT 7（x≈13.26 m）恢复，3D IoU=0.705；同时 5 个 baseline TP 丢失。
- PointRCNN Line B / PU-EdgeFormer：GT 4 恢复，3D IoU=0.795；同时 9 个 baseline TP 丢失。
- CenterPoint Line B / PDANS：27.13 m 的 Cyclist 恢复，3D IoU=0.549；这与全量 Cyclist +0.06 AP 的微小增益一致。
- CenterPoint Line B / PU-GCN：同一个 Cyclist 恢复到 3D IoU=0.655，同时又恢复 31.01 m 的 Car；但该帧仍丢失 7 个 Car。

这帧是论文中很重要的反例：上采样不是“从不帮助”，而是当前管线中的局部帮助不足以抵消更大范围的损失。

### 5.3 Frame 006682：小目标可以偶发受益，Car 和总体仍下降

- CenterPoint Line A / PDANS：3 个 Cyclist 丢失、1 个 Cyclist 定位下降、1 个 Pedestrian 丢失。
- CenterPoint Line A / PU-Net：恢复 1 个 Pedestrian，但丢失 3 个 Pedestrian、4 个 Cyclist 和 1 个 Car。
- CenterPoint Line B / PU-GCN：恢复 1 个 Pedestrian，但同时丢失 1 个 Pedestrian 和 2 个 Cyclist。
- PointRCNN 为 Car-only；该帧 Line B 四方法均丢失同一个 baseline Car TP。

## 6. 为什么 PointRCNN 下降

[PointRCNN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.pdf) 是两阶段 point-based detector：RPN 从点级局部邻域生成 3D proposals，RCNN 再做 RoI pooling 和框精修。当前冻结配置的关键事实是：

- RPN 只接收 16,384 点。
- Set abstraction 的半径从 0.1/0.5 m 逐级扩大到 2/4 m。
- RPN 与 RCNN score threshold 均为 0.3，RCNN NMS threshold 为 0.1。

当前 E2 输入中，四方法两条线的 exact observed 行上界通常只有约 23%–26%，其余约 74%–77% 为生成/非 observed 行。因而：

1. 生成点直接改变 PointNet++ ball-query/kNN 邻域的成员与局部密度。
2. 错位点会制造假的前景连续性、稀释真实边界，改变 point-wise foreground score。
3. 进入固定 16,384 槽位的生成点会挤掉独立真实测量；Line A 即使 E1 保留全部 observed，E2 仍不可能全部保留。
4. proposal 的中心/尺寸/朝向偏移后，Car 很容易从 3D IoU 0.7 上方跌到下方；逐框统计中的 localization degradation 和 lost 正是这一现象。

比例实验提供独立剂量证据：生成点比例从 10% 增至 50% 时，PointRCNN Car Moderate 3D AP 在两条线、四方法上总体单调下降；相同比例使用真实点替换时明显优于生成点。唯一值得完整验证的候选是 Line A / PDANS / 2.5%，但其 +2.43 AP 中匹配真实点 control 已贡献大部分，生成几何相对 control 只高 +0.68。

## 7. 为什么 CenterPoint 下降

[CenterPoint](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html) 将对象表示为 BEV 中心点，并回归中心偏移、高度、尺寸和旋转。当前 OpenPCDet 配置使用 MeanVFE、稀疏 3D backbone、HeightCompression 和 CenterHead。

冻结数据处理为：

- 检测范围 `[0,70.4] × [-40,40] × [-3,1] m`
- voxel size `0.05 × 0.05 × 0.1 m`
- 每 voxel 最多 5 点
- test 最多 40,000 voxels

Line A 的中位预 cap 体素数与 32 帧 cap 命中为：

| 输入 | 预 cap voxels 中位数 | 40k cap 命中 |
|---|---:|---:|
| Original | 14,944 | 0/32 |
| PDANS | 36,913 | 6/32 |
| PU-GCN | 45,108 | 29/32 |
| PU-EdgeFormer | 46,182 | 29/32 |
| PU-Net* | 55,384 | 32/32 |

这说明 Line A 中部分真实测量会与生成点竞争每体素 5 点和 40k voxel 预算。只把相同 float32 点集合改成 observed-first 顺序时，Line A 的 Moderate 3D AP 有小幅恢复，尤其 PU-GCN、PU-EdgeFormer 和 PU-Net；这直接证明输入顺序/有限体素预算是一个真实机制。

但是它不是主因：

- observed-first 后所有 Line A 方法仍低于原始 baseline。
- Line B 四方法在 32/32 样本中均为 0 次触发 40k cap，Car 仍下降 17.85–52.88 AP。
- Line B 改变点顺序的 AP 变化仅约 -0.0018 到 +0.0311。

所以 CenterPoint 的主损失仍是生成体素位置/分布错误；cap 只是 Line A 的附加放大器。

## 8. 四种方法的具体解释

### 8.1 PDANS

[PDANS](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Point_Cloud_Upsampling_Using_Conditional_Diffusion_Module_with_Adaptive_Noise_CVPR_2025_paper.pdf) 使用 conditional diffusion 和 Adaptive Noise Suppression。当前结果中其生成体素参考精度最高：

- Line A：47.5%
- Line B：46.0%

因此两 detector、两条线中 PDANS 都是四方法最强。它仍下降，是因为 ANS 能抑制局部噪声，但不能把跨 30–124 m 的非局部 patch 重新变成真实局部表面，也不能凭空恢复被下采样删除的传感器射线证据。

### 8.2 PU-GCN

[PU-GCN](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_PU-GCN_Point_Cloud_Upsampling_Using_Graph_Convolutional_Networks_CVPR_2021_paper.pdf) 用 Inception DenseGCN 建模多尺度邻域，并用 NodeShuffle 扩点。当前生成体素参考精度为：

- Line A：37.2%
- Line B：33.0%

公共 patch 非局部时，图边连接的不再是同一表面的真实邻居，NodeShuffle 会把错误邻域关系扩展为更多点。其 AP 和逐框丢失率因此稳定差于 PDANS。

### 8.3 PU-EdgeFormer

[PU-EdgeFormer](https://arxiv.org/abs/2305.01148) 结合 EdgeConv 与 multi-head self-attention，希望同时学习局部和全局结构。当前生成体素参考精度降到：

- Line A：29.5%
- Line B：26.2%

在非局部 patch 上，“全局关系”会跨道路、车辆和背景建立注意力，错误关联的影响比局部 GCN 更容易扩散。它在两个 detector 上都呈现更高的 TP 丢失率和更低 AP。

### 8.4 PU-Net

[PU-Net](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.html) 是 patch-level PointNet++ 多尺度特征、feature expansion 和坐标重建方法。当前 wrapper 在第 120–124 行把米制 patch 直接送入 `bradius=1.0` 的预训练网络，然后直接保存输出；没有：

- patch centroid subtraction
- radius/scale normalization
- output inverse scale
- output centroid restoration

因此当前 PU-Net 的参考体素精度最低（Line A 17.2%，Line B 23.8%），并且 AP 最差。这是已确认的接入缺陷，不能写成“PU-Net 天生不适合检测”。

## 9. 已排除或被削弱的替代解释

| 替代解释 | 证据判断 |
|---|---|
| 只是 PointRCNN sampler crash/patch | 否。sampler-safe fallback 只触发 3–12/3,769 帧；全量 AP 和逐框下降远大于该范围 |
| 只是 PointRCNN 的 16,384 cap | 否。PointRCNN 确实受影响，但 CenterPoint 也出现相同方法排序；CenterPoint Line B 无 40k cap 仍下降 |
| 只是 CenterPoint 的 40k voxel cap | 否。Line B 0/32 cap hit，仍有大幅 Car 下降 |
| 只是点顺序 | 否。observed-first 只对 Line A 有有限恢复，对 Line B 几乎无效 |
| 只是总点数不同 | 否。Line B exact-4N 最终点数接近原始扫描，但有效参考体素恢复有限；同比例真实点 control 优于生成点 |
| 只是 AP evaluator/解析 | 否。全量 recall@0.7、每帧预测数、逐 GT lost/recovered 和具体框 IoU 均同步变化 |
| 所有新增点都有害 | 否。005625 与 006682 中有明确 recovered Car/Pedestrian/Cyclist；问题是收益数量和稳定性不足以抵消损失 |

## 10. 最可能提高性能的路线

### P0：先修复上游有效性，否则不要继续 detector 调参

1. 把公共连续空间 chunk 改为 `FPS seed + kNN/ball-query` 真局部 patch。
2. patch 必须重叠，并用距离中心权重或置信度融合，避免接缝和壳层。
3. 每个方法严格复现训练时的中心化、尺度归一化、输出逆变换；先修 PU-Net。
4. 保存每个 generated point 的来源 patch、局部尺度、邻域半径、模型置信度/扩散不确定度。

### P1：生成点不能与真实点同权

1. 永远保留 observed 点；generated 只能补充剩余预算。
2. 用局部平面残差、法向一致性、range-image 射线一致性、邻域密度上界和模型不确定度过滤 generated。
3. 对 PointRCNN，将 E2 改成 observed-first 分层采样；generated 比例先从 2.5%/5%/10% 做完整验证，不直接用 75%。
4. 对 CenterPoint，先 observed-first voxelization；同体素内优先真实点，再用生成点补到 5 点。Line A 可减少 cap 竞争，但不要期待它修复 Line B 几何。

### P2：做 detector-aware 训练，而不是只在测试时换分布

1. 用真实 KITTI + 同方法上采样混合输入微调 detector。
2. 训练时随机生成点比例和方法，避免 detector 只适应单一伪分布。
3. 加入 feature consistency：原始输入和上采样输入的 foreground heatmap/proposal/box 应一致。
4. 对 generated point/voxel 增加一位 provenance 或置信度 feature，使 detector 能学习降权；不要伪装成与真实 intensity 完全同等的测量。

### P3：最小、可证伪的后续实验

1. 完整 val：Line A / PDANS / g2.5、匹配真实点 c2.5、baseline，三者同 seed、同输入槽位。
2. 修复 patch + PU-Net wrapper 后只跑 32 帧几何门槛；生成体素精度未明显高于当前 PDANS 前不要投入全量 detector。
3. CenterPoint 做四个 ablation：原顺序、observed-first、observed-only、observed-first + filtered-generated。
4. PointRCNN 做四个比例：0/2.5/5/10%，同时记录每个 GT box 内 observed/generated 数量、最近邻距离、边界壳层比例。
5. 论文主结论使用 bootstrap frame resampling 给 AP delta/transition rate 置信区间，避免把单帧恢复写成总体提升。

## 11. 可视化和数据产物

每个 `frame × line × method × detector` 都包含：

- `full_frame_3d_bev.png`：baseline/upsampled 的 detector-effective 3D 与 BEV，显示全部 GT 和全部 final prediction boxes。
- `case_manifest.json`：点云来源、全部 GT、全部 baseline/upsampled prediction、score、BEV/3D IoU、transition 和可视化路径。
- `object_crops/`：每个 case 预渲染最多 3 个代表性变化物体；其余任意 GT 可用 Open3D viewer 的 `--gt-index` 即时裁剪。

共享点云页：

- `pointcloud_full_frame.png`：完整 frame；observed 为小灰点，generated 为小蓝点。

全量清单：

- `analysis/all_selected_frame_boxes.csv`：三帧涉及的全部 GT 和全部两个 detector 的 final prediction box。
- `analysis/selected_frame_gt_transitions.csv`：逐 GT baseline→upsampled 转移。
- `analysis/object_crop_index.csv`：预渲染裁剪索引。
- `case_index.csv`：48 个 detector case 的统一入口。

原生 3D 交互不使用 HTML：

```bash
/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python \
  scripts/view_dual_detector_evidence_open3d.py \
  --frame 005625 --line B --method pdans --detector centerpoint \
  --state upsampled --effective --gt-index 4 --point-size 1.0
```

## 12. 论文中建议采用的因果表述

可支持：

> 在冻结 detector、checkpoint、val split 和 detector-specific 输入协议后，四种上采样方法在两类结构不同的 detector 上呈现一致的 Car 性能排序和逐 GT 漏检增长。真实点比例/顺序对照证明有限输入预算是放大因素；Line B 无 voxel-cap 仍下降、同比例真实点优于生成点、生成体素参考精度与 AP 排序一致，则共同指向非局部 patch 和错误/冗余生成几何为主要原因。

不应直接声称：

- 每一个丢失框都由唯一某个 generated point 导致。
- 当前 PU-Net 数值代表论文方法上限。
- 三个可视化帧可以替代 3,769 帧官方 AP。
- CenterPoint 的小目标正增益代表整体上采样恢复成功。

