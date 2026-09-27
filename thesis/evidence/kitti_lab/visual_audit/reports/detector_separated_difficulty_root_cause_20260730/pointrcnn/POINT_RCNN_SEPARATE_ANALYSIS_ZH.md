# PointRCNN 独立分析：Easy / Moderate / Hard、正对照与根因

## 1. 分析边界

本报告只分析 PointRCNN，不与 CenterPoint 共图、共表或共享类别结论。当前冻结 PointRCNN 配置只训练和评估 `Car`，因此这里不存在 Pedestrian/Cyclist 缺图问题。

- 正式双线结果：KITTI val 全部 3,769 帧，canonical E2 16,384 点。
- 正对照比例实验：固定 256 帧 Car 子集，嵌套槽位替换；它用于解释机制，不能冒充全验证集结论。
- 指标：BBox、BEV、3D AP_R40，分别报告 Easy、Moderate、Hard。
- 用户所说的 `diff/difficult` 在本报告中对应KITTI官方命名 `Hard`。

## 2. 正式结果：Car

### Line A：Original baseline → Original + x4 upsampling

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 99.32 | 94.08 | 90.10 |
| PDANS | 93.32 (-6.00) | 81.75 (-12.34) | 76.85 (-13.25) |
| PU-GCN | 92.44 (-6.87) | 71.77 (-22.32) | 67.07 (-23.03) |
| PU-EdgeFormer | 81.95 (-17.36) | 58.80 (-35.29) | 52.30 (-37.80) |
| PU-Net | 38.46 (-60.86) | 23.89 (-70.19) | 20.69 (-69.41) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 95.98 | 88.98 | 86.59 |
| PDANS | 90.00 (-5.98) | 78.02 (-10.96) | 73.21 (-13.39) |
| PU-GCN | 86.93 (-9.06) | 66.79 (-22.19) | 62.02 (-24.58) |
| PU-EdgeFormer | 75.97 (-20.01) | 53.83 (-35.15) | 48.94 (-37.65) |
| PU-Net | 28.95 (-67.03) | 18.74 (-70.24) | 16.85 (-69.75) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 92.27 | 82.26 | 77.95 |
| PDANS | 84.61 (-7.66) | 67.22 (-15.03) | 62.40 (-15.55) |
| PU-GCN | 80.17 (-12.10) | 58.28 (-23.98) | 53.47 (-24.47) |
| PU-EdgeFormer | 65.68 (-26.60) | 44.96 (-37.30) | 40.19 (-37.75) |
| PU-Net | 11.76 (-80.51) | 8.38 (-73.88) | 7.44 (-70.50) |

图：

- `figures/car_line_a_absolute_ap.png`
- `figures/car_line_a_delta_ap.png`

难度结论：

1. PDANS 的 3D 下降为 Easy `-7.66`、Moderate `-15.03`、Hard `-15.55`。中/高难度目标比 Easy 多损失约 7–8 AP，说明远距离、遮挡和截断目标对错误新增邻域更敏感。
2. PU-GCN 与 PU-EdgeFormer在 Moderate/Hard 的下降显著大于 Easy；生成点首先破坏的不是明显近车，而是原本只有少量边界点的困难车辆。
3. PU-Net三种难度均失效；Easy下降更大不是“Easy更脆弱”，而是当前坐标归一化接入缺陷造成的全局几何错位，AP已接近下限。

### Line B：Downsampled-x4 baseline → Downsampled + x4 upsampling

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 95.24 | 79.95 | 75.58 |
| PDANS | 83.75 (-11.50) | 60.33 (-19.62) | 53.79 (-21.79) |
| PU-GCN | 60.61 (-34.63) | 38.61 (-41.34) | 32.40 (-43.19) |
| PU-EdgeFormer | 52.75 (-42.49) | 33.49 (-46.46) | 28.71 (-46.88) |
| PU-Net | 33.62 (-61.62) | 20.51 (-59.44) | 18.40 (-57.18) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 91.04 | 76.40 | 72.03 |
| PDANS | 76.63 (-14.41) | 54.57 (-21.83) | 48.14 (-23.89) |
| PU-GCN | 57.25 (-33.79) | 36.19 (-40.20) | 31.43 (-40.61) |
| PU-EdgeFormer | 43.65 (-47.39) | 27.74 (-48.66) | 23.38 (-48.65) |
| PU-Net | 23.77 (-67.27) | 15.52 (-60.88) | 13.62 (-58.41) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 85.18 | 65.75 | 61.38 |
| PDANS | 65.01 (-20.16) | 44.08 (-21.67) | 39.17 (-22.21) |
| PU-GCN | 45.93 (-39.25) | 28.68 (-37.07) | 24.13 (-37.25) |
| PU-EdgeFormer | 33.76 (-51.41) | 20.57 (-45.18) | 17.62 (-43.77) |
| PU-Net | 12.41 (-72.77) | 8.78 (-56.97) | 7.76 (-53.62) |

图：

- `figures/car_line_b_absolute_ap.png`
- `figures/car_line_b_delta_ap.png`

Line B中，PDANS即使是Easy也下降 `-20.16`，Moderate/Hard分别下降 `-21.67/-22.21`；PU-GCN和EdgeFormer下降更大。这说明网络没有重建被删掉的真实扫描证据，而是用生成点替换了PointRCNN固定16,384点预算中的一部分可靠观测。

## 3. 为什么以前会出现合理上升

此前的正结果是：

`Original / PDANS / 2.5% generated slots / Car / 3D Moderate`

它从 `82.66` 提高到 `85.10`，即 `+2.43`。但三种难度和四类指标必须完整展开：

| Metric | Difficulty | Baseline | Observed control (Δ) | PDANS g2.5 (Δ) | PDANS − control |
|---|---|---|---|---|---|
| BBox | Easy | 99.37 | 99.49 (+0.13) | 99.22 (-0.14) | -0.27 |
| BBox | Moderate | 94.12 | 93.93 (-0.19) | 92.31 (-1.81) | -1.62 |
| BBox | Hard | 89.59 | 89.48 (-0.11) | 89.53 (-0.06) | +0.05 |
| BEV | Easy | 96.45 | 96.53 (+0.08) | 96.43 (-0.03) | -0.10 |
| BEV | Moderate | 90.91 | 90.83 (-0.08) | 91.08 (+0.17) | +0.25 |
| BEV | Hard | 88.44 | 86.36 (-2.08) | 86.37 (-2.07) | +0.00 |
| 3D | Easy | 95.09 | 94.80 (-0.29) | 95.44 (+0.35) | +0.64 |
| 3D | Moderate | 82.66 | 84.42 (+1.75) | 85.10 (+2.43) | +0.68 |
| 3D | Hard | 79.93 | 79.82 (-0.10) | 78.63 (-1.30) | -1.20 |
| AOS | Easy | 99.36 | 99.48 (+0.13) | 99.22 (-0.14) | -0.27 |
| AOS | Moderate | 94.01 | 93.80 (-0.21) | 92.19 (-1.82) | -1.61 |
| AOS | Hard | 89.44 | 89.33 (-0.11) | 89.34 (-0.10) | +0.02 |

关键判断：

1. 真正明显上升的主要是 **3D Moderate**。3D Easy只增加 `+0.35`，3D Hard反而下降 `-1.30`；BBox Moderate下降 `-1.81`，BEV Moderate只增加 `+0.17`。因此它不是“整体检测性能提高”，而是一个难度/指标特定的局部最优点。
2. 同槽位真实点对照的3D Moderate已经从 `82.66`升至`84.42`，贡献`+1.75`；PDANS超过匹配对照仅`+0.68`。大部分提升来自固定点预算下的采样覆盖变化，不能全部归因于生成模型。
3. 2.5%只对应约410个生成槽位，约97.5%的PointRCNN输入仍为真实观测。少量高质量PDANS点可能补到个别被采样遗漏的车体表面，同时不足以大范围改变邻域统计。
4. 当比例提高到5%–10%，正增益立即消失；粗比例10%–50%的所有方法总体单调下降。这形成“低剂量局部补证据、高剂量污染邻域”的剂量—反应证据。
5. 该结果来自256帧筛选集，尚不能取代3,769帧正式结果。论文中应称为 `positive mechanism screen` 或 `positive-control candidate`。

对应图：

- `figures/car_pdans_g025_positive_attribution.png`
- `figures/car_fine_ratio_easy.png`
- `figures/car_fine_ratio_moderate.png`
- `figures/car_fine_ratio_hard.png`

## 4. PointRCNN特有的根因链

PointRCNN直接在点上进行前景分割、局部特征聚合和proposal生成。固定输入点数意味着新增生成点不会免费加入：它们会改变被保留真实点、球邻域成员、局部密度和proposal回归证据。

全验证集框迁移进一步证明这不是AP解析问题：

- Line A Car lost/baseline-TP：PDANS `23.1%`、PU-GCN `32.7%`、EdgeFormer `47.9%`、PU-Net `80.9%`。
- Line B：`35.4% / 55.4% / 67.0% / 76.5%`。
- 丢失率随0–20m、20–40m、40m+距离整体上升；具体见 `figures/car_line_*_loss_by_distance.png`。
- sampler-safe fallback只影响3–12/3769帧，比例低于0.4%，排除其作为主因。

所以因果链是：

`生成点几何误差或冗余 → 固定点预算内真实点/生成点构成变化 → 局部邻域与前景得分变化 → proposal置信度/定位退化 → Moderate/Hard丢框率上升 → AP下降`

## 5. 针对PointRCNN的改进优先级

1. 把2.5%作为安全起点，并在完整3,769帧上复验PDANS与匹配真实点对照；不应直接使用严格x4高生成比例。
2. 保留真实点优先；生成点只填充空槽位或低覆盖目标邻域，禁止随机替换可靠真实点。
3. 为生成点提供置信度，按局部表面一致性、range-image邻接和法向残差过滤。
4. 用目标距离/原始点数自适应比例：近距离完整车辆接近0%，稀疏远车允许少量补点。
5. 修复共同patch局部性和PU-Net归一化后再比较模型能力。
6. 若必须使用较高生成比例，应以相同混合分布微调PointRCNN，而不是只改变测试输入。

## 6. 可复核数据

- `tables/formal_ap_all_difficulties.csv`
- `tables/transitions_full_validation.csv`
- `tables/transitions_by_distance.csv`
- `tables/positive_control_256_frame_all_metrics.csv`
- `tables/positive_control_attribution.csv`
