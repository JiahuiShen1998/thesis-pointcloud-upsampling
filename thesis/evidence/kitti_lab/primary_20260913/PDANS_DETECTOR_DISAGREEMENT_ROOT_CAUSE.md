# PDANS旧patch与cover-kNN-v3检测器分歧归因

## 结论

当前 `fps_ball_cover_knn_v3` 的 `k256` 配置不能作为最终公共patch。它虽然显著改善生成点的局部几何，但每个2048点patch只保证至少256个唯一源点，其余位置允许重复填充。重复输入经PDANS和重叠patch合并后形成高度集中的近重复输出。

CenterPoint通过“真实观测点优先 + 体素化”吸收了这种集中：真实点全部保留，每个体素最多5点、测试时最多40000个体素，因而大量重复点不会线性放大，旧patch产生的噪声体素和低分假阳性反而被清除。PointRCNN直接从原始行中固定采样16384点，没有体素去重；近重复点占用采样预算，导致空间覆盖和目标框内真实结构显著下降。

因此，这不是“局部patch对CenterPoint正确、对PointRCNN错误”这么简单，而是当前 `k256` 的重复填充与两种检测器输入机制发生了不同交互。

## 配对AP结果

所有结果均来自同一256帧协议。CenterPoint旧值取自本次箱体迁移所使用的原始 `result.pkl` 对应原始评估日志；根目录中的旧汇总CSV与原始日志存在小幅不一致，不能替代该配对日志。

| 检测器/类别 | 指标 | 旧patch Moderate | 新patch Moderate | 差值 |
|---|---|---:|---:|---:|
| PointRCNN Car | BEV AP_R40 | 70.3912 | 51.6618 | -18.7293 |
| PointRCNN Car | 3D AP_R40 | 57.1829 | 43.6976 | -13.4853 |
| CenterPoint Car | BEV AP_R40 | 70.6567 | 86.4220 | +15.7653 |
| CenterPoint Car | 3D AP_R40 | 61.1172 | 77.4150 | +16.2978 |
| CenterPoint Pedestrian | BEV AP_R40 | 40.7677 | 50.1468 | +9.3791 |
| CenterPoint Pedestrian | 3D AP_R40 | 39.8318 | 41.1749 | +1.3431 |
| CenterPoint Cyclist | BEV AP_R40 | 57.1362 | 77.7645 | +20.6283 |
| CenterPoint Cyclist | 3D AP_R40 | 52.3814 | 75.8675 | +23.4861 |

## GT匹配迁移

使用同类别、KITTI标准3D IoU阈值匹配预测框与GT。该分析用于解释变化，不代替官方AP。

| 检测器/类别 | 旧匹配GT | 新匹配GT | 净变化 | 丢失 | 恢复 | 置信度下降 | 定位下降 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PointRCNN Car | 539 | 402 | -137 | 184 | 47 | 263 | 26 |
| CenterPoint Car | 621 | 748 | +127 | 31 | 158 | 19 | 6 |
| CenterPoint Pedestrian | 65 | 65 | 0 | 10 | 10 | 4 | 0 |
| CenterPoint Cyclist | 38 | 46 | +8 | 1 | 9 | 3 | 1 |

CenterPoint Car的低阈值预测总数从7304降至4850，GT未匹配框从6683降至4102；但分数不低于0.7的框从551增至710。这表明它主要在清除低质量响应的同时提高真实目标置信度。

PointRCNN Car的匹配GT净减少137个，另有263个仍匹配目标出现至少0.1的置信度下降，和AP下降方向一致。

## patch重复来源

帧 `000001` 的patch元数据给出：

- patch数：277；
- 源点覆盖率：99.9867%；
- 每个源点平均进入patch次数：4.72；
- 单patch唯一源点中位数：464/2048；
- 单patch唯一源点P10：256/2048；
- 重复填充行：337167/567296，约59.4%。

提取器的配置语义是：`--min-ball-points 256` 只保证256个唯一邻点，之后通过 `np.resize` 重复到2048。严格×4适配器从合并raw输出中无放回抽样，本身没有制造重复；高度重复已经存在于局部patch推理的合并输出中。

## 点密度与体素占用

20帧中位数：

| 实际输入/候选 | 指标 | 旧patch | 新patch |
|---|---|---:|---:|
| 严格4N | 1毫米唯一点比例 | 99.95% | 45.27% |
| 严格4N | 5cm×5cm×10cm占用体素 | 97158.5 | 44608.5 |
| CenterPoint实际输入 | 1毫米唯一点比例 | 99.96% | 61.59% |
| CenterPoint实际输入 | 占用体素 | 97932.0 | 58352.0 |
| CenterPoint实际输入 | 点落在最密10%体素的比例 | 40.83% | 62.52% |
| PointRCNN实际16384点样本 | 1毫米唯一点比例 | 99.90% | 57.70% |
| PointRCNN实际16384点样本 | 占用体素 | 11584.5 | 2124.5 |
| PointRCNN实际16384点样本 | 点落在最密10%体素的比例 | 27.23% | 87.85% |

PointRCNN的实际采样按冻结代码和相同随机种子逐帧复现。新patch使其空间占用体素减少约81.7%，点分布严重集中。

20帧Moderate Car框内，PointRCNN采样点数相对原始观测点数的中位比值从0.804降至0.231，下降约71.3%；P10从0.293降至0.021。也就是说，部分Moderate目标在16384点采样后几乎没有保留有效局部结构。

CenterPoint输入始终以全部真实观测点为前缀。Moderate Car框内点数相对真实观测的中位比值由4.295降至1.550，但P10仍为1.270，即真实观测证据没有丢失，只是减少了旧patch带来的过量合成点。其体素上限进一步抑制重复点，解释了假阳性减少和AP提升。

## 下一步

1. 不扩展当前 `k256` 配置到3769帧。
2. 公共patch增加硬约束：每个patch必须有2048个唯一源点，`support_repeat_count=0`。
3. 在20帧只做提取，不跑网络，对比两种候选：
   - `fps_ball_cover_knn_v3`，但将kNN floor提高到2048；
   - `fps_knn_local_v1`，始终使用2048个唯一最近邻。
4. 根据patch直径、目标源点覆盖率和跨边界代理筛掉不合格候选，不用检测AP调参。
5. 固定一个候选后先用PU-GCN快速筛查，再只让PDANS跑20帧确认。
6. PDANS候选必须同时满足：无重复填充、严格×4、PointRCNN采样覆盖不再崩塌、CenterPoint改善不被破坏；通过后才重跑256帧。

## 产物

- `root_cause/summary.json`
- `root_cause/detector_transition_summary.csv`
- `root_cause/detector_gt_transitions.csv`
- `root_cause/geometry_scene20_summary.csv`
- `root_cause/geometry_object20_summary.csv`
- `pointrcnn_boxes/box_change_summary.json`
- `centerpoint_boxes/box_change_summary.json`
