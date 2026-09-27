# ModelNet40 PointNet++ Line B Result Interpretation

- Generated at: 2026-07-07 19:50:00 UTC
- Scope: formal Line B full training (5 branches, 200 epochs, seed=42)

## 1. Line B 实验目的

比较 **Downsampled ×4 baseline（256 pts）** 与 **Downsampled ×4 + Upsampling（1024 pts）** 四种方法（EAR、PDANS、PU-Net、PU-GCN）在 PointNet++ 分类任务上的表现。

## 2. Baseline

- **Downsampled ×4 baseline**, 256 points
- Best overall accuracy: **90.85%**
- Best class accuracy: **87.37%**
- Final epoch (200) test overall: **90.46%**

## 3. Upsampling methods（1024 pts）

| method | best overall | Δ vs baseline | best class | Δ vs baseline |
| --- | ---: | ---: | ---: | ---: |
| EAR | 88.81% | −2.04 pp | 84.50% | −2.86 pp |
| PDANS | 90.34% | −0.51 pp | 86.20% | −1.17 pp |
| PU-Net | **91.27%** | **+0.42 pp** | **87.86%** | **+0.49 pp** |
| PU-GCN | 90.06% | −0.79 pp | 86.41% | −0.96 pp |

## 4. 分类结果观察

- **PU-Net** 是唯一在 best overall / best class 上均略高于 downsampled baseline 的方法（+0.42 pp / +0.49 pp）。
- **EAR** 分类表现最低，低于 baseline 约 2 个百分点。
- **PDANS** 和 **PU-GCN** 介于 EAR 与 PU-Net 之间，均未超过 baseline 的 best overall。

## 5. 几何指标对照

| method | CD (Δ vs Original) | HD | NUC | best overall cls |
| --- | --- | --- | --- | --- |
| Downsampled x4 baseline | 0.077 (+0.028) | 0.214 | 2.049 | 90.85% |
| + EAR | 0.081 (+0.032) | 0.209 | 0.958 | 88.81% |
| + PDANS | 0.063 (+0.014) | 0.206 | 1.090 | 90.34% |
| + PU-Net | 0.057 (+0.008) | 0.138 | 1.058 | **91.27%** |
| + PU-GCN | 0.055 (+0.005) | 0.165 | 1.486 | 90.06% |

- PU-GCN 和 PU-Net 在 **CD** 上最接近 Original baseline（几何重建较好）。
- **PU-GCN** 的 **NUC** 明显高于其他方法（1.49 vs ~1.06），点分布均匀性较差。
- **PU-Net** 在 HD 和 NUC 上相对更平衡。
- 几何 CD 更优的方法（PU-GCN、PU-Net）并不都对应更高的分类精度：PU-GCN CD 最低但分类低于 PU-Net。
- EAR 几何指标中等，但分类表现最差——几何与分类一致性在此 branch 上不明显。

## 6. 谨慎结论

- 当前 Line B 数据表明：**256→1024 upsampling 并非对所有方法都提升分类性能**；仅 PU-Net 带来小幅提升。
- 几何质量（CD/HD/P2F/NUC）与 downstream classification 之间**不存在简单单调关系**。
- 不宜仅凭 CD 单一指标推断分类优劣；需结合 HD、P2F、NUC 与实测分类结果综合判断。
- 本报告仅记录 Line B 观察，不做跨 Line 或最终 thesis 结论。
