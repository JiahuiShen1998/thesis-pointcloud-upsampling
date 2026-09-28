# ModelNet40 PointNet++ Line B Result Interpretation

- Generated at: 2026-07-07 19:50:00 UTC
- Scope: formal Line B full training (5 branches, 200 epochs, seed=42)

## 1. Line B experimental purpose

Comparison **Downsampled ×4 baseline (256 pts)** with **Downsampled ×4 + Upsampling (1024 pts)** The performance of four methods (EAR, PDANS, PU-Net, PU-GCN) in the PointNet++ classification mission.

## 2. Baseline

- **Downsampled ×4 baseline**, 256 points
- Best overall accuracy: **90.85%**
- Best class accuracy: **87.37%**
- Final epoch (200) test overall: **90.46%**

## 3. Upsampling methods (1024 pts)

| method | best overall | Δ vs baseline | best class | Δ vs baseline |
| --- | ---: | ---: | ---: | ---: |
| EAR | 88.81% | −2.04 pp | 84.50% | −2.86 pp |
| PDANS | 90.34% | −0.51 pp | 86.20% | −1.17 pp |
| PU-Net | **91.27%** | **+0.42 pp** | **87.86%** | **+0.49 pp** |
| PU-GCN | 90.06% | −0.79 pp | 86.41% | −0.96 pp |

## 4. classification Results Watch

- **PU-Net** The only method (+ 0.42 pp / +0.49 pp) for which best overall / best class is used is to provide an average of above downsampled baseline.
- **EAR** classification has the lowest performance, below baseline about 2 percentage points.
- **PDANS** and **PU-GCN** Between EAR and PU-Net, there are no more best overall than baseline.

## 5. geometric metrics Contrast

| method | CD (Δ vs Original) | HD | NUC | best overall cls |
| --- | --- | --- | --- | --- |
| Downsampled x4 baseline | 0.077 (+0.028) | 0.214 | 2.049 | 90.85% |
| + EAR | 0.081 (+0.032) | 0.209 | 0.958 | 88.81% |
| + PDANS | 0.063 (+0.014) | 0.206 | 1.090 | 90.34% |
| + PU-Net | 0.057 (+0.008) | 0.138 | 1.058 | **91.27%** |
| + PU-GCN | 0.055 (+0.005) | 0.165 | 1.486 | 90.06% |

- PU-GCN and PU-Net are in **CD** It is the closest Original baseline (geometric is better to rebuild).
- **PU-GCN** It's... **NUC** Obviously above other methods (1.49 vs ~ 1.06) are less evenly distributed.
- **PU-Net** It's more balanced on HD and NUC.
- The geometric CD method (PU-GCN, PU-Net) does not always correspond to a higher classification accuracy: PU-GCN CD is the lowest but classification below PU-Net.
- EAR geometric metrics is medium, but classification is the worst performer - geometric and classification are not clear on branch.

## 6. Careful conclusion

- Current Line B data:**256  /  1024 upsampling does not promote classification performance in all methods**; only PU-Net brings a small increase.
- Between geometric quality (CD/HD/P2F/NUC) and downstream classification**There is no simple monotonous relationship**.
- It is not appropriate for only to extrapolate classification from a single indicator of CD;The findings of HD, P2F, NUC and classification need to be combined.
- This report, only, records Line B observations, does not cross Line or eventually thesis conclusions.
