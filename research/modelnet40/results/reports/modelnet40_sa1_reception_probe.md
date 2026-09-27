# What PointNet++ actually receives at SA1 — a reception probe

- Generated: 2026-08-17
- Split: test, N = 2468
- Layer: `pointnet2_cls_ssg.sa1` — npoint=512, radius=0.2, nsample=32
- Script: `scripts/compute_sa1_reception_probe.py`
- Raw: `reports/modelnet40_sa1_reception_probe.json`

## Why this probe exists

SA1 consumes the raw (re-normalised) input cloud, so its intake is measurable directly from the
data, without training anything. `models/pointnet2_utils.py :: query_ball_point` shows the layer
has a **fixed reading capacity of 32 points per neighbourhood**:

- more than 32 points inside the radius -> the surplus is **discarded** (survivors are the lowest
  storage indices, i.e. file order);
- fewer than 32 -> the empty slots are **filled by repeating the first in-ball point**.

So a cloud can fail the layer in two opposite ways, and both are countable.

## Results

| Pts | Cloud | Mean in ball | Discarded by cap | Slots that are repeats | Saturated balls | Starved balls | Density CV | kept/random spread | Best OA |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | Downsampled ×4 | 14.5 | 3.1 % | 59.2 % | 6.9 % | 92.6 % | 0.406 | 1.000 | 90.85 % |
| 256 | Mesh-ref 256 | 14.5 | 3.1 % | 59.2 % | 6.9 % | 92.7 % | 0.407 | 1.000 | 90.88 % |
| 1024 | Original | 54.8 | 31.9 % | 12.1 % | 61.1 % | 37.1 % | 0.363 | 1.001 | 91.95 % |
| 1024 | EAR | 72.3 | 53.0 % | 14.5 % | 67.0 % | 32.1 % | 0.645 | 1.054 | 88.81 % |
| 1024 | PDANS | 58.7 | 36.1 % | 9.0 % | 68.6 % | 29.6 % | 0.379 | 1.025 | 90.34 % |
| 1024 | PU-Net | 55.1 | 31.3 % | 10.8 % | 63.0 % | 35.0 % | 0.333 | 1.001 | 91.27 % |
| 1024 | PU-GCN | 52.7 | 30.7 % | 13.1 % | 59.2 % | 39.0 % | 0.374 | 1.016 | 90.06 % |
| 1024 | PU-EdgeFormer | 57.6 | 35.9 % | 11.1 % | 65.0 % | 33.2 % | 0.417 | 0.985 | 89.32 % |
| 4096 | Mesh-ref 4096 | 213.4 | 78.7 % | 0.2 % | 99.3 % | 0.6 % | 0.352 | 1.002 | 92.00 % |
| 4096 | EAR | 234.5 | 81.7 % | 2.1 % | 92.3 % | 7.2 % | 0.661 | 1.086 | 91.48 % |
| 4096 | PDANS | 215.1 | 78.9 % | 0.1 % | 99.3 % | 0.6 % | 0.359 | 1.034 | 91.62 % |
| 4096 | PU-Net | 216.6 | 79.2 % | 0.1 % | 99.5 % | 0.5 % | 0.319 | 0.997 | 90.95 % |
| 4096 | PU-GCN | 211.1 | 78.1 % | 0.1 % | 99.5 % | 0.5 % | 0.334 | 1.026 | 91.63 % |
| 4096 | PU-EdgeFormer | 222.2 | 79.8 % | 0.2 % | 99.3 % | 0.7 % | 0.365 | 0.970 | 90.54 % |

## Reading

1. **The density curve's shape is explained by the two failure modes trading places.**

   | Point count | Slots that are repeats | Discarded by the cap | Best OA |
   |---:|---:|---:|---:|
   | 256 | 59.2 % | 3.1 % | 90.88 % |
   | 1024 | 12.1 % | 31.9 % | 91.95 % |
   | 4096 | 0.2 % | 78.7 % | 92.00 % |

   Going 256 -> 1024 eliminates the padding that filled 59 % of every neighbourhood: +1.07 pp.
   Going 1024 -> 4096 has almost no padding left to remove (12 % -> 0.2 %) while the discarded
   share rises to 78.7 %, i.e. the extra points cannot reach the layer at all: +0.05 pp.

2. **Densification therefore has no channel through which to help.** Any gain an upsampler could
   offer at 4096 must pass through a layer that is already discarding four fifths of what it is
   given. The error an upsampler introduces, by contrast, is fully visible: it changes which 32
   points populate each ball and where the FPS centroids land.

3. **EAR is the clearest illustration.** Its density CV is 0.645 at 1024 and 0.661 at 4096 against
   roughly 0.33-0.42 for every other variant, and it is simultaneously the most saturated (53.0 %
   discarded at 1024) and among the most starved (14.5 % of slots repeated). Concentrating points
   on edges wastes them where they pile up and starves the layer where they do not. EAR is last in
   the 1024 bucket at 88.81 %.

4. **The two 256-point constructions are indistinguishable to the layer** (14.5 in ball, 3.1 %
   discarded, 59.2 % repeated for both), which is why they classify within 0.03 pp of each other.

## Ruled out

Because survivors are chosen by storage index, an upsampler whose output ordering correlated with
position would hand the layer a spatially clustered sample. Measured: spread(kept 32) /
spread(random 32 in-ball) lies in **0.970-1.086** across all 14 variants. The surviving subset is
spatially representative, so index ordering is **not** the mechanism.

## Still open

No single statistic here orders the accuracy *within* the 4096 bucket. Spearman rank correlations
with Best OA over the five upsampling methods are +0.30 (CD), +0.10 (HD) and -0.20 (|dNUC|) at
4096, against +0.10 (CD), +0.70 (HD) and +0.70 (|dNUC|) at 1024. Once the layer saturates the
remaining differences are small (1.09 pp spread vs 2.46 pp at 1024) and unexplained. With n = 5
methods and a single seed per branch these coefficients are descriptive, not inferential.

## Caveat

SA1 requests 512 FPS centroids. For 256-point clouds that exceeds the cloud size, so the
centroid set necessarily repeats points. This affects both 256-point variants identically and does
not bias the comparison between them, but it does mean the 256 bucket runs a degenerate SA1.
