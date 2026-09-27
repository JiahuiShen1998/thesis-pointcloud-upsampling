# ModelNet40 PointNet++ Thesis Result Analysis

- Generated at: 2026-07-12 17:58:43 UTC
- Scope: finalized two-line protocol on ModelNet40 with PointNet++

## 1. Experimental setup summary

This study evaluates point cloud upsampling on **ModelNet40** using **PointNet++** as the downstream classifier under a **two-line protocol**:

- **Line A:** Original 1024-point baseline vs **Original + Upsampling** at 4096 points.
- **Line B:** Downsampled ×4 (256-point) baseline vs **Downsampled ×4 + Upsampling** at 1024 points.

Upsampling methods: **EAR**, **PDANS**, **PU-Net**, **PU-GCN**.

Geometry quality is measured against dense mesh references using **CD**, **HD**, **exact P2F**, and **NUC** (lower is better). Classification uses **final** and **best overall / class accuracy** from 200-epoch training (seed=42).

## 2. Geometric quality analysis

**Original baseline** (1024 pts) is the reference: CD=0.049548, HD=0.119720, exact P2F=0.082469, NUC=1.061943.

In **Line B**, **PU-GCN** and **PU-Net** achieve the smallest CD deltas vs Original (PU-GCN ΔCD=0.005279, PU-Net ΔCD=0.007567).

**PU-Net** is more balanced on **HD** (ΔHD=0.017954) and **NUC** (ΔNUC=-0.003731).

**PU-GCN** shows competitive CD but clearly elevated **NUC** (ΔNUC=0.424526), indicating less uniform point distribution despite good Chamfer distance.

**exact P2F** is fully computed (not approximate or pending), providing mesh-surface distance evidence alongside CD/HD/NUC.

No single geometry metric should dominate interpretation: e.g., EAR has relatively low NUC but poor CD/HD; PDANS improves CD over EAR but HD remains high.

## 3. Line B downstream classification analysis

**Downsampled ×4 baseline** (256 pts) reaches best overall accuracy **90.85%** and final overall **90.46%**.

**PU-Net** achieves the highest best overall accuracy at **91.27%** (+0.42pp vs baseline) and is the **only** upsampling method slightly above the downsampled baseline on best overall.

EAR (−2.04 pp), PDANS (−0.51 pp), and PU-GCN (−0.79 pp) all remain below baseline on best overall accuracy.

These results show that recovering point count after ×4 downsampling does **not** automatically improve PointNet++ classification; only PU-Net yields a marginal gain.

## 4. Line A downstream classification analysis

**Original baseline** (1024 pts): best overall **91.95%**, final overall **91.31%**.

All **Original + Upsampling** methods at 4096 points remain **below** the 1024 baseline on both best and final overall accuracy:

- Original + EAR 4096: best 91.48% (-0.47pp), final 90.85% (-0.46pp)
- Original + PDANS 4096: best 91.62% (-0.33pp), final 90.87% (-0.44pp)
- Original + PU-Net 4096: best 90.95% (-1.00pp), final 90.55% (-0.76pp)
- Original + PU-GCN 4096: best 91.63% (-0.32pp), final 91.00% (-0.32pp)

Increasing to 4096 points does **not** produce a stable classification benefit in Line A; the native 1024-point baseline remains strongest.

## 5. Geometry vs classification discussion

Geometric quality and classification performance are **not fully aligned**.

PU-GCN has the best Line B CD (Δ=0.005279) but classifies below PU-Net (CD=0.054827 vs 91.27% best overall for PU-Net).

PU-Net's stronger classification may relate to more balanced **HD** and **NUC**, suggesting PointNet++ is sensitive to local structure and point distribution—not only global Chamfer error.

Geometric recovery quality and downstream task utility should be reported and interpreted separately.

## 6. Thesis-ready conclusion

- **Upsampling does not universally improve downstream classification.**
- **Some methods improve geometric quality but do not improve classification** (e.g., PU-GCN on CD vs classification; PDANS partial geometry gains without classification gains).
- **PU-Net appears to be the most promising Line B method** in this experiment, as the only upsampling variant slightly exceeding the downsampled baseline.
- **The effect of point cloud upsampling depends on the upsampling method and the downstream task.**
