# ModelNet40 Results Presentation — Speaker Notes

File: `presentations/ModelNet40_PointNet2_Results_Presentation.pptx` (27 slides)

## How to present (≈12–15 min)

### Opening (slides 1–3)
- **Goal:** Evaluate whether point cloud upsampling helps geometry *and* PointNet++ classification on ModelNet40.
- **Two lines:**
  - **Line A:** Original 1024 → upsample to 4096 (densification).
  - **Line B:** Downsample to 256 → upsample back to 1024 (recovery).
- **Methods:** EAR, PDANS, PU-Net, PU-GCN, PU-EdgeFormer.

### Geometry (slides 4–5)
- Metrics: **CD, HD, NUC, exact P2F** vs dense mesh (lower better).
- Line B: **PU-GCN / PU-Net** closest on CD; **PU-GCN NUC** clearly worse; **PU-Net** more balanced.
- Emphasize: mesh reference ≠ experimental baseline.

### Point clouds (slides 6–14)
- Show Line B grids (Original / Down256 / EAR / PDANS / PU-Net / PU-GCN).
- Then Line A grids (Original / 4096 ups.).
- Optional live demo: open HTML under `figures/modelnet40/pointcloud_examples_interactive_v2/`.

### Classification Line A (slides 15–16)
- **Original 1024 = 91.95%** best.
- All 4096 upsampling methods **below** baseline (PU-EdgeFormer worst, −1.41 pp).
- Takeaway: denser input ≠ better PointNet++ accuracy here.

### Classification Line B (slides 17–19)
- Downsampled baseline **90.85%**.
- **Only PU-Net** above it: **91.27% (+0.42 pp)**.
- Still below Original 1024 (gap −0.68 pp for PU-Net).

### Three comparisons + geometry vs cls (slides 20–23)
1. Geometry quality  
2. Line A classification  
3. Line B classification  
- **Geometry best ≠ classification best** (PU-GCN CD vs PU-Net accuracy).

### Close (slides 24–27)
- Upsampling is **method- and task-dependent**, not universal.
- Thesis tip: tables keep baseline rows; main figures use methods-only deltas.
- Interactive HTML for demo; static PNGs for thesis PDF.

## One-sentence summary for supervisor

> On ModelNet40 with PointNet++, upsampling does not generally improve classification; Line A never beats Original 1024, and in Line B only PU-Net gains a small +0.42 pp over the downsampled baseline, while geometry and accuracy rankings disagree.

## Key numbers card

| Item | Value |
|------|-------|
| Line A best | Original 1024 → **91.95%** |
| Line B baseline | Down ×4 256 → **90.85%** |
| Line B best upsampling | PU-Net → **91.27% (+0.42 pp)** |
| Line A all ups. | all negative vs Original |
| Geometry vs cls | not aligned |

## Live demo links (local)

- Line B chair depth grid:  
  `figures/modelnet40/pointcloud_examples_interactive_v2/lineB/lineB_chair_chair_0890_interactive_grid_depth.html`
- Line B dropdown:  
  `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown/lineB_chair_chair_0890_dropdown.html`
