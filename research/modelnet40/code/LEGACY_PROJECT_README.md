# ModelNet40 PointNet++ Upsampling — Share Package

Thesis materials for **geometry vs Original** evaluation and **two-line PointNet++ classification**.

## Quick links (after hosting)

| Item | Path |
|------|------|
| Editable PPTX | [`presentations/ModelNet40_PointNet2_Original_Reference_Revised.pptx`](presentations/ModelNet40_PointNet2_Original_Reference_Revised.pptx) |
| PDF companion | [`presentations/ModelNet40_PointNet2_Original_Reference_Revised.pdf`](presentations/ModelNet40_PointNet2_Original_Reference_Revised.pdf) |
| Revision notes | [`presentations/ModelNet40_Presentation_Revision_Notes.md`](presentations/ModelNet40_Presentation_Revision_Notes.md) |
| Interactive 3D HTML | [`figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/`](figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/) |

## Protocol (short)

- **Geometry reference:** corresponding Original **1024** point cloud only (no mesh / no P2F).
- **Line A:** Original → ×4 up → **4096**; PointNet++ **retrained from scratch at 4096**.
- **Line B:** Down ×4 (**256**) → ×4 up → **1024**; PointNet++ **retrained at 256 / 1024**.
- **Unequal cardinality** (4096 vs 1024; 256 vs 1024): CD/HD = discrete-set consistency, not continuous-surface accuracy.
- **Best Overall:** highest checkpoint accuracy; ModelNet40 PointNet++ setup has **no val split** (test used for checkpoint selection).

## Key results

- Line A: Original 1024 best (**91.95%**); no 4096-retrained branch improves classification.
- Line B: **PU-Net** only above Downsampled 256 (**91.27%**, **+0.42 pp**); still **−0.68 pp** below Original.

## What is NOT in this repo

Large local artifacts are gitignored (~12G total project):

- `datasets/`, `quality_metrics/`, `pointnet2_results/`, `logs/`, `outputs/`, checkpoints

Ask the author for data access on the HPC workspace if needed.

## Regenerate presentation

```bash
cd modelnet40_pointnet2_upsampling
# After GitHub/GitLab Pages is live:
export PUBLIC_INTERACTIVE_BASE="https://USER.github.io/REPO/figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2"
python scripts/generate_original_reference_presentation.py
```

## License / academic use

Internal thesis materials. Contact the repository owner before redistribution.
