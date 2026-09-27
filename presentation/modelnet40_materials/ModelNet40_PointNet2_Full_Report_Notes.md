# ModelNet40 PointNet2 Full Report — Notes

- Generated: 2026-08-04 11:52:39 UTC
- PPTX: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/presentations/ModelNet40_PointNet2_Full_Report.pptx`
- PDF: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/presentations/ModelNet40_PointNet2_Full_Report.pdf`
- Slides: 28

## Numbers verified from metrics.json

- Original Best OA: 91.95%
- Mesh-ref 4096 Best OA: 92.00%
- Line A Δ vs Mesh-ref 4096:
  - PU-GCN: 91.63% (-0.36 pp)
  - PDANS: 91.62% (-0.38 pp)
  - EAR: 91.48% (-0.52 pp)
  - PU-Net: 90.95% (-1.05 pp)
  - PU-EdgeFormer: 90.54% (-1.46 pp)

## Why retrain (included)

- Dedicated English + Chinese slides explaining PointNet++ SSG from-scratch retrain
- Clarifies: not finetune; not training upsampler; matched num_point; independent checkpoints

## Regenerate

```bash
python scripts/generate_full_thesis_presentation.py
```
