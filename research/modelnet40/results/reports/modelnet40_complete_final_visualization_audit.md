# Complete Final Visualization Audit

- Generated: `2026-07-17 08:35:11 UTC`
- Overall: **PASS**

| Check | Status | Detail |
|---|---|---|
| point cloud static included | PASS | copied=10 |
| point cloud interactive included | PASS | html_count=52 |
| geometry Line B focus included | PASS |  |
| geometry supplementary included | PASS |  |
| classification Line A split | PASS |  |
| classification Line B split | PASS |  |
| combined geometry vs classification | PASS |  |
| all upsampling methods in geometry | PASS | ['EAR', 'PDANS', 'PU-Net', 'PU-GCN', 'PU-EdgeFormer'] |
| Line A methods include EdgeFormer | PASS |  |
| Line B methods include EdgeFormer | PASS |  |
| both lines included | PASS |  |
| classification CSV has pp columns | PASS |  |
| no fabricated values | PASS | read from existing reports |
| final package figures generated | PASS | png_count=33 |

## Reference logic

- Line B geometry delta: vs Original baseline 1024
- Line B classification primary: vs Downsampled ×4 baseline 256
- Line B classification secondary gap: vs Original baseline 1024
- Line A classification: vs Original baseline 1024
- Delta figures: no baseline bar; y=0 reference line
- Delta units: pp (percentage points)

## Notes

- `pointcloud_examples_v2/` static directory not found; static index uses `pointcloud_examples/`.
- Legacy static panels omit PU-EdgeFormer; interactive HTML covers all methods.
- No retraining / no geometry recomputation / no dataset modification.
