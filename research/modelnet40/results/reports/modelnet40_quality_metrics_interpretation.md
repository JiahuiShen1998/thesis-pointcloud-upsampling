# ModelNet40 Quality Metrics Interpretation

- Generated at: 2026-07-07 07:03:13 UTC

Original baseline is the **reference quality level** for this protocol. All other branches should be interpreted relative to that row rather than in isolation.

For the main Line B comparison, **Downsampled x4 + Upsampling** methods are compared directly against the Original baseline. On **CD**, **PU-GCN** and **PU-Net** are the closest to Original baseline among the Line B upsampling methods, with PU-GCN showing the smallest delta and PU-Net the second smallest.

However, the final interpretation should **not** rely on CD alone. It must also consider **HD**, **exact P2F**, and **NUC**. For example, PU-Net is stronger than EAR/PDANS on HD, while PU-GCN has competitive CD but noticeably worse NUC than Original baseline. EAR has the worst CD/HD among the Line B methods, even though its NUC is lower than Original baseline.

**exact P2F is completed**, not pending. This means the current geometry analysis already includes surface-distance evidence from the original `.off` meshes, so no pending mesh-distance caveat remains for the present tables.

These geometric metrics should next be analyzed together with the later **PointNet++ classification accuracy** results. The key question is whether methods that look geometrically closer to Original baseline also preserve downstream recognition performance better after the x4 downsample-and-recover pipeline.
