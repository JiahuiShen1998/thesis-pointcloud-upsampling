# Line B — Downsampled ×4 + Upsampling Count Audit

- Generated at: 2026-07-02 20:23:16 UTC
- Downsampled input N/4: 256
- Expected upsampling output N: 1024

**Note:** Legacy `downsampled50_up/*_x4` (512→2048) is NOT valid for this protocol.

| method | train | test | shape | status |
| --- | ---: | ---: | --- | --- |
| EAR | 298 | 286 | (1024, 3) | partial |
| PU-Net | 0 | 0 | (2048, 3) | legacy_wrong_protocol_512_to_2048 |
| PU-GCN | 0 | 0 | None | pending_generation |
| PDANS | 0 | 0 | (2048, 3) | legacy_wrong_protocol_512_to_2048 |