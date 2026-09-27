# Line A — Original + Upsampling ×4 Count Audit

- Generated at: 2026-07-02 19:25:37 UTC
- Input N: 1024
- Expected output 4N: 4096

| method | train | test | shape | status |
| --- | ---: | ---: | --- | --- |
| EAR | 9843 | 2468 | (4096, 3) | reused_legacy |
| PU-Net | 9843 | 2468 | (4096, 3) | reused_legacy |
| PU-GCN | 0 | 0 | None | legacy_incomplete |
| PDANS | 9843 | 2468 | (4096, 3) | reused_legacy |