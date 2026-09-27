# PU-GCN detector adaptation: full KITTI validation

Required frame count per arm: **3769**.

| Detector | Line | Input | Weights | Frames | Car 3D AP_R40 E/M/H | Δ moderate | Status |
|---|---|---|---|---:|---:|---:|---|
| PointRCNN | A | baseline N | official | 3769 | 92.4325/81.9528/77.8468 | +0.0000 | PASS |
| PointRCNN | A | PU-GCN direct 4N | official | 3769 | 84.7443/61.9294/57.7869 | -20.0235 | PASS |
| PointRCNN | A | PU-GCN direct 4N | adapted | 3769 | 85.1671/68.8800/64.6697 | -13.0728 | PASS |
| PointRCNN | A | observed N + predicted 3N | official | 3769 | 85.0374/64.2748/60.0911 | -17.6780 | PASS |
| PointRCNN | A | observed N + predicted 3N | adapted | 3769 | 86.7068/71.0075/66.7939 | -10.9453 | PASS |
| PointRCNN | B | baseline M | official | 3769 | 85.2016/65.5757/61.2831 | +0.0000 | PASS |
| PointRCNN | B | baseline M | adapted | 3769 | 84.3920/68.3312/64.2613 | +0.0000 | PASS |
| PointRCNN | B | PU-GCN direct 4M | official | 3769 | 46.4657/30.0909/25.8000 | -35.4848 | PASS |
| PointRCNN | B | PU-GCN direct 4M | adapted | 3769 | 68.2592/46.6150/40.2487 | -21.7163 | PASS |
| PointRCNN | B | observed M + predicted 3M | official | 3769 | 55.0177/35.5289/30.9359 | -30.0469 | PASS |
| PointRCNN | B | observed M + predicted 3M | adapted | 3769 | 74.1970/55.2145/49.0583 | -13.1167 | PASS |
| CenterPoint | A | baseline N | official | 3769 | 88.3907/79.2773/76.7371 | +0.0000 | PASS |
| CenterPoint | A | PU-GCN direct 4N | official | 3769 | 81.8888/60.6081/57.9061 | -18.6692 | PASS |
| CenterPoint | A | observed N + predicted 3N | official | 3769 | 83.3121/63.2064/61.0418 | -16.0709 | PASS |
| CenterPoint | A | observed N + predicted 3N | adapted | 3769 | 86.8154/74.7012/72.6570 | -4.5761 | PASS |
| CenterPoint | B | baseline M | official | 3769 | 81.4003/64.5979/59.9701 | +0.0000 | PASS |
| CenterPoint | B | baseline M | adapted | 3769 | 83.7514/68.0490/64.6309 | +0.0000 | PASS |
| CenterPoint | B | PU-GCN direct 4M | official | 3769 | 56.4211/35.3530/31.2043 | -29.2449 | PASS |
| CenterPoint | B | observed M + predicted 3M | official | 3769 | 66.9217/44.0929/40.0268 | -20.5050 | PASS |
| CenterPoint | B | observed M + predicted 3M | adapted | 3769 | 80.4109/61.6639/57.5252 | -6.3851 | PASS |

Complete arms: **20/20**.

For Line A adapted rows, no separately adapted original-N baseline was trained; their displayed delta therefore references the official Line A baseline and changes both input and weights. Other deltas use the same detector-weight family when available.
