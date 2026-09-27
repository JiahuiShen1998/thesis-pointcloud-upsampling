# ModelNet40 Upsampling Ratio Audit

- Generated at: 2026-06-29 14:07:37 UTC
- Project: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

## Main Methods (xyz upsampling)

EAR, PU-Net, PU-GCN, PDANS — main protocol **R=×4**.

- Line A upsampling target: **4096** (1024 → ×4)
- Line B upsampling target: **2048** (512 → ×4)

| Line | Method | Dataset | Input | Target | Output (min/max/mean) | Actual ratio | Configured | Status |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| Line A | EAR | `datasets/modelnet40_original_up/ear_x4` | 1024 | 4096 | 4096/4096/4096.0 | 4.0000–4.0000 (μ=4.0000) | target_n=4096 / up_factor=4.0 (main protocol R=×4) | **main_compatible** |
| Line A | EAR | `datasets/modelnet40_original_up/ear_x4_smoke` | 1024 | 4096 | 4096/4096/4096.0 | 4.0000–4.0000 (μ=4.0000) | target_n=4096 / up_factor=4.0 → 1024→4096 | **main_smoke_pass** |
| Line B | EAR | `datasets/modelnet40_downsampled50_up/ear_x4` | 512 | 2048 | 2048/2048/2048.0 | 4.0000–4.0000 (μ=4.0000) | target_n=2048 / up_factor=4.0 (main protocol R=×4) | **main_compatible** |
| Line B | EAR | `datasets/modelnet40_downsampled50_up/ear_x4_smoke` | 512 | 2048 | 2048/2048/2048.0 | 4.0000–4.0000 (μ=4.0000) | target_n=2048 / up_factor=4.0 → 512→2048 | **main_smoke_pass** |
| Line A | EAR | `datasets/modelnet40_original_up/ear` | 1024 | 4096 | — | — | legacy ×2 wrapper: target_n=1024 / up_factor=2.0 (identity at 1024 input) | **ablation_preliminary_x2** |
| Line B | EAR | `datasets/modelnet40_downsampled50_up/ear` | 512 | 2048 | 1024/1024/1024.0 | 2.0000–2.0000 (μ=2.0000) | legacy ×2: target_n=1024 / up_factor=2.0 → 512→1024 | **ablation_preliminary_x2** |
| Line A | PDANS | `datasets/modelnet40_original_up/pdans` | 1024 | 4096 | — | — | R=4 (default, configurable via --R) | **pending_generation** |
| Line B | PDANS | `datasets/modelnet40_downsampled50_up/pdans` | 512 | 2048 | — | — | R=4 (default) or R=2 (planned) | **pending_generation** |
| Line A | PU-Net | `datasets/modelnet40_original_up/punet` | 1024 | 4096 | — | — | up_ratio=4 (default in main.py) | **pending_generation** |
| Line B | PU-Net | `datasets/modelnet40_downsampled50_up/punet` | 512 | 2048 | — | — | up_ratio=4 (default); configurable | **pending_generation** |
| Line A | PU-GCN | `datasets/modelnet40_original_up/pugcn` | 1024 | 4096 | — | — | up_ratio=4 (default in configs.py) | **pending_generation** |
| Line B | PU-GCN | `datasets/modelnet40_downsampled50_up/pugcn` | 512 | 2048 | — | — | up_ratio=4 (default); configurable | **pending_generation** |

## Supplementary / Special Method

### TULIP

**Tags:** range-image-based / LiDAR-style upsampling candidate / supplementary candidate

TULIP is **not** grouped with EAR / PU-Net / PU-GCN / PDANS for main-protocol ratio alignment.
It uses KITTI range-image upsampling (16×1024 low-res → 64×1024 high-res); theoretical image upsampling is **×4**.

| Line | Method | Dataset | Input | Output (min/max/mean) | Actual ratio | Configured | Status |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| Line A | TULIP | `datasets/modelnet40_original_up/tulip` | 1024 | — | — | range-image ×4 (16×1024 → 64×1024); KITTI-trained tulip_base | **supplementary_only** |
| Line B | TULIP | `datasets/modelnet40_downsampled50_up/tulip` | 512 | — | — | range-image ×4 (16×1024 → 64×1024); xyz count not fixed | **supplementary_only** |

#### ModelNet40 TULIP scan result

- `datasets/modelnet40_original_up/tulip/` — **not created** (0 `.npy` / `.ply` / `.bin`)
- `datasets/modelnet40_downsampled50_up/tulip/` — **not created** (0 `.npy` / `.ply` / `.bin`)
- No ModelNet40 wrapper or generation job exists in `modelnet40_pointnet2_upsampling`.

#### KITTI supplementary reference (xyz point counts, not ModelNet40)

Used only to characterize TULIP xyz behavior when back-projected from range images.

- **Line A (KITTI original→tulip, n=3769):** input 78596/127452/118708 → output 29795/52500/49240; ratio 0.3590–0.4428 (μ=0.4149, σ=0.0085)
- **Line B (KITTI downsampled50→tulip, n=3769):** input 39298/63726/59354 → output 26366/48996/44740; ratio 0.6454–0.8105 (μ=0.7537, σ=0.0212)

## Per-Method Notes

### Line A — EAR (`datasets/modelnet40_original_up/ear_x4`)

- Group: **main**
- Status: **main_compatible**

- Train/test file count: 9843 / 2468
- Ratio source: `scripts/ear_modelnet40_utils.py:TARGET_POINTS_X4_ORIGINAL=4096; UP_FACTOR_X4=4.0`
- Scripts: scripts/run_ear_x4_smoke.py; scripts/run_ear_x4_chunk.py
- Notes: EAR ×4 main: 1024→4096, ratio≈4

### Line A — EAR (`datasets/modelnet40_original_up/ear_x4_smoke`)

- Group: **main**
- Status: **main_smoke_pass**

- Train/test file count: 20 / 20
- Ratio source: `scripts/ear_modelnet40_utils.py:TARGET_POINTS_X4_ORIGINAL=4096`
- Scripts: scripts/run_ear_x4_smoke.py
- Notes: Smoke EAR ×4: 1024→4096, ratio≈4

### Line B — EAR (`datasets/modelnet40_downsampled50_up/ear_x4`)

- Group: **main**
- Status: **main_compatible**

- Train/test file count: 9843 / 2468
- Ratio source: `scripts/ear_modelnet40_utils.py:TARGET_POINTS_X4_DOWN=2048; UP_FACTOR_X4=4.0`
- Scripts: scripts/run_ear_x4_smoke.py; scripts/run_ear_x4_chunk.py
- Notes: EAR ×4 main: 512→2048, ratio≈4

### Line B — EAR (`datasets/modelnet40_downsampled50_up/ear_x4_smoke`)

- Group: **main**
- Status: **main_smoke_pass**

- Train/test file count: 20 / 20
- Ratio source: `scripts/ear_modelnet40_utils.py:TARGET_POINTS_X4_DOWN=2048`
- Scripts: scripts/run_ear_x4_smoke.py
- Notes: Smoke EAR ×4: 512→2048, ratio≈4

### Line A — EAR (`datasets/modelnet40_original_up/ear`)

- Group: **main**
- Status: **ablation_preliminary_x2**

- Train/test file count: 0 / 0
- Ratio source: `scripts/ear_modelnet40_utils.py:TARGET_POINTS=1024 (legacy)`
- Scripts: scripts/run_ear_smoke.py (legacy)
- Notes: Legacy ×2 wrapper (target_n=1024); main protocol target=4096

### Line B — EAR (`datasets/modelnet40_downsampled50_up/ear`)

- Group: **main**
- Status: **ablation_preliminary_x2**

- Train/test file count: 9843 / 2468
- Ratio source: `scripts/ear_modelnet40_utils.py; scripts/run_ear_full_lineB_chunk.py`
- Scripts: scripts/run_ear_full_lineB_chunk.py
- Notes: Legacy EAR ×2 full output preserved: 512→1024; not main protocol (main target=2048)

### Line A — PDANS (`datasets/modelnet40_original_up/pdans`)

- Group: **main**
- Status: **pending_generation**

- Train/test file count: 0 / 0
- Ratio source: `external_lab_migrated/PDANS/pointnet2/samples.py:--R default 4`
- Scripts: (not yet in project; planned step7)
- Notes: No ModelNet40 outputs yet; main protocol R=×4, target=4096; main protocol R=×4, target=4096

### Line B — PDANS (`datasets/modelnet40_downsampled50_up/pdans`)

- Group: **main**
- Status: **pending_generation**

- Train/test file count: 0 / 0
- Ratio source: `external_lab_migrated/PDANS/pointnet2/example_eval.py:R=4`
- Scripts: (not yet in project)
- Notes: No ModelNet40 outputs yet; main protocol R=×4, target=2048; main protocol R=×4, target=2048

### Line A — PU-Net (`datasets/modelnet40_original_up/punet`)

- Group: **main**
- Status: **pending_generation**

- Train/test file count: 0 / 0
- Ratio source: `external_lab_migrated/PU-Net/code/main.py:--up_ratio default 4`
- Scripts: (not yet in project)
- Notes: No ModelNet40 outputs yet; main protocol R=×4, target=4096; main protocol R=×4, target=4096

### Line B — PU-Net (`datasets/modelnet40_downsampled50_up/punet`)

- Group: **main**
- Status: **pending_generation**

- Train/test file count: 0 / 0
- Ratio source: `external_lab_migrated/PU-Net/code/main.py:--up_ratio default 4`
- Scripts: (not yet in project)
- Notes: No ModelNet40 outputs yet; main protocol R=×4, target=2048; main protocol R=×4, target=2048

### Line A — PU-GCN (`datasets/modelnet40_original_up/pugcn`)

- Group: **main**
- Status: **pending_generation**

- Train/test file count: 0 / 0
- Ratio source: `external_lab_migrated/PU-GCN/Upsampling/configs.py:--up_ratio default 4`
- Scripts: (not yet in project)
- Notes: No ModelNet40 outputs yet; main protocol R=×4, target=4096; main protocol R=×4, target=4096

### Line B — PU-GCN (`datasets/modelnet40_downsampled50_up/pugcn`)

- Group: **main**
- Status: **pending_generation**

- Train/test file count: 0 / 0
- Ratio source: `external_lab_migrated/PU-GCN/Upsampling/configs.py:--up_ratio default 4`
- Scripts: (not yet in project)
- Notes: No ModelNet40 outputs yet; main protocol R=×4, target=2048; main protocol R=×4, target=2048

### Line A — TULIP (`datasets/modelnet40_original_up/tulip`)

- Group: **supplementary**
- Status: **supplementary_only**
- Tags: range-image-based / LiDAR-style upsampling candidate / supplementary candidate
- Train/test file count: 0 / 0
- Ratio source: `external_lab_migrated/TULIP/scripts/tulip_kitti_smoke_infer.py: img_size=(16,1024) target_img_size=(64,1024)`
- Scripts: external_lab_migrated/TULIP/scripts/tulip_kitti_smoke_infer.py; tulip/main_lidar_upsampling.py
- Notes: No ModelNet40 .npy/.ply/.bin outputs; output dir not created. TULIP is range-image-based / LiDAR-style — requires xyz→range-image adapter for objects. Tags: range-image-based / LiDAR-style upsampling candidate / supplementary candidate. KITTI supplementary reference (n=3769): xyz output min/max/mean=29795/52500/49239.7; actual xyz ratio min/max/mean=0.3590/0.4428/0.4149 (std=0.0085). Theoretical range-image upsampling is ×4 (16-line→64-line), but xyz point ratio is not ×4 and varies per frame due to valid-pixel mask / projection / filtering. Status: supplementary_only — not aligned with fixed-R xyz main protocol.

### Line B — TULIP (`datasets/modelnet40_downsampled50_up/tulip`)

- Group: **supplementary**
- Status: **supplementary_only**
- Tags: range-image-based / LiDAR-style upsampling candidate / supplementary candidate
- Train/test file count: 0 / 0
- Ratio source: `external_lab_migrated/TULIP/tulip/model/tulip.py; TULIP_pointRCNN_consistency.md`
- Scripts: external_lab_migrated/TULIP/scripts/convert_tulip_ply_to_kitti_bin.py
- Notes: No ModelNet40 .npy/.ply/.bin outputs; output dir not created. TULIP is range-image-based / LiDAR-style — requires xyz→range-image adapter for objects. Tags: range-image-based / LiDAR-style upsampling candidate / supplementary candidate. KITTI supplementary reference (n=3769): xyz output min/max/mean=26366/48996/44739.9; actual xyz ratio min/max/mean=0.6454/0.8105/0.7537 (std=0.0212). Theoretical range-image upsampling is ×4 (16-line→64-line), but xyz point ratio is not ×4 and varies per frame due to valid-pixel mask / projection / filtering. Status: supplementary_only — not aligned with fixed-R xyz main protocol.

## Baseline Native Counts (no upsampling)

| Line | Dataset | Native points |
| --- | --- | ---: |
| A | `datasets/modelnet40_original` | 1024 |
| B | `datasets/modelnet40_downsampled50` | 512 |

See `reports/modelnet40_upsampling_ratio_recommendation.md` for unified R=×4 main protocol.