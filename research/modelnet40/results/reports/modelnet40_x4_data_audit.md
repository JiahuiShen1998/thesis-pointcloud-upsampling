# ModelNet40 ×4 Protocol — Data Audit

- Generated at: 2026-07-02 19:23:03 UTC
- PROJECT_ROOT: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`

## Protocol point counts (detected from data)

| Symbol | Value |
| --- | ---: |
| N (original baseline) | 1024 |
| N/4 (downsampled ×4) | 256 |
| 4N (Line A upsampling output) | 4096 |
| N (Line B upsampling output) | 1024 |

## Original ModelNet40

- Train: 9843 (expected 9843)
- Test: 2468 (expected 2468)
- Classes: 40 (expected 40)
- Dominant shape: (1024, 3)

## Downsampled ×4 database (`modelnet40_downsampled_x4`)

- Path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4`
- Exists: True
- Valid for reuse: **True**
- Counts: train=9843, test=2468
- Expected point count: 256
- Dominant shape: (256, 3)

## Legacy `modelnet40_downsampled50` (preserved, NOT x4)

- Path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled50`
- Dominant shape: (512, 3) (N/2, old protocol)
- **Do not use as x4 downsampled database.**

## Legacy Line A upsampling outputs (1024→4096)

| method | exists | train | test | shape | reusable for Line A |
| --- | --- | ---: | ---: | --- | --- |
| ear | True | 9843 | 2468 | (4096, 3) | True |
| punet | True | 9843 | 2468 | (4096, 3) | True |
| pugcn | True | 0 | 0 | None | False |
| pdans | True | 9843 | 2468 | (4096, 3) | True |

## Legacy Line B upsampling outputs (512→2048, OLD protocol)

| method | exists | train | test | shape | valid for NEW Line B |
| --- | --- | ---: | ---: | --- | --- |
| ear | True | 9843 | 2468 | (2048, 3) | **NO** (need 256→1024) |
| punet | True | 6609 | 2468 | (2048, 3) | **NO** (need 256→1024) |
| pugcn | True | 0 | 0 | None | **NO** (need 256→1024) |
| pdans | True | 9843 | 2468 | (2048, 3) | **NO** (need 256→1024) |

## Mesh / P2F availability

- Mesh root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/data/raw/ModelNet40`
- .off files in mesh root: 12311
- Manifest-linked .off available: 12311
- Manifest-linked .off missing: 0
- P2F status: **available**

## Old protocol artifacts (preserved)

- `downsampled50` (512 pts) — old Line B baseline
- `downsampled50_up/*_x4` (2048 pts) — old Line B upsampling
- `512→2048` PointNet++ results — superseded by new Line B (256→1024)
