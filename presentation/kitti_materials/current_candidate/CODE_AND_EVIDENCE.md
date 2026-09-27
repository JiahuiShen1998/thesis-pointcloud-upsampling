# PU-GCN detector adaptation: code and evidence companion

## 1. Terminology

PU-GCN was **not retrained on KITTI** in this experiment. It used the fixed released PU1K `model-100` checkpoint. “Unadapted” means the detector used its original official weights; “adapted” means the detector was initialized from the official checkpoint and adapted for three epochs on the matching input distribution.

Primary result source: `../pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.csv`.

## 2. Original unadapted vs 3-epoch adapted detectors

All values are KITTI Car 3D Moderate AP_R40 on all 3,769 validation frames. Each row holds detector, line, and PU-GCN input fixed; only detector weights change.

| Detector / input | Unadapted | Adapted | Gain |
|---|---:|---:|---:|
| PointRCNN A direct | 61.9294 | 68.8800 | +6.9506 |
| PointRCNN A observed-first | 64.2748 | 71.0075 | +6.7327 |
| PointRCNN B direct | 30.0909 | 46.6150 | +16.5241 |
| PointRCNN B observed-first | 35.5289 | 55.2145 | +19.6856 |
| CenterPoint A observed-first | 63.2064 | 74.7012 | +11.4948 |
| CenterPoint B observed-first | 44.0929 | 61.6639 | +17.5710 |

All six paired PU-GCN arms improve after detector adaptation. No tested adapted PU-GCN arm reaches its reference baseline; the remaining Car Moderate 3D AP_R40 gaps range from 4.5761 to 21.7163 AP.

## 3. Why three epochs?

### Recorded evidence

- `scripts/run_pointrcnn_full_train_stage.py:31` sets `--epochs` to `3` by default.
- `scripts/run_centerpoint_full_train.py:29` sets `--epochs` to `3` by default.
- All 13 executed `adaptation_protocol.json` files record `"epochs": 3`.
- PointRCNN passes `ckpt_save_interval = epochs` at lines 151–165.
- CenterPoint passes `ckpt_save_interval = epochs` and `num_epochs_to_eval = 0` at lines 106–124.
- The completed artifacts contain ten PointRCNN epoch-3 checkpoints and three CenterPoint epoch-3 checkpoints, with no epoch-1/2 checkpoints for these arms.

### Defensible explanation

Three epochs function as a fixed, short, compute-bounded adaptation schedule applied consistently across arms starting from pretrained detector checkpoints. However, the experiment record does not document an empirical study that selected three instead of six or ten epochs. Therefore, “compute-bounded short adaptation” is an operational interpretation of the implementation, not a proven historical reason.

### Convergence verdict

All 13 recorded training-loss series decreased from epoch 1 to epoch 3. This shows optimization progress. It does **not** establish validation convergence because:

- only epoch 3 was checkpointed;
- epoch 1/2 validation AP was not evaluated;
- no early-stopping or plateau rule was declared;
- training loss decreasing is not equivalent to validation AP reaching an optimum.

The supported statement is: “The three-epoch adaptation schedule completed and all recorded training-loss series decreased, but validation convergence and the optimality of epoch 3 were not demonstrated.”

Full convergence evidence: `../pugcn_detector_adaptation_full_val_20260908/reports/convergence_audit.json` and `convergence_audit.md`.

## 4. What 3N/3M means

For each frame:

1. The observed cloud contains `N` rows on Line A or `M=floor(N/4)` rows on Line B.
2. Merged PU-GCN patch predictions are converted into a strict `4N`/`4M` generated cloud.
3. Exactly `3N`/`3M` row indices are sampled uniformly without replacement from that strict generated cloud.
4. The detector input concatenates all observed rows first, then the selected generated rows, giving exactly `4N`/`4M` rows.

`3N` is not the “three children per input anchor.” It is a second frame-level sample from the strict PU-GCN output.

### Exact selection code

Source: `scripts/prepare_centerpoint_observed_first_train.py:14–21,52–75`.

```python
BASE_SEED = 20260718

def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF

expected = 4 * observed.shape[0]
if predicted.shape[0] != expected:
    raise ValueError(...)

seed = stable_seed(BASE_SEED, "e1", args.reference_token, frame)
rng = np.random.default_rng(seed)
selected = rng.choice(expected, size=3 * observed.shape[0], replace=False)
final = np.concatenate((observed, predicted[selected]), axis=0)
```

Line A uses `reference_token=original_x4_pu_gcn`; Line B uses `reference_token=downsampled_x4_pu_gcn`.

The selector does not use confidence, uncertainty, distance to observed points, FPS, curvature, voxel quotas, foreground labels, or coordinate deduplication. “Generated” is a provenance label, not a geometric set-difference definition.

## 5. 3N/3M audit evidence

Evidence files:

- `../pugcn_detector_adaptation_full_val_20260908/inputs/line_a_pugcn_observed_first_manifest.csv`
- `../pugcn_detector_adaptation_full_val_20260908/inputs/line_b_pugcn_observed_first_manifest.csv`

Both manifests contain 3,769 rows. Every row is `PASS`; every frame satisfies:

- `predicted_points = 4 × observed_points`;
- `selected_generated_points = 3 × observed_points`;
- `output_points = 4 × observed_points`;
- `observed_prefix_exact = True`;
- `generated_suffix_exact = True`.

Each row also stores `selected_indices_sha256` and `output_float32_sha256`.

Example, frame `000001`:

| Line | Observed | Predicted | Selected | Output | Seed | Status |
|---|---:|---:|---:|---:|---:|---|
| A | 120,268 | 481,072 | 360,804 | 481,072 | 1091044475 | PASS |
| B | 30,067 | 120,268 | 90,201 | 120,268 | 2918483960 | PASS |

## 6. PU-GCN and strict-4× principle

- `external/PU-GCN/Upsampling/model.py:251–258` centers and normalizes each patch, runs prediction, then transforms output back to metric coordinates.
- `external/PU-GCN/Upsampling/generator.py:228–266` extracts graph features, upsamples them, predicts coordinate residuals, tiles each input anchor four times, and adds residuals.
- A 2,048-row patch therefore produces 8,192 generated rows.
- `scripts/wrappers/strict_x4_from_merged_raw.py:118–130,194–220` uniformly samples merged raw rows without replacement to the exact frame target and assigns intensity from the nearest observed point.

Strict selection code:

```python
def choose_strict(raw_xyz, target_points, seed):
    rng = np.random.default_rng(seed)
    if raw_xyz.shape[0] == target_points:
        return raw_xyz.copy()
    if raw_xyz.shape[0] > target_points:
        idx = rng.choice(raw_xyz.shape[0], size=target_points, replace=False)
        return raw_xyz[idx].copy()
    raise ValueError("strict x4 output cannot be filled")
```

## 7. Byte-level source evidence

| Item | SHA-256 |
|---|---|
| `scripts/prepare_centerpoint_observed_first_train.py` | `f5e521219f8d49b8fed8d9d40192749378b02a0f54888b92e949d041db1f4f42` |
| `scripts/run_pointrcnn_full_train_stage.py` | `4b177d0dd27769a5f562ec45060e09537654965ab94936d9bd2df8f43b43828a` |
| `scripts/run_centerpoint_full_train.py` | `a71cbac6817271e53c19828cefbcdbc5ae1ad9d9c9bf5d4e665d16f211801ee6` |
| `reports/convergence_audit.json` | `3326d80c8916023936b9be7ad2b27f8469a8c9f32079fe8107ed60e8fadeacca` |
| `reports/full_val_detector_matrix.csv` | `68606b1062b93c8ae7ac1965f62e53221df2851eb533883e69ecb3632f3c63bd` |

The source snapshot package is under `../pugcn_detector_adaptation_full_val_20260908/reports/`:

- `pugcn_full_val_sources.tar.gz`
- `source_manifest.json`
- `experiment_source_full.md`

## 8. Reproduction entry point

```bash
cd /home/ra87racy/projects/baseline_detectors/PointRCNN
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh all
```
