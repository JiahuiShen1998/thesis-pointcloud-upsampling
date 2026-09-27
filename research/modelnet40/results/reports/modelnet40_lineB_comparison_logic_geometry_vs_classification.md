# ModelNet40 Line B Comparison Logic (Geometry vs Classification)

- Updated: `2026-07-16`
- Scope: Line B (Downsampled ×4 + Upsampling). Applies to all methods including PU-EdgeFormer.

## Do not mix these comparisons

| Question | Modality | Primary reference | Formula |
|---|---|---|---|
| Did upsampling recover **geometry** toward Original quality? | Geometry | **Original baseline 1024** | `delta_*_vs_original = metric(method) − metric(Original baseline)` |
| Did upsampling improve **downstream classification** over the degraded input? | Classification | **Downsampled ×4 baseline 256** | `delta_accuracy_vs_downsampled_baseline = Acc(method) − Acc(Downsampled baseline)` |
| How far is classification still from full Original performance? | Classification (secondary) | **Original baseline 1024** | `gap_accuracy_vs_original_baseline = Acc(method) − Acc(Original baseline)` |

## Geometry (absolute + delta)

1. **Absolute metrics** (CD / HD / NUC / exact P2F):
   - computed vs **dense mesh surface reference** / mesh reference / GT reference
   - never call the mesh reference a “baseline”

2. **Geometry deltas** (Line B):
   - always vs **Original baseline 1024**
   - example: `delta_CD_vs_original = CD(Downsampled x4 + method) − CD(Original baseline)`
   - meaning: does the upsampled cloud match Original’s geometric quality relative to the mesh?

3. Geometry delta figures:
   - Original baseline is **y=0 only** (not a method bar)

## Classification (primary + secondary)

1. **Primary Line B classification delta** (must be used in main Line B accuracy tables/figures):
   - vs **Downsampled ×4 baseline 256**
   - `delta_accuracy_vs_downsampled_baseline = Acc(LineB method) − Acc(Downsampled baseline)`
   - answers: “better than the downsampled input?”

2. **Secondary gap column** (required in PointNet++ final report; not the primary ranking delta):
   - vs **Original baseline 1024**
   - `gap_accuracy_vs_original_baseline = Acc(LineB method) − Acc(Original baseline)`
   - answers: “recovered to Original level?”

3. Line A classification remains vs **Original baseline 1024** (its line baseline).

## Explicit non-rules

- Do **not** use Downsampled baseline as the geometry delta zero for Line B.
- Do **not** use Original baseline as the *primary* Line B classification delta.
- Do **not** plot Original baseline as a geometry-delta method bar.
- Do **not** invent PU-EdgeFormer classification numbers while PointNet++ is pending.

## PU-EdgeFormer status

- Geometry deltas vs Original: **available**
- Classification primary Δ vs Downsampled baseline: **pending**
- Classification secondary gap vs Original baseline: **pending**
- `POINTNET_CLASSIFIER_STARTED=NO`
