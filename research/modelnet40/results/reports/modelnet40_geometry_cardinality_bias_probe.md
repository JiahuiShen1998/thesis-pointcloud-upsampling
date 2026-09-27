# Line A cardinality-bias probe — why CD against Original 1024 is not a surface metric

- Generated: 2026-08-13
- Split: test, N = 2468
- Script: `scripts/compute_cardinality_bias_probe.py`
- Raw: `reports/modelnet40_geometry_cardinality_bias_probe.json`

## Why this probe exists

Line A upsamples **Original 1024 → 4096**. The pre-equal-N geometry table scored that output
against **Original 1024**, which is also the method's own *input*. Chamfer Distance is
`CD = mean_a dist(a,B) + mean_b dist(b,A)`; under this pairing the two terms measure different
things:

- **forward** (4096 outputs → nearest Original point) grows precisely when a method places points
  *between* the input points, i.e. whenever it actually upsamples;
- **backward** (1024 Originals → nearest output) falls to zero whenever the input points are kept.

So the score rewards *not* upsampling. The `repeat_x4` control makes this explicit: it tiles the
Original cloud four times, performs no upsampling whatsoever, and attains the best obtainable
score under the old reference while ranking last under equal-N.

## Results

| Variant | CD vs Original 1024 | forward | backward | backward share | HD vs Original 1024 | CD vs mesh-ref 4096 | HD vs mesh-ref 4096 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repeat ×4 (control) | 0.000000 | 0.000000 | 0.000000 | — | 0.000000 | 0.055734 | 0.117819 |
| PDANS | 0.025907 | 0.023111 | 0.002796 | 10.8 % | 0.083822 | 0.046821 | 0.113510 |
| EAR | 0.029171 | 0.029171 | 0.000000 | 0.0 % | 0.098926 | 0.054722 | 0.115396 |
| PU-EdgeFormer | 0.030343 | 0.018357 | 0.011986 | 39.5 % | 0.045665 | 0.046666 | 0.110932 |
| PU-GCN | 0.034710 | 0.028211 | 0.006499 | 18.7 % | 0.101875 | 0.042306 | 0.093219 |
| PU-Net | 0.051957 | 0.031874 | 0.020084 | 38.7 % | 0.088135 | 0.049719 | 0.111177 |

## Reading

1. **`repeat_x4` scores CD = HD = 0.000000 under the old reference** — a perfect result for an
   output that contains no new information — and is *last of all six* under equal-N
   (CD 0.055734, HD 0.117819).
2. **EAR's backward term is exactly 0.00000000**: every Original point has an exact counterpart in
   its output, i.e. EAR retains the input verbatim and adds to it. This is why it placed 2nd under
   the old reference and last (5th) under equal-N.
3. The **forward term supplies 61–100 %** of every old CD score, so the old ranking is largely a
   ranking of how conservative each method was.
4. This probe **independently reproduces both stored tables to six decimals**
   (`modelnet40_geometry_vs_original_summary.json` and `modelnet40_geometry_equal_n_summary.csv`),
   so the discrepancy between them is attributable to the reference point set alone and not to the
   metric implementation.

## Corroborating control

Line B upsamplers were already evaluated at 1024 against a 1024-point reference. Their CD values
are **unchanged** between the two protocols (largest relative difference 2.2e-16; 2 of 5 rows
bit-identical). The numbers moved only where the cardinalities disagreed.

## Practice adopted

Before trusting a geometric benchmark, feed it a degenerate output that performs no work
(here: each input point repeated R times). If it scores well, the metric is measuring the wrong
quantity. Note that the pre-equal-N geometry metric could never have detected a repeat-×R fake
upsampler — it would have awarded it the top score; only the dedicated authenticity audit
(`pu_edgeformer_modelnet40_authenticity_verification_20260716.md`) could.
