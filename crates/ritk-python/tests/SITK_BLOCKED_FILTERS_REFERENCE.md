# Blocked-filter sitk parity reference

Validated state of the SimpleITK filters that are **bound but do not yet match
sitk** under a real value-semantic differential. The coverage generator
(`_gen_sitk_coverage.py`) counts a filter "covered" when `sitk.<Name>` appears in
any `test_*.py`, so weak existence-only structural tests inflate the count.
**This file is the authoritative record of which "covered" filters actually
match sitk**, plus the correct algorithm for each that does not, so the real
ports can be done against a known reference rather than re-derived.

Measured by `ritk.<binding>` vs `sitk.<Filter>` on the inputs below
(single-threaded, `sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)`).

## Validation table (concurrent-agent bindings)

| Filter | ritk vs sitk | Verdict |
|--------|--------------|---------|
| ContourExtractor2D | iso-contour vertices set-equal (square 24/24, two-blobs 26/26) | ✓ **correct** |
| IsolatedWatershed | 0.0 label match | ✗ wrong algorithm |
| PatchBasedDenoising | 25.1 max abs error | ✗ wrong |
| ScalarChanAndVeseDenseLevelSet | 0.19 segmentation match | ✗ wrong |
| AntiAliasBinary | corr −0.90, range ±1 vs sitk ±3 | ✗ wrong (no smoothing) |
| CannySegmentationLevelSet | 6.73 max abs error | ✗ wrong |
| CoherenceEnhancingDiffusion | — | no sitk oracle in this build |

Real validated-correct coverage is **~292/298**, not the generator's ~297.

## Correct algorithm per filter (from prototype-vs-sitk derivation)

### ContourExtractor2D — CORRECT, just needs a vertex-exact differential test
Marching squares: for each adjacent-pixel edge crossing the iso-value, the
vertex is `p0 + t·(p1−p0)` with `t = (cv − v0)/(v1 − v0)`. Horizontal edge
`(y,x)-(y,x+1)` → vertex `(x+t, y)`; vertical → `(x, y+t)`. The agent's
`contour_extractor_2d.rs` produces these exactly; assert the vertex **set** equals
`sitk.ContourExtractor2DImageFilter`'s `GetContour` (note: contours come from
`GetNumberOfCountours()` [ITK typo] + `GetContour(i)` after `Execute`, not from
the `Execute` return which is `None`).

### IsolatedWatershed — needs ITK hierarchical `WatershedImageFilter`
sitk output = each seed's watershed **catchment basin** (steepest-descent on the
gradient), labels `{0=other, 1=seed1, 2=seed2}`. The agent's threshold-
connectivity (`lowest T* s.t. seeds split in {I≤T*}`, labels 1/2/3) is a
different algorithm → 0.0 match. The isolated level = the minimax barrier between
seeds (a Dijkstra, validated exact, B=2.0 on the 7×7 relief), but the **regions
are catchment basins, not threshold/minimax sets** (proven: a basin pixel can
have gradient > B). Requires porting `itk::watershed::Segmenter` (descent flood +
boundary-saliency merge tree) — multi-session.

### PatchBasedDenoising — needs the seeded RNG sampler
Scalar default-config update (numberOfIterations=1, noiseModelFidelityWeight=0 ⇒
no noise term, KernelBandwidthEstimationOff ⇒ fixed σ): rescale intensities to
`[0,100]` (`v=(I−min)·100/(max−min)`); cubic-smoothstep patch weights
(`UseSmoothDiscPatchWeights=true`); per candidate `g=exp(−Σ wt·(v_p−v_j)²/(2σ²))`,
σ=400; `gradient=Σ g·(v_j[center]−v_p[center]) / (Σ g + minProb)`;
`out = v_p + 0.2·gradient` (stepSizeSmoothing=0.2, smoothingWeight=1.0); rescale
back. A full-window NLM prototype reaches **1.7%** but cannot match bit-exact: an
impulse test shows sitk's output is **asymmetric while deterministic** — the
`GaussianRandomSpatialNeighborSubsampler` (radius 25, draws-with-replacement)
weights candidates by seeded per-position frequencies. Requires porting that
sampler bit-for-bit (MT19937 already exists in `noise/mersenne.rs`).

### ScalarChanAndVeseDenseLevelSet — needs the dense multiphase solver
Curvature term `κ·δ` with `κ = N/|∇φ|³` (mean curvature numerator), `δ =
(1/π)·ε/(ε²+φ²)`; region term `δ·[λ1(I−c1)² − λ2(I−c2)²]`; `c1,c2` =
Heaviside-weighted inside/outside means recomputed per iteration; adaptive global
`dt = m_DT/max|curvature_term|`, `m_DT = 1/(2·D)`; output is hard-binary; RMS-stop
at maximumRMSError. A clean prototype reaches only **0.16** — the SharedData mean
weighting, adaptive dt, Heaviside sign, and RMS iteration count are entangled and
must all be bit-exact (the boundary is hyper-sensitive). Multi-session.

### AntiAliasBinary / CannySegmentationLevelSet — need SparseFieldLevelSet
Both are `itk::SparseFieldLevelSetImageFilter` narrow-band evolutions
(1000 iterations, status-layer active-set updates, RMS convergence) — highly
order-sensitive. AntiAliasBinary smooths a binary image into a level set
(range ±3, sitk); the agent's output is ±1 and inverted (no curvature-flow
evolution). CannySegmentationLevelSet adds a Canny edge speed term. Both
multi-session; the SparseField active-set update order must match ITK exactly.
