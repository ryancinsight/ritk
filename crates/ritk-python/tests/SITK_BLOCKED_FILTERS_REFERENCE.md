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
| AntiAliasBinary | sign FIXED (corr +0.90, 100% sign-agree); range ±1 vs sitk ±3 | ◐ sign correct, magnitude open |
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

**Source-verified merge spec** (`itkWatershedSegmentTreeGenerator.hxx`): the
`Level` maps to `threshold = m_FloodLevel · segTable->GetMaximumDepth()`; the
generator `PruneEdgeLists(threshold)` then merges the lowest-saliency edges first
through a `OneWayEquivalencyTable` (union-find), where each edge's saliency is the
boundary height relative to the adjacent basins' **depth** (computed by the
`Segmenter`'s segment table). So a faithful port is **two** ITK classes —
`Segmenter` (steepest-descent basins + segment table of per-basin depths and
inter-basin edge saliencies, ~600 lines) and `SegmentTreeGenerator` (the
FloodLevel·MaxDepth merge hierarchy, ~583 lines) — plus IsolatedWatershed's
binary search over `Level`. ~1000 lines across 2 classes; deterministic but
genuinely multi-session. My prototypes (descent, minimax, minimax-reachability,
catchment, immersion, full-immersion) all fail because none computes the exact
depth-relative edge saliency + FloodLevel·MaxDepth merge order.

**SALIENCY FORMULA CRACKED + VALIDATED** (this session, prototype vs
`sitk.MorphologicalWatershed(level)`): the merge algorithm that all 6 prior
prototype variants missed is now pinned and validated to **exact segment count at
all 10 levels (0.0–1.0) across 6 images** (sym7x7 + 5 random):
1. **Basins** = regional minima (plateau-aware: a connected equal-value region
   with no strictly-lower neighbour is one basin) + priority-queue immersion flood
   from those minima in increasing value order. Matches sitk at level 0 exactly.
2. **Saddle** between two basins = min over their shared boundary of
   `max(value_a, value_b)` (lowest connecting ridge height).
3. **Saliency** (the crux) = `saddle − max(depth_a, depth_b)` where depth = the
   basin's MINIMUM value. It is the SHALLOWER basin's persistence (shallower =
   higher minimum = `max` of the two minima), NOT `saddle − min`. Using `min`
   under-merges; `max` matches sitk's segment counts exactly.
4. **Merge** = dynamic/iterative: pop the lowest-saliency edge, union the two
   basins (merged depth = `min` of the two minima — inherits the deeper), then
   recompute the saliency of the merged basin's edges (lazy re-push on stale).
   Stop when the lowest remaining saliency `> level`.
5. **Level** is the absolute saliency threshold (sitk's `level` param directly;
   for the normalized form `m_FloodLevel·MaxDepth`).
REMAINING: only fine **partition tie-breaking** (same segment count, slightly
different basin grouping / watershed-line pixels) when edges share equal saliency
— needs ITK's exact immersion FIFO order + merge tie order. The hard part (the
saliency definition) is solved; this de-risks the port to: regional-minima +
immersion + this dynamic merge + tie-break + IsolatedWatershed binary search.

**No closed-form shortcut for the raw Segmenter** (`itkWatershedSegmenter.hxx`, 1315 lines):
`MaxDepth = maximum − minimum` (intensity range), but the per-edge saliency
emerges from the full flooding + `AnalyzeBoundaryFlow` + flat-region merge +
`UpdateSegmentTable` pipeline — it cannot be reduced to a boundary-min closed
form, so a faithful prototype requires reimplementing the Segmenter itself. Total
port ≈ 1900 lines across the two classes; no single-turn path exists.

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
Source-verified (`itkScalarChanAndVeseLevelSetFunction.hxx`,
`itkRegionBasedLevelSetFunction.hxx`, both in Modules/Nonunit/Review/include):
`ComputeInternalTerm` returns **raw** `(I−c1)²`, `ComputeExternalTerm` raw
`(I−c2)²` — **no feature normalization anywhere**; `c1 = Σ(I·H)/ΣH`,
`c2 = Σ(I·(1−H))/Σ(1−H)`. `ComputeGlobalTimeStep` sets `dt = m_DT/m_MaxCurvatureChange`
(**curvature change only**; `m_MaxGlobalChange` is tracked but never used),
`m_DT = 1/(2·D)`. Curvature `κ = N/|∇φ|³`, `δ = (1/π)·ε/(ε²+φ²)`, global term
`δ·[λ1(I−c1)² − λ2(I−c2)²]`; output hard-binary; RMS-stop at maximumRMSError.
**Two single-layer fix hypotheses empirically FALSIFIED** (this session):
(1) feature-normalization — sitk does NOT normalize (scale 1 vs 100 give
different segmentations, and saturate identically for scale ≥ 100, fg 476);
(2) per-step signed-distance reinit — a full prototype with EDT reinit across
{globsign ±1}×{reinit on/off} matches at best 0.285 (168 fg vs sitk 476), reinit
makes it *worse*. The stabilization of raw ~thousand-scale region energy under a
curvature-only dt comes from the `MultiphaseDenseFiniteDifferenceImageFilter`
driver's internal mechanics (SharedData incremental c1/c2, reinitSmoothingWeight,
the exact spacing-aware curvature, area/volume terms) acting together — genuinely
multi-LAYERED, not a single missing piece. Multi-session; a clean prototype caps
at ~0.16–0.285 and both single-layer shortcuts are now dead.

### AntiAliasBinary / CannySegmentationLevelSet — need SparseFieldLevelSet
Both are `itk::SparseFieldLevelSetImageFilter` narrow-band evolutions
(status-layer active-set updates, RMS convergence) — highly order-sensitive.

**AntiAliasBinary sign defect FIXED this session** (commit b17c2292):
source-verified from `itkAntiAliasBinaryImageFilter.hxx::CalculateUpdateValue`
— a foreground voxel (`== m_UpperBinaryValue`) is clamped to `max(new,0)` and
background to `min(new,0)`, so foreground carries the **positive** sign and the
zero iso-surface sits on the binary boundary. ritk initialised foreground to −1
and omitted the per-step constraint; corrected to fg=+1/bg=−1 plus the clamp.
Measured: corr −0.90 → **+0.9022, 100% per-voxel sign agreement**. The old
fg-negative sign is incompatible with the constraint (collapses to 0).

**Residual is magnitude only** and is genuinely SparseField-bound: ritk emits a
±1 mean-curvature level set; sitk stores a ±3 narrow-band **signed distance**
with subvoxel boundary precision (29 distinct values, e.g. ±1.509, ±1.697).
Best dense approximations measured this session: clamped signed-distance of the
original binary → corr 0.9947, **mean-err 0.073**; with curvature-flow evolution
→ 0.18 (over-smooths). An **exhaustive active-band curvature-flow sweep**
(band ∈ {1, 1.5, 2, ∞} × dt ∈ {0.05, 0.1, 0.125, 0.25}, 50 iters + EDT reinit +
constraint) caps at mean-err ≥ 0.09 / max-err ≥ 1.05 — every config *worse* than
the static distance, with max-err pinned ~1.264 by layer-quantization voxels.
The dense-approximation space is therefore **exhausted**: all ≫ the 1e-2
threshold. The per-step SparseField narrow-band corner-antialiasing + exact layer
value propagation must be ported; no sign fix, static distance map, or dense
curvature-flow variant suffices. Do not re-attempt the dense route.

CannySegmentationLevelSet adds a Canny edge speed term over the same SparseField.
Both multi-session; the SparseField active-set update order must match ITK exactly.
