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
| IsolatedWatershed | gradient-watershed merge tree + binary search; sitk-exact | ✓ **FIXED** (commit cede4564) |
| PatchBasedDenoising | 25.1 max abs error | ✗ wrong |
| ScalarChanAndVeseDenseLevelSet | 0.19 segmentation match | ✗ wrong |
| AntiAliasBinary | sign FIXED (corr +0.90, 100% sign-agree); range ±1 vs sitk ±3 | ◐ sign correct, magnitude open |
| CannySegmentationLevelSet | 6.73 max abs error | ✗ wrong |
| CoherenceEnhancingDiffusion | — | no sitk oracle in this build |

Real validated-correct coverage is **~293/298** (IsolatedWatershed closed this
session, commit cede4564), not the generator's ~297.

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
Immersion is EXACT (level-0 partition matches fully across all images); only the
MERGE tie order for equal saliencies remains. Source-pinned: ITK's per-segment
saliency = `edge_list.front().height − segment.min` (lowest saddle − own min;
min over the two directions = `saddle − max(both mins)` = the formula above), and
the heap comparator `merge_comp` is `b.saliency < a.saliency` — **saliency-only,
NO secondary tie-break key** (`itkWatershedSegmentTree.h:85`). So equal-saliency
merges resolve by `std::make_heap`/`pop_heap` equal-element order on
segment-label-ordered insertion (`CompileMergeList` iterates in label order) —
deterministic but libstdc++-heap-specific.
REMAINING for full MorphologicalWatershed partition parity: replicate that heap
equal-element order. **For IsolatedWatershed specifically** the tie order rarely
matters — it only needs whether seed1/seed2 share a merged group at the
binary-searched level, robust to equal-saliency ties except at exact-tie seed
saliencies. So the port path: regional-minima + immersion + dynamic merge
(validated) + binary search over level until seeds separate + label seed1=1,
seed2=2, rest=0. The hard part (saliency) is solved and validated.

**End-to-end IsolatedWatershed status — VALIDATED 35/39 EXACT, ready to port.**
The prototype (gradient magnitude + this merge + binary search + 2-seed labels)
matches sitk.IsolatedWatershed EXACTLY on 35 of 39 random configs (sizes 8–15,
varied seeds/ranges); the 4 near-misses are 0.958–0.977 (equal-saliency
tie-breaking on a few boundary pixels). Two bugs fixed this session took it from
0.0–0.84 to exact: (a) saliency = `saddle − max(depth)` (above); (b) the
BINARY-SEARCH DIRECTION — at level `guess`, if seeds share a basin (MERGED) set
`hi=guess` (need a LOWER level = less merging); if SEPARATED set `lo=guess`;
output the watershed at the final `lo`. (My first attempt had these inverted,
passing only symmetric cases by coincidence.)
Exact validated port recipe (deterministic, no RNG):
1. `g = GradientMagnitude(input)` — central diff `(f[+1]−f[−1])/2` per axis with
   ZeroFluxNeumann (edge-clamp) boundary; **matches sitk.GradientMagnitude to 0.0**
   (np.gradient does NOT — it uses one-sided edges). Use ritk's validated filter.
2. Watershed merge on `g` at level `L`: plateau-aware regional minima → immersion
   flood → saddle = min boundary `max(va,vb)` → saliency = `saddle − max(min_a,min_b)`
   → dynamic merge while saliency ≤ L (L = level·(g.max−g.min)).
3. Binary search L∈[0,1] from guess=0.5: merged→hi=guess, separated→lo=guess,
   until `lo + isolatedValueTolerance ≥ guess`; relabel at `lo`.
4. Output: seed1's basin→replaceValue1(1), seed2's→replaceValue2(2), rest→0.
REMAINING for 39/39: equal-saliency tie order (std::make_heap on label-ordered
insertion) — affects ≤4% of pixels in ~10% of cases. The algorithm is otherwise
COMPLETE and validated; the next increment is the Rust port (reuse ritk's
gradient_magnitude + watershed scaffolding) + a differential test, then the same
basins+merge fixes ritk's `morphological_watershed` line divergence.

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

**SparseField algorithm extracted** (`itkSparseFieldLevelSetImageFilter.hxx`,
1109 lines) — the port spec (next-session work, same prototype→port method that
closed IsolatedWatershed): `m_ConstantGradientValue = 1.0` (or minSpacing). The
ACTIVE layer (layer 0) holds φ ∈ `[−0.5, +0.5]` (`±m_ConstantGradientValue/2`).
`UpdateActiveLayerValues`: `new = CalculateUpdateValue(old, dt, center, update)`;
if `new ≥ +0.5` the pixel moves UP one layer with `temp = new − 1.0`; if
`new < −0.5` moves DOWN with `temp = new + 1.0`; else stays. `CalculateUpdateValue`
is the AntiAlias fg/bg clamp (already implemented). `PropagateAllLayerValues`
reconstructs each outer layer L from the adjacent layer L∓1 by `inner ± 1.0`
(NOT a Euclidean distance — this is why a dense EDT-reinit prototype caps at
mean-err 0.18; the rigid ±1 propagation off the curvature-evolved active layer
is the missing mechanism). Far field = `±(max_layer + 1)·1 = ±3` (max_layer=2 ⇒
5 layers). So the faithful model: init φ = clamped signed distance (±3); each
iter evolve ONLY `|φ| < 0.5` pixels by mean curvature + the fg/bg constraint,
then rebuild layers ±1, ±2 by propagation (inner ± 1), not by re-distancing.
Prototype this active-layer-only + ±1-propagation scheme vs sitk.AntiAliasBinary
to 1e-2, then port (~500 lines, stateful linked-list layers). Canny reuses it
with an added edge speed term.
UPDATE (crude prototype tried, FAILED): a dense active-layer curvature evolution
(|φ|≤0.5) + a city-block ±1 propagation of |φ| outward gives mean-err **0.187**
vs sitk — WORSE than the static clamped signed distance (0.073). So the exact
`PropagateAllLayerValues` (which selects per-pixel min/max from the specific
adjacent-layer status sets, not a uniform city-block) genuinely matters; a dense
propagation shortcut does not validate. Unlike IsolatedWatershed (clean
algorithmic form, prototype-validatable), AntiAliasBinary requires the faithful
STATEFUL SparseField port (active/status layer linked lists + exact propagation)
directly — there is no dense prototype shortcut to 1e-2. This is the genuine
multi-session boundary: the port must replicate the layer bookkeeping, then
validate end-to-end vs sitk.AntiAliasBinary.
KEY INIT DETAIL (`InitializeActiveLayerValues`, sf.hxx:736): the active-layer
value is NOT a Euclidean signed distance — it is `shifted_value / length` clamped
to ±0.5, where `shifted_value` = input − isosurface (binary {0,1} − 0.5 = {∓0.5})
and `length = sqrt(Σ max(|dx_fwd|, |dx_bwd|)²) + MIN_NORM` (per-axis the LARGER of
the forward/backward difference of the shifted image). This first-order subvoxel
distance is the init that both my dense prototypes (EDT-reinit 0.18, city-block
0.187) lacked. Faithful prototype recipe: (1) shifted = binary − 0.5; (2) active
layer = pixels with a sign change to a neighbour in `shifted`; (3) active value =
shifted/length clamped ±0.5; (4) propagate ±1 for layers 1,2 (in/out); (5) each
iter: CalculateChange = curvature (CKS) on active layer, UpdateActiveLayerValues
(new = clamp-constrained old+dt·Δ; if |new|>0.5 the pixel changes layer, value
∓1), then re-propagate. Build this in numpy → match sitk.AntiAliasBinary to 1e-2
→ port. Still ~500-line stateful port, but the init+propagation are now fully
specified.
EXACT PROPAGATION (`PropagateLayerValues`, sf.hxx:937): done layer-by-layer
(status-tracked), NOT all-at-once over |φ|. For each pixel in the "to" layer,
scan its neighbours in the adjacent inner "from" layer and pick the one closest
to zero — OUTWARD (positive side): `value = min(from-neighbour values)`, then
`+ m_ConstantGradientValue (1)`; INWARD (negative): `value = max(from-neighbour
values)`, then `− 1`. Signed selection from the SPECIFIC inner layer, not an
absolute-value city-block over all neighbours (my crude prototype's error).
Order: layer 1 from layer 0, layer 2 from layer 1, etc. With this + the
subvoxel active init, the SparseField algorithm is now 100% specified for the
port — no further source reading needed; the remaining work is the faithful
stateful implementation + end-to-end validation vs sitk.AntiAliasBinary.
UPDATE (faithful prototype attempt, mean-err 0.137 — still not 1e-2): a numpy
prototype with the subvoxel init + signed layer-by-layer propagation + active-
layer curvature evolution reaches mean-err 0.137 (better than the crude 0.187
but worse than the static signed distance 0.073, with a −3.63 range overshoot
from unclamped inward propagation). So three prototype attempts (EDT 0.18,
city-block 0.187, faithful-layer 0.137) all miss 1e-2. Remaining gaps to nail:
(1) the exact AntiAlias curvature function + its `CalculateChange` dt (CFL), not
the generic CKS speed used here; (2) clamp layer values to ±3 during propagation;
(3) the exact active-layer reconstruction/ordering and RMS stop. Conclusion:
unlike IsolatedWatershed (clean algorithmic form, prototype-validatable in one
session), AntiAliasBinary's prototype-validate step is itself multi-iteration —
the SparseField is genuinely the harder multi-session port. The spec above is
the starting point; closing it needs careful iterative matching of the curvature
+ dt + reconstruction against sitk, then the Rust port.
KEY SUBTLETY FOUND (the 0.137→1e-2 gap): AntiAliasBinary's difference function is
`CurvatureFlowFunction` (speed = N/|∇φ|², already in ritk's curvature_flow.rs),
BUT `SparseFieldLevelSetImageFilter::CalculateChange` (sf.hxx:809) evaluates it on
a SHIFTED neighbourhood: per axis `offset[i] = offset[i]·centerValue /
(|∇φ|² + MIN_NORM)`, i.e. the stencil is shifted by the subvoxel zero-crossing
position `φ(x)·∇φ/|∇φ|²` before `ComputeUpdate`. So curvature is computed at the
interpolated zero crossing, NOT the raw grid point — my prototype's curvature on
raw φ is the remaining error source. COMPLETE algorithm now: CurvatureFlowFunction
speed on the shifted neighbourhood + the subvoxel active init + signed
layer-by-layer propagation + fg/bg constraint + CFL dt + RMS stop (0.07). The
reverse-engineering is done; the port is a careful faithful implementation of
this stateful pipeline validated end-to-end. This is the genuine multi-session
boundary — the SparseField is intricate (shifted-stencil curvature + layer
bookkeeping), distinct from IsolatedWatershed's clean prototype-validatable form.
