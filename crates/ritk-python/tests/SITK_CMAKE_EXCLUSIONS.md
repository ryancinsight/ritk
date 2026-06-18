# SimpleITK cmake-coverage: investigated exclusions

Per-filter reasons that the remaining uncovered SimpleITK cmake filters are **not**
booked as ritk parity. Each was probed against sitk and found to have a genuine
algorithmic / convention / determinism difference — not a fixable composition.
Recorded so the investigation is not repeated. No approximate or partial-parameter
parity is booked as coverage (integrity: no fabricated parity).

## Algorithmic differences (no clean ritk composition)

- **ErodeObjectMorphology** — not `grayscale_erosion`. On a clean (margined) box it
  coincidentally equals `grayscale_erosion(2·r)`, but that breaks on an L/disk shape
  at r=2,3 (maxdiff 1) — a genuinely different boundary-erosion rule. (DilateObject
  *is* covered via `grayscale_dilation(r)`; the erode dual is not the mirror.)
- **ZeroCrossingBasedEdgeDetection** — `discrete_gaussian → laplacian → zero_crossing`
  diverges because ritk `laplacian` and ITK `Laplacian` differ by the spacing² scaling
  (615 maxdiff on cthead, spacing 0.353), flipping thousands of near-zero crossings.
  `zero_crossing` itself is bit-exact (covered as `ZeroCrossing`).
- **DiscreteGaussianDerivative** — ITK `GaussianDerivativeOperator` is the discrete
  (Bessel) Gaussian derivative; ~2.6 % off the sampled continuous form and not equal to
  `discrete_gaussian ⊛ central-diff` (best simple-difference match 2.8 % off). Generating
  formula not recoverable without the ITK source.
- **BinaryPruning** — non-trivial asymmetric 3×3-template spur removal (Lam/Lee/Suen),
  not simple endpoint deletion.
- **SignedDanielssonDistanceMap** — Danielsson vector propagation; ritk `distance_transform`
  is Maurer-style (agrees with `SignedMaurer` only to MAE < 0.15, not bit-exact).
- **LabelSetDilate / LabelSetErode** — structuring-element shape differs from ritk
  `label_dilation/erosion` (box); matches only coincidentally at isolated radii.
- **MorphologicalWatershed** — different basin count (ritk 12167 vs sitk 7458 on the
  cthead gradient); `watershed_segment` has no `level`/`markWatershedLine` controls.
- **LabelMapContourOverlay** — contour-overlay rendering differs even at dilationRadius 0
  (0.8 % of pixels).
- **RelabelLabelMap** — different size-tie ordering than `relabel_components` (label-2
  size 12 vs 19).

## Parameter / config gated

- **ThresholdMaximumConnectedComponents** — default `MinimumObjectSizeInPixels = 0` is
  degenerate (maximising component count → isolated pixels); ITK's bisection search would
  need an exact reproduction to match a non-default config.
- **FastApproximateRank** — covered at the default rank 0.5 (median); non-default ranks use
  ITK's per-axis-adjusted approximation that ritk does not replicate.

## Non-deterministic (RNG)

- **AdditiveGaussianNoise, SaltAndPepperNoise, ShotNoise, SpeckleNoise, Noise** — ITK uses
  its `MersenneTwisterRandomVariateGenerator`; ritk's noise RNG differs, so seeded output is
  not reproducible. Would require reimplementing ITK's MT19937 + normal-variate sequence
  exactly.

## Geometry-blocked (pending core reconciliation)

- **Warp, TransformToDisplacementField, InvertDisplacementField, InverseDisplacementField,
  IterativeInverseDisplacementField, TransformGeometry, DICOMOrient** — depend on the
  `make_image` vs `ritk.io` geometry axis-order divergence (memory
  `ritk-displacement-field-axis-bug`, task to reconcile in `ritk-image`). `Warp` ships but is
  wrong on loaded anisotropic data; the rest are deferred until the core is fixed.

## Needs substantial new core (iterative / level-set / template)

AdaptiveHistogramEqualization (no binding), AntiAliasBinary, ApproximateSignedDistanceMap,
BSplineDecomposition (no binding), Binary/MinMaxCurvatureFlow, CannySegmentationLevelSet,
CollidingFronts, ContourExtractor2D, *DemonsRegistration, FastMarching*, IsoContourDistance,
Isolated{Connected,Watershed}, LaplacianSharpening (nonlinear), LevelSetMotionRegistration,
MaskedFFTNormalizedCorrelation (no mask support), MergeLabelMap (label packing),
MultiLabelSTAPLE (EM), NormalizedCorrelation (template), PatchBasedDenoising,
{Real→HalfHermitian, HalfHermitian→Real}FFT (half-spectrum layout), ReinitializeLevelSet,
SLIC, ScalarChanAndVeseDenseLevelSet, StochasticFractalDimension, Toboggan,
Vector{ConfidenceConnected,ConnectedComponent}, BitwiseNot (integer-width semantics).
