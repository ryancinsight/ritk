# SimpleITK cmake-coverage: investigated exclusions

Per-filter reasons the **43 still-uncovered** SimpleITK cmake filters are not booked
as ritk parity. Each was probed against sitk and found to have a genuine
algorithmic / determinism / type-system difference, or a binding-surface blocker —
not a fixable bit-exact composition. No approximate or partial-parameter parity is
booked as coverage (integrity: no fabricated parity).

This list is kept in sync with `SITK_CMAKE_COVERAGE.md` "Not yet covered". Filters
that were once here and have since shipped bit/float-exact — DiscreteGaussianDerivative,
StochasticFractalDimension, LaplacianSharpening, ZeroCrossingBasedEdgeDetection,
SignedDanielssonDistanceMap, Noise (local estimator), ErodeObjectMorphology,
RealToHalfHermitianForwardFFT, HalfHermitianToRealInverseFFT, TransformToDisplacementField,
BinaryThinning, BinaryPruning, ThresholdMaximumConnectedComponents, IsoContourDistance,
IsolatedConnected — have been removed. The `Warp` geometry divergence is **resolved**
(rewritten on the canonical tensor path, float-exact on loaded anisotropic).

## Non-deterministic (RNG generators)

- **AdditiveGaussianNoise, SaltAndPepperNoise, ShotNoise, SpeckleNoise** — ITK's
  `MersenneTwisterRandomVariateGenerator` with a cached second normal variate, a specific
  open-range uniform divisor, and per-region threaded seeding. A faithful MT19937 + polar
  Box-Muller does not reproduce `sitk.AdditiveGaussianNoise(seed=42)` even on an 8-px image.
  (The deterministic local-noise *estimator* `Noise` is covered.)

## Iterative / non-bit-exact convergence

- **AntiAliasBinary, BinaryMinMaxCurvatureFlow, MinMaxCurvatureFlow, CannySegmentationLevelSet,
  ScalarChanAndVeseDenseLevelSet, ReinitializeLevelSet, LevelSetMotionRegistration,
  PatchBasedDenoising** — iterative PDE / level-set solvers whose per-step floating-point
  accumulation compounds; not bit-exact across independent implementations.
- **DiffeomorphicDemonsRegistration, FastSymmetricForcesDemonsRegistration,
  SymmetricForcesDemonsRegistration** — iterative deformable registration; non-reproducible.

## Fast-marching family

- **FastMarching** is now shipped float-exact (`filter.fast_marching`): the upwind quadratic
  Eikonal solve + min-heap gives the unique arrival-time field, so heap tie-ordering is
  irrelevant. Verified ≤5e-7 vs sitk on 2-D/3-D and varying speed.
- **FastMarchingBase, FastMarchingUpwindGradient** are also covered: their arrival-time output
  equals `FastMarching`'s (the upwind-gradient secondary output is not the primary image), so
  `filter.fast_marching` matches both float-exactly (≤1e-6).
- **CollidingFronts** — runs two `FastMarchingUpwindGradient` passes (seeds1→targets2 and
  seeds2→targets1) with `GenerateGradientImageOn`, then outputs `−(∇T1·∇T2)` (the dot product of
  the two **upwind** gradient fields, `m_NegativeEpsilon = −1e-6`). Needs the upwind-gradient
  output added to `FastMarchingFilter` (ITK computes ∇T at each node as it goes alive, not by
  finite differences), then the dot product. Reachable, deferred.

## Watershed (RESOLVED, Sprint 489 — MorphologicalWatershed now covered)

- **MorphologicalWatershed** is now shipped bit-exact via the composition
  `MorphologicalWatershedFromMarkers(f, ConnectedComponent(RegionalMinima(HMinima(f, level))))`,
  after fixing ritk's `MarkerControlledWatershed` to match ITK exactly (collision
  non-propagation + hierarchical-FIFO flooding; the divergence was ~5.5 % of watershed-line
  voxels on complex reliefs, now 0).
- **IsolatedWatershed** remains uncovered: it is the *isolated*-watershed variant (binary-search
  the flooding level that separates two seeds), not the marker-less filter — needs its own
  bisection over the marker-watershed, deferred.

## Label-map / vector-image types ritk lacks

- **LabelMapContourOverlay, LabelSetDilate, LabelSetErode, MergeLabelMap, RelabelLabelMap,
  MultiLabelSTAPLE** — ITK LabelMap (run-length object) algebra; ritk has only dense label
  images. **VectorConfidenceConnected, VectorConnectedComponent** — operate on vector-pixel
  images, which ritk's scalar-f32 backend does not represent.

## Template / masked correlation

- **NormalizedCorrelation** — spatial template matching: the template is a NeighborhoodOperator
  and a mask gates pixels; the per-pixel local-norm normalization needs faithful replication.
- **MaskedFFTNormalizedCorrelation** — Padfield masked NCC via multiple FFTs (no mask support
  in ritk's FFT NCC yet).

## Approximate by design (not bit-exact)

- **ApproximateSignedDistanceMap** — chamfer / isocontour approximation; 3.4 % off ritk's exact
  signed EDT (a different algorithm, not a tolerance gap).
- **SLIC** — iterative k-means superpixels with gradient-perturbed seeds; ritk has
  `SlicSuperpixelFilter` but two independent SLIC impls cannot match label-for-label.

## Binding-surface / representation blocked

- **TransformGeometry, DICOMOrient** — both mutate the image Direction matrix, but ritk's Python
  `Image` exposes no direction setter/getter (constructor takes `array/spacing/origin` only), so
  the output geometry cannot be represented or validated through the binding.
- **InverseDisplacementField, InvertDisplacementField, IterativeInverseDisplacementField** —
  invert a dense displacement field via scattered-data / fixed-point iteration over a vector
  field; needs vector-field warping (per-component) plus iteration, deferred.
- **BitwiseNot** — bitwise NOT depends on the integer pixel width (uint8 vs int16 …); ritk's
  scalar-f32 backend carries no bit-width, so the result is undefined.

## No usable sitk oracle in this build

- **AdaptiveHistogramEqualization** — ITK's α/β contextual method (Stark); ritk has global-HE
  and CLAHE, neither equal to it.
- **CoherenceEnhancingDiffusion** — ritk has `CoherenceEnhancingDiffusionFilter`, but this
  SimpleITK build does not expose `sitk.CoherenceEnhancingDiffusion`, so there is no oracle.
- **ContourExtractor2D, Toboggan** — output polylines / an over-segmentation that this build
  either does not expose as a comparable image or whose labeling is implementation-defined.
