# SimpleITK cmake-coverage: investigated exclusions

Per-filter reasons the **30 still-uncovered** SimpleITK cmake filters are not booked
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
- **CollidingFronts** is now shipped float-exact (`filter.colliding_fronts`): two `fast_marching`
  passes (from `seeds1`, from `seeds2`), the upwind gradient of each arrival-time field
  (`itk::FastMarchingUpwindGradient`'s per-axis one-sided difference), their dot product `∇T1·∇T2`,
  seeds pinned to `m_NegativeEpsilon = −1e-6`, and (with `applyConnectivity`) a face-connected
  flood from `seeds1` over `{P ≤ −1e-6}`. Verified ≤1e-3 (f32 arrival-time rounding) vs
  `sitk.CollidingFronts` for both connectivity modes.

## Watershed (RESOLVED, Sprint 489 — MorphologicalWatershed now covered)

- **MorphologicalWatershed** is now shipped bit-exact via the composition
  `MorphologicalWatershedFromMarkers(f, ConnectedComponent(RegionalMinima(HMinima(f, level))))`,
  after fixing ritk's `MarkerControlledWatershed` to match ITK exactly (collision
  non-propagation + hierarchical-FIFO flooding; the divergence was ~5.5 % of watershed-line
  voxels on complex reliefs, now 0).
- **IsolatedWatershed** — investigated to source and reclassified as **not bit-exact reproducible**:
  `itk::IsolatedWatershedImageFilter` binary-searches the `Level` of `itk::WatershedImageFilter`
  (Vincent–Soille hierarchical *segmentation* — a saliency merge-tree over flooded basins, NOT ritk's
  marker/Meyer flooding) until two seeds separate, then relabels. The basin labeling and merge order of
  that watershed are implementation-specific (order-sensitive, like Toboggan), so a different watershed
  engine cannot reproduce the seed labels — and the binary search amplifies any divergence. Needs an
  exact ITK WatershedImageFilter port, which is not bit-reproducible.

## Label-map / vector-image types ritk lacks

- **LabelMapContourOverlay, LabelSetDilate, LabelSetErode, MergeLabelMap, RelabelLabelMap** — ITK
  LabelMap (run-length object) algebra; ritk has only dense label images.
  **VectorConfidenceConnected, VectorConnectedComponent** — operate on vector-pixel images, which
  ritk's scalar-f32 backend does not represent.
- **MultiLabelSTAPLE** is now shipped float-exact (`segmentation.multi_label_staple`): it operates on
  dense label maps (not LabelMap objects), so it was reachable — LabelVoting-seeded confusion-matrix
  EM (column-stochastic) emitting the per-voxel argmax consensus (ties → undecided = max label + 1).
  The discrete label output is robust to weight ULP-jitter; verified `array_equal` to
  `sitk.MultiLabelSTAPLE` across 3 noisy 4-rater cases.

## Template / masked correlation

- **NormalizedCorrelation** is now shipped float-exact (`filter.normalized_correlation`): the template
  is normalized to mean-zero/unit-norm (`k = std·√(N−1)`), then per masked voxel `out = Σ I·nt /
  √(ΣI² − (ΣI)²/N)` over the ZeroFluxNeumann neighbourhood. The earlier non-convergence was for lack of
  ITK's exact functor (the `k` scaling and the local-energy denominator); verified ≤1e-4 vs sitk
  (full + partial mask). The mask must be cast to float32 for `sitk.NormalizedCorrelation`.
- **MaskedFFTNormalizedCorrelation** — the single remaining *genuinely reachable* filter (deterministic),
  but it needs a new multi-FFT subsystem ritk lacks. Full algorithm understood from source
  (`itk::MaskedFFTNormalizedCorrelationImageFilter`, Padfield 2012): rotate the moving image+mask 180°;
  via ~10 FFTs at an FFTPad good-size (only 2/3/5 prime factors ≥ N₁+N₂−1) compute
  `numberOfOverlapPixels = round(IFFT(M̂f·M̂t))` (positive-clamped), `numerator = IFFT(F̂·T̂) −
  IFFT(F̂·M̂t)·IFFT(M̂f·T̂)/overlap`, fixed/moving denominator parts from `IFFT(F̂²·M̂t)` and
  `IFFT(M̂f·T̂²)` minus their mask-correlation² /overlap, `denominator = √(fixedPart·movingPart)`,
  `NCC = numerator/denominator` post-processed (0 where `denominator < precisionTolerance` or
  `overlap < requiredOverlap`; clamp to [−1, 1]). The "full" output layout (size N₁+N₂−1) and the
  CalculatePrecisionTolerance gating must match exactly. Scoped as a dedicated multi-turn port — not
  rushed, since a wrong FFTPad size or tolerance would defeat bit-exactness.

## Approximate by design (not bit-exact)

- **ApproximateSignedDistanceMap** is now shipped float-exact (`filter.approximate_signed_distance_map`):
  composed exactly as ITK's mini-pipeline — `IsoContourDistance(level=(in+out)/2, far=⌊√Σsizeᵢ²⌋+1)`
  (ritk's iso matches ITK bit-exact) then a `FastChamferDistance` two-pass sweep with weights
  `[0.92644,1.34065,1.65849]`, on the negated iso (ritk iso is inside-positive, ITK ASDM inside-negative;
  chamfer is antisymmetric). The earlier "3.4 % off" was comparing ritk's *exact EDT* — the fix was to
  port ITK's *approximate* algorithm. Verified ≤1e-4 vs sitk.
- **SLIC** — iterative k-means superpixels with gradient-perturbed seeds; ritk has
  `SlicSuperpixelFilter` but two independent SLIC impls cannot match label-for-label.

## Binding-surface / representation blocked

- **DICOMOrient** is now shipped float-exact (`filter.dicom_orient`): a signed axis permutation
  (no resampling) computed from the input direction cosines to the target orientation code,
  transforming data + spacing + origin + direction together (`OrientImageFilter`). `PyImage.direction`
  now exposes the cosine matrix in SimpleITK's `(x,y,z)` row-major layout (`D_sitk[i][j] =
  D_core[(i, 2-j)]`). Verified float-exact (array ≤1e-5, geometry ≤1e-9) across 8 orientation codes.
- **TransformGeometry** is now shipped float-exact (`filter.transform_geometry`): applies an affine
  transform to the image's origin + direction (pixels/spacing unchanged) via ITK's inverse linear
  map `origin' = A⁻¹·(origin − c − t) + c`, `direction'.col = A⁻¹·direction.col`. Verified vs
  `sitk.TransformGeometry` across translation, rotation, and non-identity-direction cases.
- **InvertDisplacementField** is now shipped float-exact (`filter.invert_displacement_field`): ITK's
  Chen et al. fixed-point scheme over the 3 world components (`v ← v + ε·clamp(−(v + u∘(id+v)))`,
  ε = 0.75 then 0.5, scaled-norm clamp, boundary pinned). The key to bit-exactness was matching ITK's
  vector-linear `IsInsideBuffer` edge semantics (continuous index outside `[−0.5, size−0.5]` → zero
  vector; upper neighbour clamped). Internal f64; verified ≤1e-4 vs sitk on a smooth aniso field.
- **IterativeInverseDisplacementField** is now shipped float-exact
  (`filter.iterative_inverse_displacement_field`): a negated-field-warped first guess refined by a
  per-voxel coordinate-descent line search (step-halving, ±step along each physical axis), reusing the
  shared ITK-faithful vector interpolation. Verified ≤1e-4 vs sitk (prototype matched to 2e-15 in f64).
- **InverseDisplacementField** — investigated to source and reclassified as **not bit-exact
  reproducible**: ITK's `InverseDisplacementFieldImageFilter` subsamples the field into landmark pairs
  and fits a `ThinPlateSplineKernelTransform`, whose `ComputeWMatrix` solves the dense landmark
  L-matrix with `vnl_svd` (threshold 1e-8). The result is a continuous field determined by VNL's
  specific SVD rounding over a moderately ill-conditioned matrix; no independent linear-algebra
  implementation reproduces that bit-for-bit, so float-exact parity is unattainable (same class as the
  iterative PDE solvers below). The other two field inversions are covered — they are deterministic
  interpolation + arithmetic, not a dense solve.
- **BitwiseNot** — bitwise NOT depends on the integer pixel width (uint8 vs int16 …); ritk's
  scalar-f32 backend carries no bit-width, so the result is undefined.

## No usable sitk oracle in this build

- **AdaptiveHistogramEqualization** is now shipped float-exact (`filter.adaptive_histogram_equalization`):
  Stark's α/β contextual method implemented directly as ITK's deterministic windowed cumulative sum
  (`cumf(u,v) = 0.5·sgn(u−v)·|2(u−v)|^α − 0.5·β·sgn(u−v)·|2(u−v)| + β·u`, box window shrinking at the
  border). Earlier set aside because ritk's *global*-HE/CLAHE differ — the faithful fix was to port the
  exact ITK functor. Verified ≤1e-3 vs sitk across (α,β) = (0.3,0.3),(0,0),(1,0.5).
- **CoherenceEnhancingDiffusion** — ritk has `CoherenceEnhancingDiffusionFilter`, but this
  SimpleITK build does not expose `sitk.CoherenceEnhancingDiffusion`, so there is no oracle.
- **ContourExtractor2D, Toboggan** — output polylines / an over-segmentation that this build
  either does not expose as a comparable image or whose labeling is implementation-defined.
