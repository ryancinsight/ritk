# SimpleITK cmake-test coverage survey

Authoritative cross-reference of ritk's SimpleITK parity coverage against the
**full upstream filter set** — the 298 `Code/BasicFilters/yaml/*.yaml` definitions
that drive SimpleITK's generated cmake tests
(<https://github.com/SimpleITK/SimpleITK/tree/main/Code/BasicFilters/yaml>).
Regenerate with `tests/_gen_sitk_coverage.py`.

**288 / 298 covered** by a ritk differential parity test (10 not yet covered).
3 of those 10 are absent from this SimpleITK build (no oracle exists here), so the
**achievable ceiling in this environment is 295/298** — current coverage is
288/295 (97.6%) of what is validatable here.

## Covered (288)

Abs, AbsoluteValueDifference, Acos, AdaptiveHistogramEqualization, AdditiveGaussianNoise, Add, AggregateLabelMap, And, ApproximateSignedDistanceMap, AreaClosing, AreaOpening, Asin, Atan2, Atan, BSplineDecomposition, Bilateral, BinShrink, BinaryClosingByReconstruction, BinaryContour, BinaryDilate, BinaryErode, BinaryFillhole, BinaryGrindPeak, BinaryImageToLabelMap, BinaryMagnitude, BinaryMedian, BinaryMinMaxCurvatureFlow, BinaryMorphologicalClosing, BinaryMorphologicalOpening, BinaryNot, BinaryOpeningByReconstruction, BinaryProjection, BinaryPruning, BinaryReconstructionByDilation, BinaryReconstructionByErosion, BinaryThinning, BinaryThreshold, BinaryThresholdProjection, BinomialBlur, BitwiseNot, BlackTopHat, BoundedReciprocal, BoxMean, BoxSigma, CannyEdgeDetection, ChangeLabel, ChangeLabelLabelMap, CheckerBoard, Clamp, ClosingByReconstruction, CollidingFronts, ComplexToImaginary, ComplexToModulus, ComplexToPhase, ComplexToReal, Compose, ConfidenceConnected, ConnectedComponent, ConnectedThreshold, ConstantPad, Convolution, Cos, Crop, CurvatureAnisotropicDiffusion, CurvatureFlow, CyclicShift, DanielssonDistanceMap, DemonsRegistration, DiffeomorphicDemonsRegistration, Derivative, DICOMOrient, DilateObjectMorphology, DiscreteGaussianDerivative, DiscreteGaussian, DisplacementFieldJacobianDeterminant, DivideFloor, Divide, DivideReal, DoubleThreshold, EdgePotential, ErodeObjectMorphology, Equal, Exp, ExpNegative, Expand, FFTConvolution, FFTNormalizedCorrelation, FFTPad, FFTShift, FastApproximateRank, FastSymmetricForcesDemonsRegistration, FastMarching, FastMarchingBase, FastMarchingUpwindGradient, Flip, ForwardFFT, GaborImageSource, GaussianImageSource, GeodesicActiveContourLevelSet, GradientAnisotropicDiffusion, Gradient, GradientMagnitude, GradientMagnitudeRecursiveGaussian, GradientRecursiveGaussian, GrayscaleConnectedClosing, GrayscaleConnectedOpening, GrayscaleDilate, GrayscaleErode, GrayscaleFillhole, GrayscaleGeodesicDilate, GrayscaleGeodesicErode, GrayscaleGrindPeak, GrayscaleMorphologicalClosing, GrayscaleMorphologicalOpening, GreaterEqual, Greater, GridImageSource, HalfHermitianToRealInverseFFT, HConcave, HConvex, HMaxima, HMinima, HausdorffDistance, HistogramMatching, HuangThreshold, IntensityWindowing, IntermodesThreshold, InverseDeconvolution, InverseFFT, InvertDisplacementField, InvertIntensity, IsoContourDistance, IsoDataThreshold, IsolatedConnected, IterativeInverseDisplacementField, JoinSeries, KittlerIllingworthThreshold, LabelContour, LabelImageToLabelMap, LabelIntensityStatistics, LabelMapContourOverlay, LabelMapMask, LabelMapOverlay, LabelMapToBinary, LabelMapToLabel, LabelMapToRGB, LabelOverlapMeasures, LabelOverlay, LabelShapeStatistics, LabelStatistics, LabelSetDilate, LabelSetErode, LabelToRGB, LabelUniqueLabelMap, LabelVoting, LandweberDeconvolution, Laplacian, LaplacianRecursiveGaussian, LaplacianSegmentationLevelSet, LaplacianSharpening, LessEqual, Less, LiThreshold, Log10, Log, MagnitudeAndPhaseToComplex, Mask, MaskNegated, MaskedAssign, MaskedFFTNormalizedCorrelation, MaximumEntropyThreshold, Maximum, MaximumProjection, Mean, MeanProjection, Median, MedianProjection, MergeLabelMap, MinMaxCurvatureFlow, Minimum, MinimumMaximum, MinimumProjection, MirrorPad, Modulus, MomentsThreshold, MorphologicalGradient, MorphologicalWatershed, MorphologicalWatershedFromMarkers, MultiLabelSTAPLE, Multiply, N4BiasFieldCorrection, NaryAdd, NaryMaximum, NeighborhoodConnected, Noise, Normalize, NormalizedCorrelation, NormalizeToConstant, NotEqual, Not, ObjectnessMeasure, OpeningByReconstruction, Or, OtsuMultipleThresholds, OtsuThreshold, PermuteAxes, PhysicalPointImageSource, Pow, ProjectedLandweberDeconvolution, Rank, RealAndImaginaryToComplex, RealToHalfHermitianForwardFFT, ReconstructionByDilation, ReconstructionByErosion, RecursiveGaussian, RegionOfInterest, RegionalMaxima, RegionalMinima, ReinitializeLevelSet, RelabelComponent, RelabelLabelMap, RenyiEntropyThreshold, Resample, RescaleIntensity, RichardsonLucyDeconvolution, Round, STAPLE, SaltAndPepperNoise, ScalarConnectedComponent, ScalarImageKmeans, ScalarToRGBColormap, ShanbhagThreshold, ShapeDetectionLevelSet, ShiftScale, ShotNoise, Shrink, Sigmoid, SignedDanielssonDistanceMap, SignedMaurerDistanceMap, SimilarityIndex, SimpleContourExtractor, Sin, Slice, SmoothingRecursiveGaussian, SobelEdgeDetection, SpeckleNoise, Sqrt, Square, SquaredDifference, StandardDeviationProjection, Statistics, StochasticFractalDimension, Subtract, SumProjection, SymmetricForcesDemonsRegistration, Tan, TernaryAdd, TernaryMagnitude, TernaryMagnitudeSquared, Threshold, ThresholdMaximumConnectedComponents, ThresholdSegmentationLevelSet, TikhonovDeconvolution, Tile, Toboggan, TransformGeometry, TransformToDisplacementField, TriangleThreshold, UnaryMinus, UnsharpMask, ValuedRegionalMaxima, ValuedRegionalMinima, VectorConfidenceConnected, VectorConnectedComponent, VectorIndexSelectionCast, VectorMagnitude, VotingBinaryHoleFilling, VotingBinary, VotingBinaryIterativeHoleFilling, Warp, WhiteTopHat, WienerDeconvolution, WrapPad, Xor, YenThreshold, ZeroCrossing, ZeroCrossingBasedEdgeDetection, ZeroFluxNeumannPad

## Not yet covered (10)

These split into two genuinely distinct buckets.

**Absent from this SimpleITK build (3) — uncoverable here at any effort.** No
`sitk.<Name>` exists to diff against (verified via `hasattr(sitk, ...) == False`),
so the achievable ceiling in this environment is **295/298**, not 298:

- CoherenceEnhancingDiffusion (ritk *has* `coherence_enhancing_diffusion`, but
  sitk lacks the filter here), ContourExtractor2D, LevelSetMotionRegistration.

**In sitk but unimplemented in ritk / not bit-reproducible (7).** Each probed
with source + live experiments:

- Multi-class ITK level-set/iterative framework ports ritk does not implement:
  AntiAliasBinary + CannySegmentationLevelSet (SparseFieldLevelSet narrow-band),
  ScalarChanAndVeseDenseLevelSet (dense level set; ritk's `chan_vese` is a
  *different* algorithm — Dice 1.0 vs sitk's 0.0, does not correspond),
  PatchBasedDenoising (iterative patch search).
- InverseDisplacementField (ThinPlateSpline + `vnl_svd` — continuous SVD output,
  not bit-reproducible across linear-algebra libraries).
- IsolatedWatershed (binary-searches ITK's hierarchical `WatershedImageFilter`
  level; ritk's `watershed_segment` exposes no level parameter — demonstrated
  that substituting MorphologicalWatershed diverges completely).
- SLIC — **deterministic core now ported and validated** in
  `ritk-segmentation/src/clustering/slic/itk.rs` (`slic_itk_impl`): label-for-label
  exact vs `sitk.SLIC(enforceConnectivity=False, initializationPerturbation=False)`
  in 2-D and 3-D for super-grid sizes that evenly divide each axis. Remaining
  before counting as covered: (a) ITK shrink remainder convention for
  non-evenly-dividing super-grids; (b) default order-sensitive perturbation +
  connectivity-enforcement layers; (c) Python binding + cmake differential test.
