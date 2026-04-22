# RITK Gap Audit — ITK / SimpleITK / ANTs / Grassroots DICOM Comparison

**Audit Date:** 2025-07-14 (updated Sprint 8, 2025-07-18; roadmap refresh 2026-04-20)**
**Auditor:** Ryan Clanton (@ryancinsight)
**Codebase Revision:** Confirmed via direct file inspection of `crates/ritk-{core,registration,io,model,python,cli}`
**Status:** Active — feeds `backlog.md` and `checklist.md`

## Update Note

**Sprint 24 (2026-04-20):** Next-stage roadmap refreshed to prioritize DICOM object-model preservation, VTK data-model expansion, ITK/SimpleITK breadth, ITK-SNAP workflow primitives, ANTs workflow refinement, and Python parity benchmarking. Existing image-series DICOM I/O, VTK legacy image I/O, registration, and Python bindings remain as previously recorded.

`Analyze` format support is present in `crates/ritk-io/src/format/analyze/` and should be treated as implemented. This audit now focuses on the remaining imaging gaps relative to DICOM, ITK, SimpleITK, VTK, ITK-SNAP, and ANTs.

**Audit Date:** 2025-07-14 (updated Sprint 8, 2025-07-18; roadmap refresh 2026-04-20)**
**Auditor:** Ryan Clanton (@ryancinsight)
**Codebase Revision:** Confirmed via direct file inspection of `crates/ritk-{core,registration,io,model,python,cli}`
**Status:** Active — feeds `backlog.md` and `checklist.md`

## Update Note

**Sprint 24 (2026-04-20):** Next-stage roadmap refreshed to prioritize DICOM object-model preservation, VTK data-model expansion, ITK/SimpleITK breadth, ITK-SNAP workflow primitives, ANTs workflow refinement, and Python parity benchmarking. Existing image-series DICOM I/O, VTK legacy image I/O, registration, and Python bindings remain as previously recorded.

`Analyze` format support is present in `crates/ritk-io/src/format/analyze/` and should be treated as implemented. This audit now focuses on the remaining imaging gaps relative to DICOM, ITK, SimpleITK, VTK, ITK-SNAP, and ANTs.

**Audit Date:** 2025-07-14 (updated Sprint 8, 2025-07-18)**
**Auditor:** Ryan Clanton (@ryancinsight)
**Codebase Revision:** Confirmed via direct file inspection of `crates/ritk-{core,registration,io,model,python,cli}`
**Status:** Active — feeds `backlog.md` and `checklist.md`

## Update Note

**Sprint 21 (2025-07-XX):** `BinaryFillHoles` (6-connected border flood-fill) and `MorphologicalGradient` (dilation AND NOT erosion) added to `ritk-core/src/segmentation/morphology/`. `multires_demons_register` Python binding and `multires-demons` CLI method added. DICOM binary writer replaced with real `InMemDicomObject` + `FileMetaTableBuilder` implementation; DICM magic verified.

`Analyze` format support is present in `crates/ritk-io/src/format/analyze/` and should be treated as implemented. This audit now focuses on the remaining imaging gaps relative to DICOM, ITK, SimpleITK, and VTK.

---

## Confirmed RITK Inventory (Source-Verified)

The following capabilities are **confirmed present** by reading `lib.rs` / `mod.rs` entry points and
selected implementation files. Items listed in comments or `TODO` blocks are excluded.

| Crate | Module | Confirmed Symbols |
|---|---|---|
| `ritk-core` | `filter` | `GaussianFilter`, `DownsampleFilter`, `ResampleImageFilter`, `MultiResolutionPyramid`, `N4BiasFieldCorrectionFilter`, `AnisotropicDiffusionFilter`, `GradientMagnitudeFilter`, `LaplacianFilter`, `FrangiVesselnessFilter`, `RecursiveGaussianFilter`, `CannyEdgeDetector`, `LaplacianOfGaussianFilter`, `GrayscaleErosion`, `GrayscaleDilation`, `SobelFilter` |
| `ritk-core` | `interpolation` | `BSplineInterpolator`, `LinearInterpolator` (1–4D), `NearestInterpolator`, `TensorTrilinearInterpolator` |
| `ritk-core` | `transform` | `AffineTransform`, `BSplineTransform`, `ChainedTransform`, `CompositeTransform`, `DisplacementFieldTransform`, `RigidTransform`, `ScaleTransform`, `StaticDisplacementFieldTransform`, `TransformDescription`, `TranslationTransform`, `VersorTransform` |
| `ritk-core` | `spatial` | `Direction`, `Point`, `Spacing`, `Vector` |
| `ritk-core` | `image` | `Image<B,D>`, `ImageGrid`, `ImageMetadata` |
| `ritk-core` | `segmentation` | `OtsuThreshold`, `MultiOtsuThreshold`, `BinaryErosion`, `BinaryDilation`, `BinaryOpening`, `BinaryClosing`, `ConnectedComponentsFilter`, `LabelStatistics`, `ConnectedThresholdFilter`, `LiThreshold`, `YenThreshold`, `KapurThreshold`, `TriangleThreshold`, `KMeansSegmentation`, `WatershedSegmentation`, `ChanVeseSegmentation`, `GeodesicActiveContourSegmentation` |
| `ritk-core` | `statistics` | `ImageStatistics`, `compute_statistics`, `masked_statistics`, `dice_coefficient`, `hausdorff_distance`, `mean_surface_distance`, `HistogramMatcher`, `MinMaxNormalizer`, `ZScoreNormalizer`, `estimate_noise_mad`, `psnr`, `ssim`, `NyulUdupaNormalizer` |
| `ritk-registration` | `metric` | `CorrelationRatio`, `LocalNCC`, `MSE`, `MutualInformation` (Standard / Mattes / NMI), `NCC`, DL-loss module, Parzen histogram |
| `ritk-registration` | `optimizer` | `AdamOptimizer`, `CmaEsOptimizer`, `GradientDescentOptimizer`, `MomentumOptimizer` |
| `ritk-registration` | `classical` | Kabsch-SVD landmark rigid (bug-fixed), MI hill-climb rigid/affine, temporal cross-correlation sync (bug-fixed) |
| `ritk-registration` | `demons` | `ThirionDemonsRegistration`, `DiffeomorphicDemonsRegistration`, `SymmetricDemonsRegistration` |
| `ritk-registration` | `diffeomorphic` | `SyNRegistration` (greedy SyN), `MultiResSyNRegistration` (coarse-to-fine pyramid, inverse consistency), `BSplineSyNRegistration` (B-spline velocity fields, bending energy) |
| `ritk-registration` | `regularization` | `BendingEnergy`, `Curvature`, `Diffusion`, `Elastic`, `TotalVariation` |
| `ritk-registration` | `multires` / `progress` / `validation` | `MultiResolutionSchedule`, `ProgressTracker`, `ConvergenceChecker`, `RegistrationQualityMetrics` |
| `ritk-registration` | `registration` (DL path) | `Registration`, `RegistrationConfig`, `RegistrationSummary`, DL-SSM registration, DL-loss |
| `ritk-registration` | `bspline_ffd` | `BSplineFFDRegistration`, `BSplineFFDConfig`, `BSplineFFDResult` |
| `ritk-registration` | `lddmm` | `LddmmRegistration` (geodesic shooting via EPDiff, Gaussian RKHS kernel) |
| `ritk-io` | `format` | DICOM reader/writer, NIfTI reader/writer, PNG reader/writer, MetaImage (.mha/.mhd) reader/writer, NRRD reader/writer, `TiffReader`, `TiffWriter` (multi-page z-stack, u8/u16/u32/f32/f64), `VtkReader`, `VtkWriter` (legacy structured points, ASCII/BINARY), `JpegReader`, `JpegWriter` (2-D grayscale, shape `[1,H,W]`), `MincReader`, `MincWriter` (MINC2 via consus HDF5), `AnalyzeReader`, `AnalyzeWriter` (Analyze 7.5 `.hdr`/`.img`); next stage: DICOM object model, private tags, nested sequences, multi-frame, and generalized writer architecture |
| `ritk-model` | — | `TransMorph`, `SSMMorph`, affine DL network |
| `ritk-python` | `image` | `PyImage` (NumPy bridge, `Arc<Image<NdArray,3>>`, ZYX convention) |
| `ritk-python` | `io` | `read_image`, `write_image` (NIfTI, PNG, DICOM, MetaImage, NRRD), `read_transform`, `write_transform` |
| `ritk-python` | `filter` | `gaussian_filter`, `median_filter`, `bilateral_filter`, `n4_bias_correction`, `anisotropic_diffusion`, `gradient_magnitude`, `laplacian`, `frangi_vesselness`, `canny`, `laplacian_of_gaussian`, `recursive_gaussian`, `sobel_gradient`, `grayscale_erosion`, `grayscale_dilation` |
| `ritk-python` | `registration` | `demons_register` (Thirion), `diffeomorphic_demons_register`, `symmetric_demons_register`, `syn_register`, `bspline_ffd_register`, `multires_syn_register`, `bspline_syn_register`, `lddmm_register`, `build_atlas`, `majority_vote_fusion`, `joint_label_fusion_py` |
| `ritk-python` | `segmentation` | `otsu_threshold`, `li_threshold`, `yen_threshold`, `kapur_threshold`, `triangle_threshold`, `multi_otsu`, `connected_components`, `connected_threshold`, `kmeans`, `watershed`, `binary_erosion`, `binary_dilation`, `binary_opening`, `binary_closing`, `chan_vese`, `geodesic_active_contour` |
| `ritk-cli` | `commands` | `convert`, `filter` (gaussian/n4-bias/anisotropic/gradient-magnitude/laplacian/frangi/median/bilateral/canny/sobel/log/recursive-gaussian), `register` (rigid-mi/affine-mi/demons/syn), `segment` (otsu/multi-otsu/connected-threshold/li/yen/kapur/triangle/watershed/kmeans/distance-transform), `stats` (summary/dice/hausdorff/psnr/ssim) |
| `ritk-io` | `format::dicom` | `scan_dicom_directory`, `load_dicom_series`, `read_dicom_series`, `load_dicom_series_with_metadata`, `read_dicom_series_with_metadata`, `DicomSeriesInfo`, `DicomReadMetadata`, `DicomSliceMetadata` |

**Absent or incomplete at module level (zero source files, stub-only, or partial fidelity):**  
Curvature anisotropic diffusion, Sato line / Hessian blob detection, confidence-connected region growing, neighborhood-connected region growing, skeletonization, hole filling, diffeomorphic Demons exact inverse, generalized DICOM object-model preservation, private tag round-trip, nested sequence emission, multi-frame / enhanced image handling, generalized DICOM write-path support, VTK polydata / grid data models, visualization pipeline abstractions, ITK-SNAP workflow state primitives, comparison harnesses against Python reference toolkits.

---

## 1. Executive Summary

RITK has a well-structured core (image primitives, transforms, interpolation) and a strong
registration layer (classical Kabsch/MI + deep-learning TransMorph/SSMMorph). It covers the
most performance-sensitive registration metrics (MI, NCC, LNCC, NMI) and a complete
regularization suite.

**Sprint 2 (2025-07-15) completed the following previously absent components:**
- `ritk-core/segmentation`: Otsu / multi-Otsu threshold, binary morphology (erosion, dilation,
  opening, closing), Hoshen-Kopelman connected-component labeling, connected-threshold
  region growing — all with full unit-test coverage (6- and 26-connectivity, statistics).
- `ritk-core/statistics`: `ImageStatistics`, masked statistics, Dice coefficient, Hausdorff
  distance, mean surface distance, histogram matching, min-max normalisation, z-score
  normalisation — all mathematically specified and property-tested.
- `ritk-io/format`: MetaImage (`.mha`/`.mhd`) and NRRD (`.nrrd`) readers/writers with full
  round-trip test coverage, ZYX ↔ XYZ axis permutation, and external-data-file support.

- `ritk-io/format`: Analyze 7.5 reader/writer support for `.hdr` / `.img` pairs.

**Sprint 3 (2025-07-16) completed the following previously absent components:**
- `ritk-core/filter/bias`: `N4BiasFieldCorrectionFilter` (Tustison 2010) — B-spline surface
  fitting via Tikhonov-regularised normal equations, Wiener-deconvolution histogram sharpening,
  multi-resolution coarse-to-fine bias estimation. Verified: partition-of-unity, round-trip
  fidelity, stability on discrete-histogram inputs, all-positive output invariant.
- `ritk-core/filter/edge`: `GradientMagnitudeFilter` (central-difference gradient with physical
  spacing), `LaplacianFilter` (second-order FD, one-sided at boundaries). Verified: uniform→0,
  ramp→exact gradient, non-unit spacing, quadratic→exact Laplacian at interior voxels.
- `ritk-core/filter/diffusion`: `AnisotropicDiffusionFilter` — Perona-Malik (1990) PDE with
  explicit Euler, exponential and quadratic conductance functions, Neumann BC, Δt=1/16 default.
  Verified: uniform image stable, step-edge preservation, mean conservation.
- `ritk-core/filter/vesselness`: `FrangiVesselnessFilter` (Frangi 1998) — discrete Hessian via
  second-order FD, analytic symmetric-3×3 eigenvalues (Kopp 2008), multiscale max aggregation,
  bright/dark vessel polarity gate. Verified: tube phantom>0.05, sphere suppression, polarity.
  Also: `compute_hessian_3d`, `symmetric_3x3_eigenvalues` (f64 precision, sorted by |λ|).
- `ritk-registration/demons`: `ThirionDemonsRegistration` (Thirion 1998) — optical-flow forces,
  fluid+diffusive regularisation, per-voxel magnitude clamping; `DiffeomorphicDemonsRegistration`
  (Vercauteren 2009) — stationary velocity field, scaling-and-squaring exp-map, BCH update;
  `SymmetricDemonsRegistration` (Pennec 1999) — combined fixed+moving gradient forces.
  Verified: identity MSE<1e-3, MSE decreases ≥50%, displacement finite, approximate symmetry.
- `ritk-registration/diffeomorphic`: `SyNRegistration` — greedy SyN with local cross-correlation
  metric (Avants 2008), symmetric forward/inverse velocity fields, scaling-and-squaring, Gaussian
  regularisation, VecDeque convergence window. Verified: identity CC>0.9, non-trivial fields,
  non-divergence, finite outputs, error on shape mismatch.
- `crates/ritk-cli`: New `ritk` binary crate with clap-derived CLI exposing `convert`, `filter`,
  `register`, and `segment` subcommands. All 5 filter variants (gaussian, n4-bias, anisotropic,
  gradient-magnitude, laplacian, frangi) now fully wired to real ritk-core implementations.
  59 tests passing (integration-style with tempfile).
- `ritk-python` extended: `n4_bias_correction`, `anisotropic_diffusion`, `gradient_magnitude`,
  `laplacian`, `frangi_vesselness` exposed in `ritk.filter`; `diffeomorphic_demons_register`,
  `symmetric_demons_register`, `syn_register` exposed in `ritk.registration`.
- `ritk-python`: Complete PyO3 0.22 extension (`_ritk`) with five submodules (`image`, `io`,
  `filter`, `registration`, `segmentation`), `abi3-py39` stable-ABI support (Python 3.9–3.14),
  MetaImage/NRRD IO wiring, Python package (`__init__.py`, `py.typed`, maturin config).
- **Bug fixes**: Kabsch SVD orientation (H matrix transposition), NMI degenerate constant-image
  case, temporal stability metric, histogram-matching self-match tolerance, connected-component
  26-connectivity diagonal test geometry — all root-cause fixes, zero tolerance relaxations.

**Sprint 4 (2025-07-17) completed the following previously absent components:**
- `ritk-core/filter`: `RecursiveGaussianFilter` (Deriche IIR, derivative orders 0/1/2),
  `CannyEdgeDetector` (Gaussian + gradient + NMS + double hysteresis), `LaplacianOfGaussianFilter`
  (separable Gaussian + Laplacian composition), `GrayscaleErosion` and `GrayscaleDilation`
  (flat structuring element, replicate padding).
- `ritk-core/segmentation/threshold`: Li minimum cross-entropy, Yen maximum correlation,
  Kapur maximum entropy, Triangle method — all with compute/apply API and convenience functions.
- `ritk-core/segmentation/clustering`: `KMeansSegmentation` (Lloyd's algorithm, k-means++
  deterministic initialization via embedded xorshift64 PRNG).
- `ritk-core/segmentation/watershed`: `WatershedSegmentation` (Meyer 1994 flooding on
  gradient magnitude, 6-connectivity).
- `ritk-core/statistics`: `estimate_noise_mad` / `estimate_noise_mad_masked` (MAD estimator,
  σ̂ = 1.4826 · median(|X - median(X)|)), `psnr` (Peak Signal-to-Noise Ratio), `ssim`
  (Structural Similarity, Wang et al. 2004 global formulation).
- `ritk-core/statistics/normalization`: `NyulUdupaNormalizer` (Nyúl-Udupa piecewise-linear
  histogram standardization, two-phase train/apply workflow).
- `ritk-registration/bspline_ffd`: `BSplineFFDRegistration` (Rueckert et al. 1999, multi-
  resolution BSpline control lattice, NCC metric, bending energy regularization, gradient descent
  on control points, subdivision-based refinement).
- **Test coverage**: 390 tests passing in ritk-core, 121 in ritk-registration, 59 in ritk-cli,
  36 in ritk-io = 606+ total. Zero failures.

**Sprint 5 (2025-07-18) completed the following previously absent components:**
- `ritk-core/segmentation/level_set`: `ChanVeseSegmentation` (Chan & Vese 2001, region-based
  active contour without edges, Mumford-Shah energy, curvature regularisation, interior/exterior
  mean fitting), `GeodesicActiveContourSegmentation` (Caselles et al. 1997, edge-based geodesic
  active contour, gradient stopping function g(|∇I|), curvature + advection PDE terms).
- `ritk-core/filter`: `SobelFilter` (3D Sobel gradient — separable 3×3×3 Sobel convolution
  producing gradient magnitude with physical spacing support).
- `ritk-core/filter`: Confirmed native `Image<B,D>` implementations for `MedianFilter` and
  `BilateralFilter` (previously mischaracterised as Python-only gaps; both operate directly
  on `Image<B,D>` in ritk-core).
- `ritk-python/segmentation`: Expanded from 2 → 16 functions: `otsu_threshold`, `li_threshold`,
  `yen_threshold`, `kapur_threshold`, `triangle_threshold`, `multi_otsu`, `connected_components`,
  `connected_threshold`, `kmeans`, `watershed`, `binary_erosion`, `binary_dilation`,
  `binary_opening`, `binary_closing`, `chan_vese`, `geodesic_active_contour`.
- `ritk-python/filter`: Expanded from 8 → 14 functions: added `canny`,
  `laplacian_of_gaussian`, `recursive_gaussian`, `sobel_gradient`, `grayscale_erosion`,
  `grayscale_dilation`.

Against **ITK** (≈1 200 image filters, full segmentation pipeline, 30+ IO formats), **SimpleITK**
(Python/R/Java/C# bindings, N4 bias field correction, histogram matching), **VTK**
(visualization, mesh/scene graph, polydata pipeline), **ITK-SNAP**
(interactive segmentation / annotation / overlay workflows), **ANTs**
(robust diffeomorphic registration workflows), and **Grassroots DICOM**
(comprehensive DICOM object model and interoperability tooling), RITK has **six structural gaps**
that collectively prevent it from being used as a drop-in toolkit in standard clinical or research
imaging workflows:

| Gap Domain | Severity | ITK Parity | SimpleITK Parity | ANTs Parity | VTK / DICOM Relevance |
|---|---|---|---|---|---|
| Segmentation | **High** | ~45% | ~45% | ~45% | ITK / SimpleITK: still missing a broad set of region, deformable, and topology-preserving operators |
| Filtering & Preprocessing | **High** | ~55% | ~55% | ~55% | ITK / SimpleITK: still missing the long tail of multiscale, PDE, and topology-aware filters |
| Diffeomorphic Registration | **Medium** | ~85% | ~85% | ~85% | ANTs: still lacking exact-inverse Demons and some production-grade inverse-consistency controls |
| Statistics & Normalization | **Medium** | ~55% | ~55% | ~55% | SimpleITK: broad utilities remain, but core normalization coverage is now substantial |
| IO Formats | **High** | ~58% | ~58% | ~58% | ITK / VTK / DICOM: still missing full codec breadth, mesh/scene formats, and deep DICOM object coverage |
| DICOM Read Metadata | **High** | N/A | N/A | N/A | DICOM: object-model preservation, private tags, nested sequences, and multi-frame / enhanced images remain incomplete |
| VTK Data Model | **High** | ~20% | ~20% | ~20% | VTK: image I/O exists, but data-object hierarchy, mesh grids, and pipeline abstractions are absent |
| ITK-SNAP Workflow | **Medium-High** | ~10% | ~10% | ~10% | ITK-SNAP: interactive segmentation state, labels, overlays, and undo/redo primitives are absent |
| VTK Data Model | **High** | ~20% | ~20% | ~20% | VTK: image I/O exists, but data-object hierarchy, mesh grids, and pipeline abstractions are absent |

| ITK-SNAP Workflow | **Medium-High** | ~10% | ~10% | ~10% | ITK-SNAP: interactive segmentation state, labels, overlays, and undo/redo primitives are absent |
| Python / CLI Bindings | **Low** | ~92% | ~92% | ~92% | SimpleITK: `ritk` is close on bindings breadth, but high-level façade conventions remain narrower |

Sprint 3 filter additions (N4, Perona-Malik, gradient magnitude, Laplacian, Frangi) moved
Filtering & Preprocessing from Critical to High severity. Addition of Thirion/Diffeomorphic/
Symmetric Demons and greedy SyN moved Diffeomorphic Registration from Critical to High severity.
The `ritk-cli` binary and extended Python bindings materially advanced CLI/Python parity.
The DICOM subsystem now has a read-side metadata slice that captures series identity plus per-slice geometry and rescale fields.

The DICOM implementation remains series-centric. The remaining DICOM backlog is:
- transfer syntax coverage audit
- private tag and nested sequence preservation strategy
- multi-frame / enhanced image support
- generalized DICOM writer
- non-image SOP-class policy
- metadata-aware read-path validation for `DicomReadMetadata` and `DicomSliceMetadata`
- object-model round-trip preservation
- explicit unknown-element retention
- object-model preservation across read/write round-trips
- explicit handling of sequence values and unknown elements

The next-stage roadmap is:
1. DICOM object-model foundation
2. VTK data model and mesh primitives
3. ITK/SimpleITK algorithm breadth expansion
4. ITK-SNAP workflow primitives
5. ANTs workflow refinement
6. Python comparison and reproducibility harness
- object-model round-trip preservation
- explicit unknown-element retention
- object-model preservation across read/write round-trips
- explicit handling of sequence values and unknown elements

The next-stage roadmap is:
1. DICOM object-model foundation
2. VTK data model and mesh primitives
3. ITK/SimpleITK algorithm breadth expansion
4. ITK-SNAP workflow primitives
5. ANTs workflow refinement
6. Python comparison and reproducibility harness

Sprint 5 level-set implementations (Chan-Vese, Geodesic Active Contour) raised Segmentation
parity from ~25% to ~35%. 3D Sobel gradient filter plus confirmation of native Median/Bilateral
`Image<B,D>` implementations raised Filtering from ~45% to ~55%. Full 16-function Python
segmentation API and 14-function Python filter API raised Python/CLI parity from ~50% to ~65%.

**Sprint 6 (2025-07-18) completed the following previously absent components:**
- `ritk-registration/diffeomorphic`: `MultiResSyNRegistration` (coarse-to-fine pyramid with
  level-doubling velocity fields, inverse consistency enforcement) and `BSplineSyNRegistration`
  (B-spline parameterized velocity fields, bending energy regularization). Closes GAP-R01.
- `ritk-registration/lddmm`: `LddmmRegistration` (geodesic shooting via EPDiff, Gaussian RKHS
  kernel, shooting-based registration from initial velocity to geodesic). Closes GAP-R03.
- `ritk-core/transform`: `CompositeTransform` and `TransformDescription` enum with JSON
  serialization/deserialization, round-trip file I/O (`composite_io.rs`). Closes GAP-R05.
- `ritk-io/format/tiff`: `TiffReader` and `TiffWriter` with multi-page z-stack support,
  u8/u16/u32/f32/f64 pixel types, BigTIFF for files >4 GB. Closes IO-07.
- `ritk-python/registration`: Expanded from 4 → 8 functions: added `bspline_ffd_register`,
  `multires_syn_register`, `bspline_syn_register`, `lddmm_register`. Closes PY-05.
- **Test coverage**: 421 tests passing in ritk-core, 150+ in ritk-registration, 50+ in ritk-io.
  Zero failures, zero warnings.

Sprint 6 multi-resolution SyN, BSplineSyN, and LDDMM raised Diffeomorphic Registration parity
from ~65% to ~80%. TIFF/BigTIFF reader/writer raised IO parity from ~30% to ~35%. Full 8-function
Python registration API raised Python/CLI parity from ~65% to ~75%.

**Sprint 7 (2025-07-18) completed the following previously absent components:**
- `ritk-registration/atlas`: `GroupwiseRegistration` (iterative template building via Multi-Res
  SyN, Avants & Gee 2004) and `JointLabelFusion` (Wang et al. 2013, patch-based locally weighted
  label voting + majority voting). Closes GAP-R04 and GAP-R06.
- `ritk-io/format/mgh`: `MghReader` and `MghWriter` with gzip compression (MGZ), 4 data types
  (u8, i32, f32, i16), FreeSurfer physical-space metadata. Closes IO-MGH.
- `ritk-core/segmentation/distance_transform`: Euclidean distance transform (Meijster et al.
  2000, linear-time separable algorithm). Closes SEG-DT.
- `ritk-core/statistics/normalization`: `WhiteStripeNormalization` (Shinohara et al. 2014,
  KDE-based white matter peak detection). Closes STA-09.
- `ritk-python/statistics`: 13 Python-callable statistics functions: image statistics,
  comparison metrics (Dice, Hausdorff, mean surface distance, PSNR, SSIM), normalization
  (z-score, min-max, histogram matching, Nyúl-Udupa), and white stripe normalization.
  Closes PY-STAT.
- **Test coverage**: 454 tests passing in ritk-core (+33), 162 in ritk-registration (+12),
  79 in ritk-io (+29). Zero failures, zero warnings.

Sprint 7 atlas registration and joint label fusion raised Diffeomorphic Registration parity
from ~80% to ~85%. MGH/MGZ reader/writer raised IO parity from ~35% to ~45%. Euclidean distance
transform raised Segmentation parity from ~35% to ~40%. White stripe normalization raised
Statistics parity from ~50% to ~55%. 13-function Python statistics API raised Python/CLI parity
from ~75% to ~80%.

**Sprint 8 (2025-07-18) completed the following previously absent components:**
- `ritk-io/format/vtk`: `VtkReader` and `VtkWriter` for VTK legacy structured-points images,
  supporting ASCII and BINARY payloads, big-endian binary encoding, and round-trip preservation
  of voxel values plus origin/spacing metadata. Closes IO-06.
- `ritk-io/format/jpeg`: `JpegReader` and `JpegWriter` for grayscale JPEG images, represented in
  RITK as 3-D images with shape `[1, height, width]`; writer rejects `nz != 1`. Closes IO-08.
- `ritk-cli`: completed command coverage for the implemented core algorithms:
  - `filter`: median, bilateral, canny, sobel, log, recursive-gaussian
  - `segment`: li, yen, kapur, triangle, watershed, kmeans, distance-transform
  - `register`: demons, syn
  - `stats`: summary, dice, hausdorff, psnr, ssim
  Closes PY-07.
- `ritk-python`: packaged `.pyi` type stubs and `py.typed`, plus Python-callable atlas building,
  majority-vote fusion, joint label fusion, and composite transform JSON I/O. Closes PY-08.
- **Verification status:** prior workspace verification recorded 864 passing tests, 0 failures.
  Current bounded reruns reached a passing `ritk-cli` suite (107 tests) before timeout while the
  workspace was rebuilding Python dependencies; no failing diagnostics were observed in captured output.

Sprint 8 VTK and JPEG support raised IO parity from ~45% to ~50%. Completed CLI command coverage
and packaged Python stubs raised Python/CLI parity from ~80% to ~90%. Atlas/label-fusion Python
exposure improved ANTs-style workflow parity without changing the underlying registration-core
parity classification.

Parity percentages are estimated against the feature count of each reference toolkit relevant to
medical 3D imaging use cases (excluding legacy 2D-only or deprecated filters).

---

## 2. Registration Gaps

### 2.1 Confirmed Present in RITK

| Algorithm | Notes |
|---|---|
| Rigid (landmark, Kabsch SVD) | `classical::engine::rigid_registration_landmarks` |
| Rigid (intensity, MI hill-climb) | `classical::engine::rigid_registration_mutual_info` |
| Affine (intensity, MI hill-climb) | `classical::engine::affine_registration_mutual_info` |
| DL deformable (TransMorph) | `ritk-model::transmorph` + `registration::dl_registration_loss` |
| DL deformable (SSMMorph) | `ritk-model::ssmmorph` + `registration::dl_ssm_registration` |
| Displacement field transform | `ritk-core::transform::DisplacementFieldTransform` |
| BSpline transform | `ritk-core::transform::BSplineTransform` |
| Multi-resolution schedule | `ritk-registration::multires` |

### 2.2 Gaps

#### GAP-R01 — SyN (Symmetric Normalization) · Severity: **Closed** (multi-resolution SyN and BSplineSyN implemented)

**Reference:** Avants et al. (2008), *Med. Image Anal.* 12(1):26–41.
ANTs' flagship algorithm. Symmetrically minimizes a geodesic distance in the space of
diffeomorphisms by composing forward (fixed→moving) and inverse (moving→fixed) displacement
fields updated at each iteration.

**Sprint 3**: Greedy SyN with local cross-correlation implemented (`SyNRegistration`).
- Forward and inverse stationary velocity fields (v₁, v₂)
- Scaling-and-squaring exponential map (n_squarings=6 default)
- Local CC gradient forces (Avants 2008, eq. 10)
- Gaussian velocity-field regularisation
- VecDeque-based convergence window

**Sprint 6**: All remaining gaps closed:
- `MultiResSyNRegistration` — coarse-to-fine pyramid with level-doubling velocity fields,
  inverse consistency enforcement (`ritk-registration/src/diffeomorphic/multires_syn.rs`)
- `BSplineSyNRegistration` — B-spline parameterized velocity fields, bending energy
  regularization (`ritk-registration/src/diffeomorphic/bspline_syn.rs`)

**Implemented location:** `crates/ritk-registration/src/diffeomorphic/`

---

#### GAP-R02 — Demons Registration Family · Severity: **Closed** (all three variants implemented)

**Sprint 3 status**: All three Demons variants are **implemented** and tested:
- `ThirionDemonsRegistration` (`demons/thirion.rs`) — optical-flow forces, fluid+diffusive reg.
- `DiffeomorphicDemonsRegistration` (`demons/diffeomorphic.rs`) — SVF + scaling-and-squaring
- `SymmetricDemonsRegistration` (`demons/symmetric.rs`) — combined gradient forces

**Implemented location:** `crates/ritk-registration/src/demons/`

---

#### GAP-R02b — Full Diffeomorphic Demons with Exact Inverse · Severity: **Medium** (new gap from Sprint 3)

**References:**
- Thirion (1998), *Med. Image Anal.* 2(3):243–260 (original Demons).
- Vercauteren et al. (2009), *NeuroImage* 45(S1):S61–S72 (Diffeomorphic Demons).
- Pennec et al. (1999), *MICCAI* (symmetric Demons).

Demons treats registration as a diffusion process driven by optical-flow-like forces.
Widely used as a fast deformable baseline for lung, liver, and cardiac motion estimation.
Diffeomorphic Demons uses the Lie algebra of diffeomorphisms (stationary velocity fields)
to guarantee invertibility.

Remaining gaps for production-grade diffeomorphic Demons:
- ICC (inverse consistency constraint) via exact field inversion (iterative Newton)
- Log-domain composition for large-deformation accuracy
- Multi-resolution pyramid driven by Demons (currently single-scale)

---

#### GAP-R03 — LDDMM (Large Deformation Diffeomorphic Metric Mapping) · Severity: **Closed** (implemented Sprint 6)

**Reference:** Beg et al. (2005), *Int. J. Comput. Vis.* 61(2):139–157.

LDDMM generates geodesic paths in the space of diffeomorphisms under a right-invariant
Riemannian metric. Necessary for morphometric analysis and atlas-based segmentation where
deformations exceed small-diffeomorphism assumptions.

**Sprint 6**: `LddmmRegistration` implemented with:
- Geodesic shooting via EPDiff (Euler-Poincaré equation on diffeomorphisms)
- Gaussian RKHS kernel on the velocity field
- Shooting-based registration (initial velocity → geodesic)
- Jacobian determinant computation for volume preservation metrics

**Implemented location:** `crates/ritk-registration/src/lddmm/mod.rs`

---

#### GAP-R04 — Groupwise / Atlas Registration · Severity: **Closed** (implemented Sprint 7)

**Reference:** Joshi et al. (2004), *MICCAI*; Guimond et al. (2000), *Comput. Vis. Image Underst.*;
Avants & Gee (2004).

Simultaneously registers N images to a latent mean template updated iteratively (Fréchet mean
in diffeomorphism space). Used for population studies, cortical thickness analysis, and
multi-atlas label propagation.

**Sprint 7**: Implemented iterative template building via Multi-Res SyN:
- `GroupwiseRegistration` with configurable iteration count and convergence threshold.
- Per-subject pairwise registration to current template estimate.
- Voxel-wise mean of warped images produces updated template each iteration.
- Warp averaging for diffeomorphic template update.
- 6 unit tests covering convergence, identity template, and multi-subject registration.

**Implemented location:** `crates/ritk-registration/src/atlas/mod.rs` (~483 lines)

---

#### GAP-R05 — Composite Transform Serialization · Severity: **Closed** (implemented Sprint 6)

RITK has `ChainedTransform` for runtime composition. Sprint 6 added full serialization
support via `CompositeTransform` and `TransformDescription` enum.

**Sprint 6**: Implemented:
- `CompositeTransform` with `TransformDescription` enum for type-safe serialization
- JSON serialization/deserialization with round-trip fidelity
- File I/O (`composite_io.rs`) for composed transform pipelines

**Implemented location:** `crates/ritk-core/src/transform/composite_io.rs`

---

#### GAP-R06 — Joint Label Fusion · Severity: **Closed** (implemented Sprint 7)

**Reference:** Wang et al. (2013), *IEEE Trans. Med. Imaging* 32(10):1837–1849.

Multi-atlas segmentation propagation with locally weighted label voting that accounts for
inter-atlas similarity. ANTs' `antsJointLabelFusion` is a standard pipeline step for
hippocampus, thalamus, and cortical parcel segmentation.

**Sprint 7**: Implemented Joint Label Fusion (Wang 2013) + Majority Voting:
- `JointLabelFusion` with patch-based local similarity weighting (constrained optimization).
- `MajorityVoting` for simple voxel-wise label consensus.
- Patch radius, regularization parameter (β), and search neighborhood configurable.
- Integration with atlas registration output (accepts pre-warped atlas images and labels).
- 16 unit tests covering single-atlas identity, multi-atlas consensus, tie-breaking,
  background handling, patch weighting correctness.

**Implemented location:** `crates/ritk-registration/src/atlas/label_fusion.rs` (~881 lines)

---

#### GAP-R07 — BSpline FFD Deformable Registration Pipeline · Severity: **High**

RITK has `BSplineTransform` in `ritk-core` and CMA-ES / gradient descent optimizers, but
has no assembled BSpline free-form deformation (FFD) registration pipeline (Rueckert et al.
1999) that drives control point optimization with a similarity metric over a multi-resolution
schedule.

**What is missing:**
- Control-point grid initialization from image geometry.
- Analytic / automatic gradient of MI/NCC w.r.t. control point displacements.
- Multi-resolution BSpline refinement (control-point doubling between levels).

---

## 3. Segmentation Gaps

**RITK has zero segmentation code.** The entire `segmentation` module tree is absent.
This is a Critical gap: segmentation is required in nearly every clinical pipeline
(tumor delineation, organ contouring, tissue classification, atlas propagation).

### 3.1 Threshold-Based Segmentation · Severity: **Critical**

| Algorithm | Reference | Notes |
|---|---|---|
| Otsu thresholding | Otsu (1979), *IEEE Trans. SMC* 9(1):62–66 | Maximizes inter-class variance; O(N) over histogram |
| Li thresholding | Li & Tam (1998), *Pattern Recognit. Lett.* 19(8) | Minimum cross-entropy |
| Yen thresholding | Yen et al. (1995), *J. Signal Process.* | Maximum correlation criterion |
| Kapur / Entropy | Kapur et al. (1985), *Comput. Vis.* | Maximum entropy |
| Multi-Otsu | Liao et al. (2001), *Image Vis. Comput.* | K-class generalization |
| Triangle method | Zack et al. (1977), *J. Histochem. Cytochem.* | Bimodal histogram assumption |
| Huang fuzzy | Huang & Wang (1995) | Fuzzy thresholding |

**Planned location:**
```
crates/ritk-core/src/segmentation/threshold/
├── mod.rs           # ThresholdSegmentation trait
├── otsu.rs
├── multi_otsu.rs
├── li.rs
├── yen.rs
├── kapur.rs
└── triangle.rs
```

### 3.2 Region Growing · Severity: **Critical**

| Algorithm | Notes |
|---|---|
| Connected threshold | Seeds + intensity interval; flood-fill |
| Neighborhood connected | Seeds + multi-neighbor consistency |
| Confidence connected | Iterative mean ± k·σ interval update |
| Isolated connected | Inverse-confidence connected |

**Planned location:**
```
crates/ritk-core/src/segmentation/region_growing/
├── mod.rs
├── connected_threshold.rs
├── neighborhood_connected.rs
└── confidence_connected.rs
```

### 3.3 Level Set Methods · Severity: **Closed** (Chan-Vese and Geodesic Active Contour implemented, Sprint 5)

**Sprint 5 status**: `ChanVeseSegmentation` and `GeodesicActiveContourSegmentation` are
**implemented** in `crates/ritk-core/src/segmentation/level_set/`.

| Algorithm | Reference | Status |
|---|---|---|
| Chan-Vese | Chan & Vese (2001), *IEEE Trans. Image Process.* 10(2):266–277 | ✓ Implemented (Sprint 5) |
| Geodesic Active Contour | Caselles et al. (1997), *IEEE Trans. Image Process.* 6(7):931–943 | ✓ Implemented (Sprint 5) |
| Shape Detection | Malladi et al. (1995), *IEEE Trans. Pattern Anal.* 17(2):158–175 | ✗ Not yet |
| Laplacian Level Set | ITK `LaplacianSegmentationLevelSetImageFilter` | ✗ Not yet |
| Threshold Level Set | ITK `ThresholdSegmentationLevelSetImageFilter` | ✗ Not yet |

Level sets evolve a signed-distance function φ under a PDE incorporating image gradient
stopping terms and curvature regularization:
`∂φ/∂t = F|∇φ|` where `F = g(|∇I|)(κ + α·advection)`.

**Implemented:**
- `ChanVeseSegmentation`: Region-based active contour without edges (Mumford-Shah energy),
  level-set evolution with curvature regularisation, interior/exterior mean fitting.
- `GeodesicActiveContourSegmentation`: Edge-based geodesic active contour, gradient stopping
  function g(|∇I|), curvature + advection PDE terms.

**Remaining:** Shape detection, Laplacian level set, threshold level set, sparse-field solver.

**Implemented location:** `crates/ritk-core/src/segmentation/level_set/`

### 3.4 Watershed Segmentation · Severity: **Medium**

**Sprint 4 status**: `WatershedSegmentation` is **implemented** in `crates/ritk-core/src/segmentation/watershed/mod.rs`. Meyer flooding, 6-connectivity. Marker-controlled variant remains.

Meyer (1994) flooding algorithm on gradient magnitude image.
Produces over-segmented basins merged via basin-adjacency graph.
Used for cell counting and 3D structure delineation.

**Planned location:**
```
crates/ritk-core/src/segmentation/watershed/
├── mod.rs
├── immersion.rs     # Meyer flooding algorithm
└── marker_controlled.rs
```

### 3.5 K-Means Clustering Segmentation · Severity: **Medium**

**Sprint 4 status**: `KMeansSegmentation` is **implemented** in `crates/ritk-core/src/segmentation/clustering/kmeans.rs`. Lloyd's algorithm with k-means++ initialization, deterministic seeding.

Lloyd's algorithm initialized by k-means++ (Arthur & Vassilvitskii 2007).
Used for tissue class initialization (CSF / GM / WM in brain MRI).

**Planned location:** `crates/ritk-core/src/segmentation/clustering/kmeans.rs`

### 3.6 Morphological Operations · Severity: **Low**

Essential post-processing for every segmentation pipeline.

| Operation | Mathematical Definition |
|---|---|
| Erosion | `(A ⊖ B)(x) = min_{b∈B} A(x+b)` |
| Dilation | `(A ⊕ B)(x) = max_{b∈B} A(x-b)` |
| Opening | `A ∘ B = (A ⊖ B) ⊕ B` |
| Closing | `A • B = (A ⊕ B) ⊖ B` |
| Morphological gradient | `(A ⊕ B) − (A ⊖ B)` — ✓ **MorphologicalGradient** (Sprint 21, `ritk-core/src/segmentation/morphology/morphological_gradient.rs`) |
| Distance transform | Exact Euclidean via Meijster et al. (2000) — ✓ **Implemented** (Sprint 7, `ritk-core/src/segmentation/distance_transform/`, 19 tests) |
| Skeletonization | Thinning via topology-preserving erosion |
| Hole filling | Geodesic dilation constrained by mask — ✓ **BinaryFillHoles** (Sprint 21, `ritk-core/src/segmentation/morphology/fill_holes.rs`) |
| Label voting | Majority vote in structuring element neighborhood |

**Planned location:**
```
crates/ritk-core/src/segmentation/morphology/
├── mod.rs           # MorphologicalOperation trait
├── erosion.rs
├── dilation.rs
├── opening.rs
├── closing.rs
├── distance_transform.rs
└── skeletonization.rs
```

### 3.7 Connected Component Analysis · Severity: **Critical**

Union-Find (Hoshen-Kopelman) connected component labeling.
Required output for: measuring lesion count, volume, shape descriptors.

| Feature | Notes |
|---|---|
| Binary connected components | 6/18/26-connectivity in 3D |
| Labeled component map | Each component gets unique integer label |
| Per-component statistics | Volume, centroid, bounding box, principal axes |
| Component filtering | Remove components by size, shape, or position |

**Planned location:**
```
crates/ritk-core/src/segmentation/labeling/
├── mod.rs
├── connected_components.rs  # Hoshen-Kopelman + union-find
└── label_statistics.rs
```

---

## 4. Filtering Gaps

RITK implements 4 filters. ITK implements approximately 250 image filters covering noise
reduction, edge detection, feature extraction, and bias correction.

### 4.1 N4 Bias Field Correction · Severity: **Closed** (implemented Sprint 3)

**Sprint 3 status**: `N4BiasFieldCorrectionFilter` is **implemented** in
`crates/ritk-core/src/filter/bias/`.

**Implemented:**
- Uniform cubic B-spline surface fitting via Tikhonov-regularised normal equations
  (nalgebra LU decomposition, partition-of-unity basis verified analytically)
- Wiener-deconvolution histogram sharpening in DFT domain (normalised histogram,
  concentration guard for discrete-spike inputs)
- Multi-resolution coarse-to-fine loop with control-point doubling per level
- `N4Config` with full parameter set: levels, iterations, convergence threshold,
  histogram bins, noise estimate, fitting points

**Known limitation (documented in tests):** For synthetic images with discrete
intensity levels (few distinct voxel values), the histogram sharpening step cannot
distinguish bias-induced spreading from the distribution itself. Real MRI data with
continuous Gaussian-noise-broadened tissue peaks converges correctly (verified by
`histogram_sharpen_continuous_bimodal_reduces_spread` test).

**Implemented location:** `crates/ritk-core/src/filter/bias/`

---

### 4.2 Anisotropic Diffusion · Severity: **Closed** (implemented Sprint 3)

**Sprint 3 status**: `AnisotropicDiffusionFilter` (Perona-Malik 1990) is **implemented** in
`crates/ritk-core/src/filter/diffusion/perona_malik.rs`.

**Implemented:** Explicit Euler FD, exponential and quadratic conductance functions,
Neumann (zero-flux) BC, Δt=1/16 stability default, `DiffusionConfig` with all parameters.

**Remaining:** Curvature anisotropic diffusion (Alvarez 1992), vector variant for tensors.

**Implemented location:** `crates/ritk-core/src/filter/diffusion/`

---

### 4.2b Gradient Magnitude / Sobel · Severity: **Closed** (implemented Sprint 3)

`GradientMagnitudeFilter` and `LaplacianFilter` implemented in
`crates/ritk-core/src/filter/edge/`. Central differences with physical spacing, one-sided
at boundaries. Both verified against exact analytical solutions.

---

### 4.3 Median Filter · Severity: **Closed** (native `Image<B,D>` implementation confirmed, Sprint 5)

Rank-order noise removal preserving edges. Removes salt-and-pepper noise without Gaussian
blurring. Used as a fast pre-step before level-set initialization.

**Sprint 5 status**: Native `Image<B,D>` implementation confirmed present in `ritk-core`.
Also exposed as `ritk.filter.median_filter` in the Python binding.
Previously mischaracterised as Python-only; the `ritk-core` `MedianFilter` operates directly
on `Image<B,D>`.

---

### 4.4 Bilateral Filter · Severity: **Closed** (native `Image<B,D>` implementation confirmed, Sprint 5)

Tomasi & Manduchi (1998). Joint spatial-range Gaussian weighting:

`BF[I](x) = (1/W(x)) Σ_p I(p) · G_σs(|x-p|) · G_σr(|I(x)-I(p)|)`

**Sprint 5 status**: Native `Image<B,D>` implementation confirmed present in `ritk-core`.
Also exposed as `ritk.filter.bilateral_filter` in the Python binding.
Previously mischaracterised as Python-only; the `ritk-core` `BilateralFilter` operates directly
on `Image<B,D>`.

---

### 4.5 Canny Edge Detection · Severity: **Medium**

**Sprint 4 status**: `CannyEdgeDetector` is **implemented** in `crates/ritk-core/src/filter/edge/canny.rs`.

Canny (1986) multi-stage algorithm:
1. Gaussian smoothing.
2. Gradient magnitude + orientation via Sobel/Prewitt.
3. Non-maximum suppression along gradient direction.
4. Double hysteresis thresholding.

Required for: initializing level-set contours, feature extraction for classical registration.

**Planned location:** `crates/ritk-core/src/filter/edge/canny.rs`

---

### 4.6 Hessian-Based Vesselness (Frangi Filter) · Severity: **Closed** (implemented Sprint 3)

**Sprint 3 status**: `FrangiVesselnessFilter` (Frangi 1998) is **implemented** in
`crates/ritk-core/src/filter/vesselness/`.

**Implemented:**
- `compute_hessian_3d`: 6-component second-order FD with physical spacing
- `symmetric_3x3_eigenvalues`: closed-form trigonometric method (f64 precision, sorted by |λ|)
- `FrangiVesselnessFilter::apply`: multiscale max aggregation, bright/dark polarity gate,
  R_A/R_B/S feature ratios, `FrangiConfig` with α/β/γ/scales/bright_vessels

**Remaining:** Sato line filter, Hessian-based blob detection.

**Implemented location:** `crates/ritk-core/src/filter/vesselness/`

---

### 4.7 Discrete and Recursive Gaussian · Severity: **High**

**Sprint 4 status**: `RecursiveGaussianFilter` is **implemented** in `crates/ritk-core/src/filter/recursive_gaussian.rs`. Deriche IIR 3rd-order approximation with derivative orders 0 (smoothing), 1 (first derivative), 2 (second derivative). Separable application across all 3D axes with physical spacing support.

RITK has a `GaussianFilter` but it is a single implementation. ITK separately provides:

| Filter | Algorithm | Use Case |
|---|---|---|
| `DiscreteGaussianImageFilter` | Convolution with sampled Gaussian kernel | Accurate smoothing, small σ |
| `RecursiveGaussianImageFilter` | Deriche IIR approximation (Deriche 1993) | Fast large-σ smoothing, derivatives |
| `SmoothingRecursiveGaussianImageFilter` | Separable recursive Gaussian | Standard preprocessing |

The recursive variant is O(N) regardless of σ, critical for large-volume 3D MRI.
Derivatives (first, second) via recursive Gaussian are required by gradient-based registration
and Hessian-based filters.

**Planned location:** `crates/ritk-core/src/filter/gaussian/` (extend existing module)

---

### 4.8 Laplacian of Gaussian / Laplacian · Severity: **Medium**

**Sprint 4 status**: `LaplacianOfGaussianFilter` is **implemented** in `crates/ritk-core/src/filter/edge/log.rs`.

`LoG(x) = -1/(πσ⁴)[1 - |x|²/2σ²]exp(-|x|²/2σ²)` — blob detection, edge enhancement.

**Planned location:** `crates/ritk-core/src/filter/edge/laplacian.rs`

---

### 4.9 3D Sobel Gradient Filter · Severity: **Closed** (implemented Sprint 5)

**Sprint 5 status**: `SobelFilter` is **implemented** in `crates/ritk-core/src/filter/`.
Separable 3×3×3 Sobel convolution producing gradient magnitude from central-difference
approximations with physical spacing support. Complements the basic `GradientMagnitudeFilter`
(§4.2b, Sprint 3) with the standard Sobel kernel weighting.

Required by: level-set stopping function, Canny, Frangi, classical registration preconditioning.
Also exposed as `ritk.filter.sobel_gradient` in the Python binding (Sprint 5).

---

### 4.10 Morphological Filters (Structuring-Element Based) · Severity: **High**

**Sprint 4 status**: Grayscale erosion and dilation are **implemented** in `crates/ritk-core/src/filter/morphology/`. Binary fill holes, label dilation, and skeletonization remain.

Binary and grayscale morphological filters as standalone preprocessing operations
(distinct from the segmentation post-processing morphology in §3.6):

- Grayscale erosion / dilation (flat structuring element).
- Morphological opening / closing for artifact removal.
- Binary fill holes.
- Label dilation for label propagation.

**Planned location:** `crates/ritk-core/src/filter/morphology/`

---

## 5. Statistics & Preprocessing Gaps

### 5.1 Histogram Matching · Severity: **Critical**

**Reference:** ITK `HistogramMatchingImageFilter`; SimpleITK `HistogramMatching`.

Nonlinear intensity normalization that maps the histogram of a source image to match a
reference image's histogram via piecewise-linear interpolation of quantile-quantile pairs.
Mandatory preprocessing step in every multi-atlas registration pipeline to reduce
inter-subject and inter-scanner intensity bias.

**Algorithm:**
1. Compute CDFs of source and reference images.
2. Build piecewise-linear mapping: for each quantile level `q`, map `F_src⁻¹(q)` → `F_ref⁻¹(q)`.
3. Apply mapping as a lookup table to all voxels.

**Planned location:** `crates/ritk-core/src/statistics/normalization/histogram_matching.rs`

---

### 5.2 Nyúl & Udupa Histogram Equalization · Severity: **High**

**Sprint 4 status**: `NyulUdupaNormalizer` is **implemented** in `crates/ritk-core/src/statistics/normalization/nyul_udupa.rs`. Two-phase train/apply with configurable percentile landmarks.

**Reference:** Nyúl & Udupa (1999), *IEEE Trans. Med. Imaging* 18(4):301–306;
Nyúl et al. (2000), *IEEE Trans. Med. Imaging* 19(2):143–150.

Piecewise-linear MRI intensity standardization. Learns landmark percentiles from a training
cohort and maps all images to a common intensity scale. The standard method for multi-site
MRI normalization in clinical studies.

**Planned location:** `crates/ritk-core/src/statistics/normalization/nyul_udupa.rs`

---

### 5.3 Intensity Normalization Suite · Severity: **High**

| Method | Formula | Use Case |
|---|---|---|
| Z-score | `(I - μ) / σ` | Zero-mean unit-variance normalization |
| Min-max | `(I - I_min) / (I_max - I_min)` | Rescale to [0, 1] |
| Percentile clip | Clamp to [p₁, p₉₉] then min-max | Robust to outliers |
| White stripe | Shinohara et al. (2014) — brain-specific | WM peak normalization — ✓ **Implemented** (Sprint 7) |

**Sprint 7 status**: `WhiteStripeNormalization` is **implemented** in `crates/ritk-core/src/statistics/normalization/white_stripe.rs`. KDE-based white matter peak detection (Shinohara et al. 2014), 14 unit tests. Z-score, min-max, percentile-clip, and histogram matching were implemented in prior sprints.

**Planned location:** `crates/ritk-core/src/statistics/normalization/`

---

### 5.4 Image Statistics · Severity: **Critical**

Currently RITK exposes no image-level statistics API. ITK provides:

| Statistic | Notes |
|---|---|
| Min / max / mean / variance / sum | Per image, per channel |
| Percentiles (arbitrary `p`) | Required for robust normalization |
| Masked statistics | Statistics restricted to a binary mask |
| Label statistics | Per-label min/max/mean/volume via `LabelStatisticsImageFilter` |
| Histogram | Fixed-bin or adaptive-bin 1D intensity histogram |

**Planned location:**
```
crates/ritk-core/src/statistics/
├── mod.rs
├── image_statistics.rs    # Min, max, mean, variance, percentile
├── masked_statistics.rs   # Mask-gated statistics
└── label_statistics.rs    # Per-label statistics over labeled map
```

---

### 5.5 Noise Estimation · Severity: **Medium**

**Sprint 4 status**: `estimate_noise_mad` and `estimate_noise_mad_masked` are **implemented** in `crates/ritk-core/src/statistics/noise_estimation.rs`.

Median-absolute-deviation (MAD) estimator: `σ̂ = 1.4826 · MAD(I)`.
Used to set adaptive regularization weights and threshold parameters.

**Planned location:** `crates/ritk-core/src/statistics/noise_estimation.rs`

---

### 5.6 Image Comparison Metrics · Severity: **Medium**

Distinct from registration metrics (which are differentiable losses); these are
evaluation-time quality measures:

| Metric | Formula |
|---|---|
| PSNR | `10 log₁₀(MAX²/MSE)` |
| SSIM | Structural similarity (Wang et al. 2004) |
| Dice coefficient | `2|A∩B| / (|A|+|B|)` — for segmentation evaluation |
| Hausdorff distance | `max(h(A,B), h(B,A))` |
| Average surface distance | `(1/|∂A|) Σ_{a∈∂A} d(a, ∂B)` |

**Planned location:** `crates/ritk-core/src/statistics/image_comparison.rs`

**Sprint 4 status**: `psnr` (Peak Signal-to-Noise Ratio) and `ssim` (Structural Similarity, Wang et al. 2004) are now **implemented** in `crates/ritk-core/src/statistics/image_comparison.rs`. Dice, Hausdorff, and average surface distance were implemented in prior sprints.

---

## 6. IO Gaps

RITK supports DICOM, NIfTI, and PNG. Medical imaging workflows require 10+ additional formats.

### 6.1 MetaImage (.mha / .mhd) · Severity: **Critical**

The default ITK image format. Nearly every ITK example, tutorial, and benchmark dataset uses
`.mha` or `.mhd`. Without MetaImage support, RITK cannot consume standard ITK test data
or participate in Medical Segmentation Decathlon / Learn2Reg benchmarks.

Format: ASCII header (`.mhd`) + binary raw data file; or combined (`.mha`).
Header encodes: dimensions, element type, spacing, origin, direction cosines.

**Planned location:**
```
crates/ritk-io/src/format/metaimage/
├── mod.rs
├── reader.rs
└── writer.rs
```

---

### 6.2 NRRD Format · Severity: **High**

**Reference:** Gordon Kindlmann, Teem library.

Used by 3D Slicer (the dominant open-source clinical workstation) and Camino.
Supports arbitrary dimensions, rich metadata, inline and detached data.
Required for interoperability with Slicer-based annotation workflows.

**Planned location:** `crates/ritk-io/src/format/nrrd/`

---

### 6.3 MINC Format (.mnc / .mnc2) · Severity: **Closed** (Sprint 12)

The format of the MNI (Montreal Neurological Institute) standard brain atlases.
HDF5-based (MNC2) with rich neuroimaging metadata. Used by ANTs for the MNI152 template.
Without MINC support, ANTs-standard atlas workflows cannot load their reference templates.

**Sprint 12**: Implemented `MincReader` and `MincWriter`:
- HDF5 parsing via `consus-hdf5` (pure-Rust, no C FFI)
- Dimension metadata extraction (start, step, length, direction_cosines)
- Spatial metadata derivation (origin, spacing, direction matrix)
- Data type conversion (u8, i8, u16, i16, u32, i32, f32, f64 → f32)
- Dimorder-aware axis mapping (default: zspace,yspace,xspace)
- Writer constructs valid HDF5 binary with MINC2 group hierarchy
- 27 unit tests covering conversion, spatial metadata, dimorder parsing

**Implemented location:** `crates/ritk-io/src/format/minc/` (~900 lines: `mod.rs`, `reader.rs`, `writer.rs`)

---

### 6.4 VTK Image Format (.vtk / .vti) · Severity: **Medium**

VTK legacy structured points format (`.vtk`) and VTK XML image data (`.vti`).
Used by ParaView for 3D visualization and by several segmentation export pipelines.

**Planned location:** `crates/ritk-io/src/format/vtk/`

---

### 6.5 TIFF / BigTIFF Support · Severity: **Closed** (implemented Sprint 6)

TIFF is the standard format for:
- Histopathology whole-slide images (WSI).
- Microscopy z-stacks.
- Multi-channel fluorescence data.

BigTIFF is required for files >4 GB (common in WSI).

**Sprint 6**: `TiffReader` and `TiffWriter` implemented with:
- Multi-page z-stack support (3D volume from TIFF page sequence)
- Pixel types: u8, u16, u32, f32, f64
- BigTIFF support for files >4 GB

**Implemented location:** `crates/ritk-io/src/format/tiff/`

---

### 6.6 Analyze Format (.hdr / .img) · Severity: **Low**

Legacy Mayo Clinic format. Superseded by NIfTI but still present in older datasets
(pre-2004 neuroimaging archives). Required for full backward compatibility.

**Planned location:** `crates/ritk-io/src/format/analyze/`

---

### 6.7 MGZ / MGH Format · Severity: **Closed** (implemented Sprint 7)

FreeSurfer's native volumetric format. Required for interoperability with cortical surface
analysis pipelines. MGH is the raw format; MGZ is gzip-compressed MGH.

**Sprint 7**: Implemented `MghReader` and `MghWriter`:
- MGH binary format with big-endian byte order, 4 data types (u8, i32, f32, i16).
- MGZ gzip-compressed variant (auto-detected via magic bytes / `.mgz` extension).
- FreeSurfer physical-space metadata (vox2ras matrix, goodRASFlag).
- Round-trip fidelity verified across all data types.
- 28 unit tests covering read/write, compression, data type conversion, metadata preservation.

**Implemented location:** `crates/ritk-io/src/format/mgh/` (~2100 lines: `mod.rs`, `reader.rs`, `writer.rs`)

---

### 6.8 JPEG 2D Support · Severity: **Low**

Natural images and 2D radiographs. JPEG lossy compression. Not suitable for quantitative
medical analysis (lossy artifacts) but required for compatibility with DICOM secondary capture
objects that embed JPEG-compressed pixel data.

**Planned location:** `crates/ritk-io/src/format/jpeg/`

---

## 7. Python Binding Gaps

### 7.1 Python Bindings — Sprint 7 Updated · Severity: **Medium** (was High)

`ritk-python` is a PyO3 0.22 native extension (`cdylib`) with six submodules.
`abi3-py39` enables CPython 3.9–3.14 compatibility without recompilation.
Sprint 3 added 8 new functions to `ritk.filter` and 3 new functions to `ritk.registration`.
Sprint 5 expanded `ritk.filter` to 14 functions and `ritk.segmentation` to 16 functions,
providing full coverage of all implemented ritk-core segmentation and filter algorithms.
Sprint 6 expanded `ritk.registration` from 4 → 8 functions: added `bspline_ffd_register`,
`multires_syn_register`, `bspline_syn_register`, `lddmm_register`.
Sprint 7 added `ritk.statistics` submodule with 13 functions: image statistics, comparison
metrics (Dice, Hausdorff, mean surface distance, PSNR, SSIM), normalization (z-score, min-max,
histogram matching, Nyúl-Udupa), and white stripe normalization.

Remaining gaps relative to SimpleITK / ANTsPy:
- No `maturin develop` / wheel publish workflow verified end-to-end in CI.
- No transform serialisation / `read_transform` / `write_transform` Python API.
- No type stubs (`.pyi` files) for IDE autocomplete.
- `py.allow_threads` GIL release not yet applied to long-running filter/registration calls.
- No Python API for atlas building or joint label fusion (Rust implementations available).

### 7.2 Python API Surface · Severity: **Medium** (was High — significantly expanded through Sprint 5)

| Capability | SimpleITK Equivalent | ANTsPy Equivalent | RITK Status |
|---|---|---|---|
| Image read/write | `sitk.ReadImage` / `sitk.WriteImage` | `ants.image_read` / `ants.image_write` | ✓ `ritk.io.read_image` / `write_image` (NIfTI, PNG, DICOM, MetaImage, NRRD) |
| NumPy ↔ Image conversion | `sitk.GetArrayFromImage` / `sitk.GetImageFromArray` | `ants.from_numpy` / `img.numpy()` | ✓ `ritk.Image(array)` / `img.to_numpy()` |
| Gaussian filter | `sitk.SmoothingRecursiveGaussian(img, σ)` | — | ✓ `ritk.filter.gaussian_filter(img, sigma)` |
| Median filter | `sitk.Median(img, radius)` | — | ✓ `ritk.filter.median_filter(img, radius)` |
| Bilateral filter | `sitk.Bilateral(img, σ_s, σ_r)` | — | ✓ `ritk.filter.bilateral_filter(img, σ_s, σ_r)` |
| N4 bias correction | `sitk.N4BiasFieldCorrection` | `ants.n4_bias_field_correction` | ✓ `ritk.filter.n4_bias_correction(img, levels, iters, noise)` |
| Anisotropic diffusion | `sitk.GradientAnisotropicDiffusion` | — | ✓ `ritk.filter.anisotropic_diffusion(img, iters, K)` |
| Gradient magnitude | `sitk.GradientMagnitude` | — | ✓ `ritk.filter.gradient_magnitude(img)` |
| Laplacian | `sitk.Laplacian` | — | ✓ `ritk.filter.laplacian(img)` |
| Vesselness | `sitk.ObjectnessMeasure` | — | ✓ `ritk.filter.frangi_vesselness(img, scales, α, β, γ)` |
| Canny edge detection | `sitk.CannyEdgeDetection` | — | ✓ `ritk.filter.canny(img, low, high, sigma)` (Sprint 5) |
| Laplacian of Gaussian | `sitk.LaplacianRecursiveGaussian` | — | ✓ `ritk.filter.laplacian_of_gaussian(img, sigma)` (Sprint 5) |
| Recursive Gaussian | `sitk.RecursiveGaussian` | — | ✓ `ritk.filter.recursive_gaussian(img, sigma, order)` (Sprint 5) |
| Sobel gradient | `sitk.SobelEdgeDetection` | — | ✓ `ritk.filter.sobel_gradient(img)` (Sprint 5) |
| Grayscale erosion | `sitk.GrayscaleErode` | — | ✓ `ritk.filter.grayscale_erosion(img, radius)` (Sprint 5) |
| Grayscale dilation | `sitk.GrayscaleDilate` | — | ✓ `ritk.filter.grayscale_dilation(img, radius)` (Sprint 5) |
| Demons registration | `sitk.DemonsRegistrationFilter` | — | ✓ `ritk.registration.demons_register` (Thirion) |
| Diffeomorphic Demons | `sitk.FastSymmetricForcesDemonsRegistration` | — | ✓ `ritk.registration.diffeomorphic_demons_register` |
| Symmetric Demons | — | — | ✓ `ritk.registration.symmetric_demons_register` |
| SyN registration | `sitk.SimpleElastix` | `ants.registration(type_of_transform='SyN')` | ✓ `ritk.registration.syn_register` (greedy SyN + local CC) |
| Multi-Res SyN registration | — | `ants.registration(type_of_transform='SyN')` | ✓ `ritk.registration.multires_syn_register` (Sprint 6) |
| BSpline SyN registration | — | `ants.registration(type_of_transform='BSplineSyN')` | ✓ `ritk.registration.bspline_syn_register` (Sprint 6) |
| LDDMM registration | — | — | ✓ `ritk.registration.lddmm_register` (Sprint 6) |
| Otsu thresholding | `sitk.OtsuThreshold` | `ants.get_mask` | ✓ `ritk.segmentation.otsu_threshold(img)` |
| Li thresholding | `sitk.LiThreshold` | — | ✓ `ritk.segmentation.li_threshold(img)` (Sprint 5) |
| Yen thresholding | `sitk.YenThreshold` | — | ✓ `ritk.segmentation.yen_threshold(img)` (Sprint 5) |
| Kapur thresholding | `sitk.MaximumEntropyThreshold` | — | ✓ `ritk.segmentation.kapur_threshold(img)` (Sprint 5) |
| Triangle thresholding | `sitk.TriangleThreshold` | — | ✓ `ritk.segmentation.triangle_threshold(img)` (Sprint 5) |
| Multi-Otsu thresholding | `sitk.OtsuMultipleThresholds` | — | ✓ `ritk.segmentation.multi_otsu(img, classes)` (Sprint 5) |
| Connected components | `sitk.ConnectedComponent` | — | ✓ `ritk.segmentation.connected_components(mask, connectivity)` |
| Connected threshold | `sitk.ConnectedThreshold` | — | ✓ `ritk.segmentation.connected_threshold(img, seeds, lo, hi)` (Sprint 5) |
| K-means segmentation | — | `ants.kmeans_segmentation` | ✓ `ritk.segmentation.kmeans(img, k)` (Sprint 5) |
| Watershed | `sitk.MorphologicalWatershed` | — | ✓ `ritk.segmentation.watershed(img)` (Sprint 5) |
| Binary erosion | `sitk.BinaryErode` | — | ✓ `ritk.segmentation.binary_erosion(mask, radius)` (Sprint 5) |
| Binary dilation | `sitk.BinaryDilate` | — | ✓ `ritk.segmentation.binary_dilation(mask, radius)` (Sprint 5) |
| Binary opening | `sitk.BinaryMorphologicalOpening` | — | ✓ `ritk.segmentation.binary_opening(mask, radius)` (Sprint 5) |
| Binary closing | `sitk.BinaryMorphologicalClosing` | — | ✓ `ritk.segmentation.binary_closing(mask, radius)` (Sprint 5) |
| Chan-Vese segmentation | — | — | ✓ `ritk.segmentation.chan_vese(img, iters)` (Sprint 5) |
| Geodesic active contour | `sitk.GeodesicActiveContourLevelSet` | — | ✓ `ritk.segmentation.geodesic_active_contour(img, init)` (Sprint 5) |
| Transform I/O | `sitk.ReadTransform` / `sitk.WriteTransform` | `ants.read_transform` | ✗ not yet implemented |
| BSpline FFD registration | `sitk.ElastixImageFilter` | — | ✓ `ritk.registration.bspline_ffd_register` (Sprint 6) |
| Image statistics | — | — | ✓ `ritk.statistics.image_statistics(img)` (Sprint 7) |
| Z-score normalization | — | — | ✓ `ritk.statistics.zscore_normalize(img)` (Sprint 7) |
| Min-max normalization | — | — | ✓ `ritk.statistics.minmax_normalize(img)` (Sprint 7) |
| Histogram matching | `sitk.HistogramMatching` | — | ✓ `ritk.statistics.histogram_matching(img, ref)` (Sprint 7) |
| Nyúl-Udupa normalization | — | — | ✓ `ritk.statistics.nyul_udupa_normalize(img)` (Sprint 7) |
| White stripe normalization | — | `ants.white_stripe` | ✓ `ritk.statistics.white_stripe_normalize(img)` (Sprint 7) |
| Dice coefficient | — | — | ✓ `ritk.statistics.dice_coefficient(a, b)` (Sprint 7) |
| Hausdorff distance | — | — | ✓ `ritk.statistics.hausdorff_distance(a, b)` (Sprint 7) |
| PSNR | — | — | ✓ `ritk.statistics.psnr(img, ref)` (Sprint 7) |
| SSIM | — | — | ✓ `ritk.statistics.ssim(img, ref)` (Sprint 7) |
| Joint label fusion | — | `ants.joint_label_fusion` | ✗ not yet in Python (Rust impl available, Sprint 7) |
| Atlas building | — | `ants.build_template` | ✗ not yet in Python (Rust impl available, Sprint 7) |

### 7.3 Implementation Status · Severity: **Medium** (implemented; minor gaps remain)

**Technology:** PyO3 0.22 with `maturin` build backend, `abi3-py39` stable ABI.
**Interop:** `numpy` crate (`PyReadonlyArray3`, `IntoPyArray`) via `pyo3-numpy`.

**Sprint 7 function counts:** 14 filter functions, 16 segmentation functions, 8 registration
functions, 13 statistics functions, 2 IO functions, image bridge — 53+ total Python-callable functions.

```
crates/ritk-python/
├── Cargo.toml            # cdylib "_ritk", pyo3 abi3-py39, numpy 0.22
├── pyproject.toml        # maturin, module-name = "ritk._ritk"
├── src/
│   ├── lib.rs            # #[pymodule] fn _ritk — registers 6 submodules
│   ├── image.rs          # PyImage(Arc<Image<NdArray,3>>), to_numpy(), shape/spacing/origin
│   ├── io.rs             # read_image / write_image (NIfTI, PNG, DICOM, MetaImage, NRRD)
│   ├── filter.rs         # 14 functions: gaussian, median, bilateral, n4_bias_correction,
│   │                     #   anisotropic_diffusion, gradient_magnitude, laplacian,
│   │                     #   frangi_vesselness, canny, laplacian_of_gaussian,
│   │                     #   recursive_gaussian, sobel_gradient, grayscale_erosion,
│   │                     #   grayscale_dilation
│   ├── registration.rs   # demons_register, diffeomorphic_demons_register,
│   │                     #   symmetric_demons_register, syn_register,
│   │                     #   bspline_ffd_register, multires_syn_register,
│   │                     #   bspline_syn_register, lddmm_register
│   ├── segmentation.rs   # 16 functions: otsu, li, yen, kapur, triangle, multi_otsu,
│   │                     #   connected_components, connected_threshold, kmeans, watershed,
│   │                     #   binary_erosion, binary_dilation, binary_opening, binary_closing,
│   │                     #   chan_vese, geodesic_active_contour
│   └── statistics.rs     # 13 functions: image_statistics, dice_coefficient,
│                         #   hausdorff_distance, mean_surface_distance, psnr, ssim,
│                         #   zscore_normalize, minmax_normalize, histogram_matching,
│                         #   nyul_udupa_normalize, white_stripe_normalize,
│                         #   estimate_noise, label_statistics (Sprint 7)
└── python/
    ├── ritk/__init__.py  # Imports from _ritk; surfaces ritk.Image at top level
    └── ritk/py.typed     # PEP 561 marker
```

**Remaining work before first usable wheel:**
- Run `maturin develop` end-to-end to validate Python import.
- Add `py.allow_threads` GIL release around long-running calls (filter, registration).
- Generate `.pyi` type stubs for IDE autocomplete.
- Add integration test comparing `ritk.io.read_image` output against SimpleITK reference values.

### 7.4 CLI Tooling Gaps · Severity: **Medium**

ANTs ships ~40 command-line executables (`antsRegistration`, `N4BiasFieldCorrection`,
`antsBrainExtraction.sh`, etc.). SimpleITK ships utility CLIs via `SimpleITK` Python module.
RITK has no CLI layer.

**Sprint 21:** `multires-demons` method added to `crates/ritk-cli/src/commands/register.rs` with `--levels` (usize, default 3) and `--use-diffeomorphic` (flag) args. 2 new CLI tests added.

**Planned location:**
```
crates/ritk-cli/
├── Cargo.toml
└── src/
    ├── main.rs
    ├── register.rs    # ritk register --fixed … --moving … --output …
    ├── segment.rs
    ├── filter.rs
    └── convert.rs     # format conversion
```

---

## 8. Implementation Priority Matrix

Scores: **C** = Critical (blocks standard workflows), **H** = High (significantly limits utility),
**M** = Medium (parity feature), **L** = Low (edge case / rarely used).

Effort estimates: **S** = ≤1 sprint (≤2 weeks), **M** = 2–4 sprints, **L** = 4+ sprints.

### 8.1 Registration

| Gap ID | Feature | Priority | Effort | Justification |
|---|---|---|---|---|
| GAP-R01 | SyN registration | **Closed** (Sprint 6) | L | Multi-res SyN + BSplineSyN + inverse consistency |
| GAP-R02 | Demons family | **Closed** (Sprint 3) | M | Thirion, Diffeomorphic, Symmetric Demons all implemented |
| GAP-R07 | BSpline FFD pipeline | **Closed** (Sprint 4) | M | BSplineFFDRegistration implemented |
| GAP-R03 | LDDMM | **Closed** (Sprint 6) | L | Geodesic shooting via EPDiff, Gaussian RKHS kernel |
| GAP-R04 | Groupwise/atlas | **Closed** (Sprint 7) | L | Iterative template building via Multi-Res SyN |
| GAP-R05 | Composite transform I/O | **Closed** (Sprint 6) | S | JSON serialization, TransformDescription enum, round-trip file I/O |
| GAP-R06 | Joint label fusion | **Closed** (Sprint 7) | M | Wang 2013 + majority voting |

### 8.2 Segmentation

| Gap ID | Feature | Priority | Effort | Justification |
|---|---|---|---|---|
| SEG-01 | Morphological operations | **Closed** (Sprint 2) | S | Binary erosion/dilation/opening/closing; grayscale variants |
| SEG-02 | Connected component labeling | **Closed** (Sprint 2) | S | Hoshen-Kopelman 6/26-connectivity, LabelStatistics |
| SEG-03 | Otsu / multi-Otsu thresholding | **Closed** (Sprint 2) | S | Otsu, multi-Otsu, Li, Yen, Kapur, Triangle |
| SEG-04 | Region growing | **Closed** (Sprint 2+10) | S | ConnectedThreshold, ConfidenceConnected, NeighborhoodConnected |
| SEG-05 | Image statistics API | **Closed** (Sprint 2) | S | compute_statistics, masked stats, Dice, Hausdorff, MSD |
| SEG-06 | Level set segmentation | **Closed** (Sprint 5) | M | Chan-Vese + Geodesic Active Contour implemented |
| SEG-07 | Watershed | **Closed** (Sprint 4) | S | Meyer flooding, 6-connectivity |
| SEG-08 | K-means clustering | **Closed** (Sprint 4) | S | k-means++ init, Lloyd iteration |
| SEG-DT | Euclidean distance transform | **Closed** (Sprint 7) | S | Meijster 2000, linear-time separable algorithm |

### 8.3 Filtering

| Gap ID | Feature | Priority | Effort | Justification |
|---|---|---|---|---|
| FLT-01 | N4 bias field correction | **Closed** (Sprint 3) | M | Tustison 2010 B-spline Tikhonov, multi-resolution |
| FLT-02 | Gradient magnitude / Sobel | **Closed** (Sprint 3 + Sprint 5 3D Sobel) | S | Required by level sets, Canny, Frangi |
| FLT-03 | Median filter | **Closed** (native `Image<B,D>` confirmed, Sprint 5) | S | Salt-and-pepper noise removal |
| FLT-04 | Recursive Gaussian (Deriche IIR) | **Closed** (Sprint 4) | S | Deriche IIR, derivative orders 0/1/2 |
| FLT-05 | Bilateral filter | **Closed** (native `Image<B,D>` confirmed, Sprint 5) | S | Edge-preserving denoising |
| FLT-06 | Frangi vesselness | **Closed** (Sprint 3+11) | M | Frangi 1998 multiscale + Sato 1998 line filter |
| FLT-07 | Anisotropic diffusion (Perona-Malik) | **Closed** (Sprint 3+11) | S | Perona-Malik + curvature anisotropic diffusion |
| FLT-08 | Canny edge detection | **Closed** (Sprint 4) | S | Gaussian + gradient + NMS + hysteresis |
| FLT-09 | Morphological filters (preprocessing) | **Closed** (Sprint 4) | S | GrayscaleErosion, GrayscaleDilation (flat cubic SE) |
| FLT-10 | Laplacian / LoG | **Closed** (Sprint 3+4) | S | LaplacianFilter, LaplacianOfGaussianFilter |

### 8.4 Statistics & Preprocessing

| Gap ID | Feature | Priority | Effort | Justification |
|---|---|---|---|---|
| STA-01 | Image statistics API | **Closed** (Sprint 2) | S | compute_statistics, masked stats, Dice, Hausdorff, MSD |
| STA-02 | Histogram matching | **Closed** (Sprint 2) | S | HistogramMatcher piecewise linear mapping |
| STA-03 | Z-score / min-max normalization | **Closed** (Sprint 2) | S | ZScoreNormalizer, MinMaxNormalizer |
| STA-04 | Nyul-Udupa normalization | **Closed** (Sprint 4) | S | Two-phase train/apply piecewise-linear standardization |
| STA-05 | Label statistics | **Closed** (Sprint 2) | S | LabelStatistics: count, centroid, bounding box per component |
| STA-06 | Noise estimation (MAD) | **Closed** (Sprint 4) | S | estimate_noise_mad, estimate_noise_mad_masked |
| STA-07 | Image comparison metrics (Dice, HD) | **Closed** (Sprint 2) | S | Dice, Hausdorff, mean surface distance |
| STA-08 | PSNR / SSIM | **Closed** (Sprint 4) | S | PSNR, SSIM Wang et al. 2004 |
| STA-09 | White stripe normalization | **Closed** (Sprint 7) | S | KDE-based WM peak detection (Shinohara 2014) |

### 8.5 IO

| Gap ID | Feature | Priority | Effort | Justification |
|---|---|---|---|---|
| IO-01 | MetaImage (.mha/.mhd) | **Closed** (Sprint 2) | S | Full round-trip, ZYX/XYZ permutation, external data file |
| IO-02 | NRRD | **Closed** (Sprint 2) | S | Full round-trip, space directions, space origin |
| IO-03 | TIFF / BigTIFF | **Closed** (Sprint 6) | M | Multi-page z-stack, u8/u16/u32/f32/f64 |
| IO-04 | MINC (.mnc2) | **Closed** (Sprint 13) | M | consus-hdf5 pure-Rust HDF5 parsing |
| IO-05 | MGZ / MGH | **Closed** (Sprint 7) | S | FreeSurfer format, gzip compression, 4 data types |
| IO-06 | VTK image | **Closed** (Sprint 8) | S | Legacy structured-points ASCII/BINARY |
| IO-07 | Analyze (.hdr/.img) | **Closed** (Sprint 2) | S | Full round-trip hdr/img pair |
| IO-08 | JPEG 2D | **Closed** (Sprint 8) | S | Grayscale read/write, [1,H,W] representation |

### 8.6 Python / CLI Bindings

| Gap ID | Feature | Priority | Effort | Justification |
|---|---|---|---|---|
| PY-01 | PyO3 Python module (`ritk-python`) | **C** | M | Categorical adoption blocker |
| PY-02 | NumPy array ↔ Image bridge | **C** | S | Required for DL pipeline integration |
| PY-03 | Python image I/O (`read_image`) | **C** | S | First function any user calls |
| PY-04 | Python filter API | **Closed** (Sprint 20) | S | 16 filter functions including curvature and sato |
| PY-05 | Python registration API | **Closed** (Sprint 6 — 8 registration functions exposed) | M | BSpline FFD, Multi-Res SyN, BSpline SyN, LDDMM added |
| PY-06 | Python segmentation API | **Closed** (Sprint 5 — all 16 segmentation algorithms exposed) | M | Full Python segmentation surface |
| PY-07 | CLI tooling (`ritk-cli`) | **M** | M | Shell-script pipeline integration |
| PY-08 | Type stubs / `py.typed` | **M** | S | IDE autocomplete, mypy compatibility |
| PY-STAT | Python statistics API | **Closed** (Sprint 7) | S | 13 functions: statistics, comparison, normalization, white stripe |

---

## 9. Architecture Plan for New Modules

All new modules follow RITK's confirmed conventions:
- DIP: trait in `mod.rs` of parent; concrete impl in child `*.rs` file.
- Files ≤ 400 lines; split by responsibility, not size alone.
- Naming: domain-relevant, no `utils.rs`, no `helpers/`.
- No API names encoding bounded variation dimensions (no `filter_f32`, `register_cpu`).

### 9.1 `ritk-core` Extensions

```
crates/ritk-core/src/
├── filter/
│   ├── mod.rs                       # FilterTrait + existing re-exports
│   ├── gaussian/
│   │   ├── mod.rs                   # GaussianVariant trait
│   │   ├── discrete.rs              # DiscreteGaussianFilter (existing: refactor in)
│   │   └── recursive.rs             # RecursiveGaussianFilter (Deriche IIR) — NEW
│   ├── bilateral.rs                 # BilateralFilter — NEW
│   ├── rank/
│   │   ├── mod.rs
│   │   └── median.rs                # MedianFilter — NEW
│   ├── edge/
│   │   ├── mod.rs
│   │   ├── gradient_magnitude.rs    # GradientMagnitudeFilter — NEW
│   │   ├── canny.rs                 # CannyEdgeDetectionFilter — NEW
│   │   └── laplacian.rs             # LaplacianFilter + LoG — NEW
│   ├── vesselness/
│   │   ├── mod.rs
│   │   ├── frangi.rs                # FrangiVesselnessFilter — NEW
│   │   ├── sato.rs                  # SatoLineFilter — NEW
│   │   └── hessian.rs               # DiscreteHessianFilter — NEW
│   ├── diffusion/
│   │   ├── mod.rs
│   │   ├── perona_malik.rs          # PeronaMalikDiffusionFilter — NEW
│   │   └── curvature_diffusion.rs   # CurvatureAnisotropicDiffusionFilter — NEW
│   ├── bias/
│   │   ├── mod.rs
│   │   ├── n4.rs                    # N4BiasFieldCorrectionFilter — NEW
│   │   └── bspline_bias.rs          # BSplineBiasSurface — NEW
│   └── morphology/                  # Preprocessing morphology — NEW
│       ├── mod.rs
│       ├── binary_erosion.rs
│       ├── binary_dilation.rs
│       ├── grayscale_erosion.rs
│       └── grayscale_dilation.rs
│
├── segmentation/                    # ENTIRE MODULE NEW
│   ├── mod.rs                       # Segmentation trait
│   ├── threshold/
│   │   ├── mod.rs
│   │   ├── otsu.rs
│   │   ├── multi_otsu.rs
│   │   ├── li.rs
│   │   ├── yen.rs
│   │   ├── kapur.rs
│   │   └── triangle.rs
│   ├── region_growing/
│   │   ├── mod.rs
│   │   ├── connected_threshold.rs
│   │   ├── neighborhood_connected.rs
│   │   └── confidence_connected.rs
│   ├── level_set/
│   │   ├── mod.rs                   # LevelSetEvolution trait
│   │   ├── geodesic_active_contour.rs
│   │   ├── shape_detection.rs
│   │   ├── chan_vese.rs
│   │   ├── laplacian.rs
│   │   └── sparse_field_solver.rs   # Narrow-band solver (Whitaker 1998)
│   ├── watershed/
│   │   ├── mod.rs
│   │   ├── immersion.rs
│   │   └── marker_controlled.rs
│   ├── clustering/
│   │   ├── mod.rs
│   │   └── kmeans.rs
│   ├── morphology/                  # Post-processing morphology
│   │   ├── mod.rs                   # MorphologicalOperation trait
│   │   ├── erosion.rs
│   │   ├── dilation.rs
│   │   ├── opening.rs
│   │   ├── closing.rs
│   │   ├── distance_transform.rs
│   │   └── skeletonization.rs
│   └── labeling/
│       ├── mod.rs
│       ├── connected_components.rs  # Hoshen-Kopelman union-find
│       └── label_statistics.rs
│
└── statistics/                      # ENTIRE MODULE NEW
    ├── mod.rs
    ├── image_statistics.rs          # Min, max, mean, variance, percentile
    ├── masked_statistics.rs
    ├── label_statistics.rs
    ├── noise_estimation.rs          # MAD estimator
    ├── image_comparison.rs          # Dice, Hausdorff, ASD, PSNR, SSIM
    └── normalization/
        ├── mod.rs                   # IntensityNormalization trait
        ├── zscore.rs
        ├── minmax.rs
        ├── histogram_matching.rs
        └── nyul_udupa.rs
```

### 9.2 `ritk-io` Extensions

```
crates/ritk-io/src/format/
├── mod.rs
├── dicom/           # Existing
├── nifti/           # Existing
├── png/             # Existing
├── metaimage/       # NEW — .mha / .mhd
│   ├── mod.rs
│   ├── reader.rs
│   └── writer.rs
├── nrrd/            # NEW
│   ├── mod.rs
│   ├── reader.rs
│   └── writer.rs
├── tiff/            # NEW — includes BigTIFF
│   ├── mod.rs
│   ├── reader.rs    # multi-page, multi-channel
│   └── writer.rs
├── minc/            # NEW — MNC2 (HDF5-based)
│   ├── mod.rs
│   ├── reader.rs
│   └── writer.rs
├── freesurfer/      # NEW — MGH / MGZ
│   ├── mod.rs
│   ├── reader.rs
│   └── writer.rs
├── vtk/             # NEW — legacy VTK + VTI
│   ├── mod.rs
│   ├── reader.rs
│   └── writer.rs
└── analyze/         # NEW — .hdr / .img (legacy)
    ├── mod.rs
    ├── reader.rs
    └── writer.rs
```

### 9.3 `ritk-registration` Extensions

```
crates/ritk-registration/src/
├── diffeomorphic/           # NEW — SyN + exponential map
│   ├── mod.rs               # DiffeomorphicRegistration trait
│   ├── syn/
│   │   ├── mod.rs
│   │   ├── velocity_field.rs
│   │   ├── exponential_map.rs
│   │   └── symmetric_energy.rs
│   └── bspline_syn/
│       ├── mod.rs
│       └── bspline_velocity.rs
├── demons/                  # NEW
│   ├── mod.rs               # DemonsRegistration trait
│   ├── thirion.rs
│   ├── diffeomorphic.rs
│   └── symmetric.rs
├── lddmm/                   # NEW
│   ├── mod.rs
│   ├── geodesic_shooting.rs
│   ├── epdiff.rs
│   └── rkhs_kernel.rs
├── atlas/                   # NEW
│   ├── mod.rs
│   ├── template_estimation.rs
│   ├── groupwise_energy.rs
│   └── frechet_mean.rs
└── label_fusion/            # NEW
    ├── mod.rs
    └── joint_label_fusion.rs
```

### 9.4 New Crate: `ritk-python`

```
crates/ritk-python/
├── Cargo.toml               # crate-type = ["cdylib"], pyo3 = { features = ["extension-module"] }
├── pyproject.toml           # [build-system] maturin; [project] name = "ritk"
├── src/
│   ├── lib.rs               # #[pymodule] fn ritk(_py: Python, m: &Bound<PyModule>)
│   ├── image.rs             # PyImage: Arc<Image<NdArray<f32>,3>>, NumPy bridge
│   ├── io.rs                # read_image(path) -> PyImage, write_image(img, path)
│   ├── filter.rs            # gaussian, median, bilateral, n4_bias_correction
│   ├── registration.rs      # register(fixed, moving, config) -> (image, transform)
│   └── segmentation.rs      # threshold, region_grow, morphology
└── python/
    ├── ritk/__init__.py
    ├── ritk/py.typed
    └── ritk/*.pyi            # generated type stubs (pyo3-stub-gen)
```

### 9.5 New Crate: `ritk-cli`

```
crates/ritk-cli/
├── Cargo.toml
└── src/
    ├── main.rs              # clap subcommand dispatch
    ├── register.rs          # ritk register --fixed F --moving M --metric mi --output O
    ├── segment.rs           # ritk segment --input I --method otsu --output O
    ├── filter.rs            # ritk filter --input I --gaussian-sigma 1.5 --output O
    └── convert.rs           # ritk convert --input I.nii.gz --output O.mha
```

---

## Appendix A — Reference Toolkit Feature Counts

Counts include 3D-capable, non-deprecated, non-legacy filter/algorithm implementations.

| Category | ITK ≈ | SimpleITK ≈ | ANTs ≈ | RITK (confirmed) |
|---|---|---|---|---|
| Registration algorithms | 25 | 15 | 12 | 8 |
| Segmentation algorithms | 45 | 30 | 5 | 10 |
| Preprocessing / denoising filters | 40 | 25 | 8 | 9 |
| Edge / feature filters | 20 | 12 | 2 | 5 |
| Morphological filters | 30 | 20 | 3 | 6 |
| Statistics operations | 25 | 18 | 5 | 10 |
| IO formats | 30+ | 30+ | 10 | 5 |
| Language bindings | C++, Python, Java, R, C# | Python, Java, R, C# | Python (ANTsPy) | Python (PyO3), CLI |

---

## Appendix B — Recommended Sprint Sequence

Based on dependency ordering and severity scores:

**Sprint 1 — Foundations (unblocks everything else):**
- STA-01: Image statistics API
- STA-03: Z-score / min-max normalization
- SEG-02: Connected component labeling
- FLT-03: Median filter
- FLT-04: Recursive Gaussian (derivative support required by level sets, Frangi)
- IO-01: MetaImage (.mha/.mhd) — benchmark data access

**Sprint 2 — Segmentation Core:**
- SEG-01: Morphological operations (erosion, dilation, opening, closing, distance transform)
- SEG-03: Otsu / multi-Otsu thresholding
- SEG-04: Region growing
- STA-05: Label statistics

**Sprint 3 — Critical Filtering:**
- FLT-01: N4 bias field correction (depends on BSplineTransform — already present)
- FLT-02: Gradient magnitude
- FLT-05: Bilateral filter
- FLT-07: Perona-Malik anisotropic diffusion
- STA-02: Histogram matching

**Sprint 4 — Advanced Segmentation + Vesselness:**
- SEG-06: Level set segmentation (depends on gradient magnitude)
- FLT-06: Frangi vesselness (depends on Hessian, recursive Gaussian)
- STA-07: Dice / Hausdorff segmentation metrics
- IO-02: NRRD

**Sprint 5 — Python Bindings (adoption enabler):**
- PY-01: `ritk-python` crate scaffold (PyO3 + maturin)
- PY-02: NumPy ↔ Image bridge
- PY-03: Python image I/O
- PY-04: Python filter API (surfaces Sprint 1–4 results)

**Sprint 6 — Deformable Registration:**
- GAP-R07: BSpline FFD pipeline
- GAP-R02: Demons (Thirion + diffeomorphic)
- GAP-R05: Composite transform I/O

**Sprint 7 — Atlas + Label Fusion + MGH + Distance Transform + White Stripe + Python Stats (COMPLETED):**
- GAP-R04: Groupwise/atlas registration (iterative template building via Multi-Res SyN)
- GAP-R06: Joint label fusion (Wang 2013 + majority voting)
- IO-MGH: MGZ/MGH reader/writer (FreeSurfer format, gzip compression)
- SEG-DT: Euclidean distance transform (Meijster 2000)
- STA-09: White stripe normalization (Shinohara 2014)
- PY-STAT: Python statistics API (13 functions)

**Sprint 8 — IO Expansion + CLI/Python Completion:**
- IO-06: VTK image format
- IO-08: JPEG 2D support
- PY-07: CLI tooling completion
- PY-08: Type stubs / `py.typed`

**Sprint 9+ — Remaining parity:**
- IO-05: MINC — **Closed** (Sprint 12, via `consus` pure-Rust HDF5)
- Remaining IO formats (Analyze)
- Remaining filters (curvature anisotropic diffusion, Sato line)
- GAP-R02b: Diffeomorphic Demons exact inverse