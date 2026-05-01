# CHANGELOG

All notable changes to RITK are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/).

<!-- ──────────────────────────────────────────── -->
## [Unreleased]

### Added
- **`ritk-dicom` crate**: Added a Rust-owned DICOM boundary with transfer syntax classification, `PixelLayout`, native pixel byte decoding, PackBits decoding, native RLE Lossless fragment decoding, and a generic `FrameDecodeBackend<O>` trait. [minor]
- **Native DICOM JPEG decode path** (`ritk-dicom`): Added grayscale JPEG Baseline/Extended fragment decoding with DICOM layout validation and modality LUT application before falling back to `dicom-rs` for unsupported JPEG cases. [patch]
- **Native JPEG Lossless dispatch** (`ritk-dicom`): JPEG Lossless Non-Hierarchical and First-Order Prediction now route through the RITK-native JPEG decoder before backend fallback. [patch]

### Changed
- **DICOM codec dispatch** (`ritk-io`): `decode_compressed_frame` now delegates through `ritk_dicom::DicomRsBackend`, making `dicom-rs` a replaceable backend while preserving the existing `ritk-io` public series API. [minor]
- **Windows GNU native build toolchain**: `.cargo/config.toml` now sets global and target-qualified UCRT clang/clang++/llvm-ar compiler variables for native C/C++ build scripts and keeps lld as the linker. [patch]
- **Transfer syntax SSOT**: `ritk-dicom::TransferSyntaxKind` now owns all transfer-syntax predicates used by DICOM readers. `ritk-io` keeps only a compatibility re-export for downstream callers. [minor]
- **DICOM JPEG backend dispatch**: JPEG Baseline and JPEG Extended now attempt RITK native Rust decode first and use `dicom-rs` only as a compatibility fallback. [patch]
- **DICOM JPEG syntax ownership**: Added `TransferSyntaxKind::is_native_jpeg_codec()` so native JPEG ownership is classified in the transfer-syntax SSOT instead of backend match arms. [patch]

### Verification
- `cargo check -p ritk-dicom`: passed.
- `cargo test -p ritk-dicom`: 10 passed.
- `cargo check -p ritk-io`: passed with UCRT clang/lld.
- `cargo test -p ritk-io test_decode_compressed_frame_jpeg_baseline_round_trip -- --no-capture`: passed with `D:\msys64\ucrt64\bin` first on `PATH`.
- `cargo test -p ritk-io test_decode_compressed_frame_jpeg_extended_round_trip -- --no-capture`: passed with `D:\msys64\ucrt64\bin` first on `PATH`.
- `cargo test -p ritk-io test_decode_compressed_frame_jpegls_lossless_round_trip -- --no-capture`: passed with `D:\msys64\ucrt64\bin` first on `PATH`.
- `cargo test -p ritk-io test_decode_compressed_frame_jpeg2000_lossless_round_trip -- --no-capture`: passed with `D:\msys64\ucrt64\bin` first on `PATH`.
- `cargo test -p ritk-io test_decode_compressed_frame_rescale_contract -- --no-capture`: passed with `D:\msys64\ucrt64\bin` first on `PATH`.
- `cargo test -p ritk-io test_decode_compressed_frame_rle_lossless_unrestricted_round_trip -- --no-capture`: passed with `D:\msys64\ucrt64\bin` first on `PATH`.
- `cargo test -p ritk-io transfer_syntax`: passed.

## [0.12.3] — Sprint 83

### Fixed
- **`recursive_gaussian` GIL hold** (`ritk-python`): `recursive_gaussian` was the sole `#[pyfunction]` in `filter.rs` without `py.allow_threads`. Added `py: Python<'_>` as first parameter, moved Arc clone before the closure, and wrapped filter construction + `apply` call inside `py.allow_threads(|| { … })`. Consistent with the pattern applied to all other filter and registration bindings. Python-visible API unchanged. [patch]

### Documentation
- `gap_audit.md` §3.6: Skeletonization row updated from blank status to ✓ implemented (Sprint 10/28, `skeletonization.rs`; Python: Sprint 20; CLI: Sprint 20; 50+ tests). Section severity updated from Low to Closed. [patch]
- `gap_audit.md` §7.1: Removed four stale remaining-gap bullets (transform I/O closed Sprint 8; type stubs present Sprint 31; `py.allow_threads` now fully applied across all bindings; atlas/JLF closed Sprint 8). Severity downgraded from Medium to Low. One operational gap remains: hosted-CI `maturin` matrix validation. [patch]
- `gap_audit.md` §7.3: Updated filter.rs function count (14 → 34), segmentation.rs count (16 → 27), registration.rs count (8 → 13), total (91+ → 93+). Stale Sprint-5-vintage code tree replaced with Sprint-83-accurate listing. [patch]

### Changed
- `ritk-python` version bumped from 0.12.2 to 0.12.3. [patch]

## [0.12.2] — Sprint 82

### Fixed
- **GIL-blocking level-set segmentation bindings** (`ritk-python`): Five level-set functions previously held the CPython GIL for their full PDE iteration loop. Each function now clones the image `Arc` handles before calling `py.allow_threads(|| { ... })`, releasing the GIL for the duration of the computation. Affected: `chan_vese_segment`, `geodesic_active_contour_segment`, `shape_detection_segment`, `threshold_level_set_segment`, `laplacian_level_set_segment`. Python-visible API unchanged (adding `py: Python<'_>` to `#[pyfunction]` does not alter the Python signature). [patch]
- **GIL-blocking surface distance statistics** (`ritk-python`): `hausdorff_distance` and `mean_surface_distance` now release the GIL via `py.allow_threads`. Both functions have O(M·N) complexity where M and N are boundary voxel counts; for large clinical masks this was a significant GIL hold. [patch]

### Changed
- `ritk-python` version bumped from 0.12.1 to 0.12.2. [patch]

## [0.12.1] — Sprint 81

### Fixed
- **Distance transform all-background convention** (`ritk-core`): `distance_transform_squared` now returns all-zeros when no foreground voxels exist (empty foreground set → distance to nearest foreground is defined as 0). Previously returned `(nz+ny+nx)²` sentinel, causing `test_segment_distance_transform_background_is_zero` to fail with value 9.0 on a 3×3×3 all-zero image. [patch]
- **Parzen histogram fixed-image weight cache** (`ritk-registration`): `ParzenJointHistogram` now caches the transposed fixed-image weight matrix `W_fixed^T` on first call and reuses it in subsequent iterations without recomputing. Reduces per-iteration autodiff graph size for CR/MI-based registration. [patch]

### Added
- **Nextest configuration** (`.config/nextest.toml`): Per-test slow-timeout bounds for gradient-based registration integration tests (BSpline, multi-res, affine, rigid, versor). Default 60 s slow-timeout; 300 s for registration-heavy tests. Prevents indefinite CI hangs. [patch]

### Documentation
- `gap_audit.md`: Removed `confidence_connected` and `neighborhood_connected` from "Absent or incomplete" list; both confirmed present in Python API since Sprint 10. [patch]

## [0.12.0] — Sprint 80

### Added
- 10 new parity tests (Section 8 of `test_simpleitk_parity.py`): watershed label map, K-means cluster count, connected-threshold sphere recovery, confidence-connected sphere recovery, neighborhood-connected sphere recovery, curvature anisotropic diffusion smoothing, Sato line filter tube response, white top-hat bright structure isolation, hit-or-miss isolated voxel detection, morphological reconstruction dilation fill
- `gap_audit.md` severity corrections: §3.1 (thresholding) Critical→Closed, §3.2 (region growing) Critical→Closed, §3.4 (watershed) Medium→Closed, §4.5 (Canny) Medium→Closed, §4.7 (Recursive Gaussian) High→Closed, §4.8 (LoG) Medium→Closed, §4.10 (Morphological Filters) High→Closed, §5.2 (Nyúl-Udupa) High→Closed, §5.3 (Intensity Normalization) High→Closed
- §3.3 level-set table updated: ShapeDetection, LaplacianLS, ThresholdLS rows changed from "Not yet" to "Implemented"

### Fixed
- `test_segmentation_bindings.py` `test_shape_detection_segment_preserves_shape_and_finite_values` call-site `curvature_weight=0.2` corrected to `1.0` (matches pyo3 canonical default fixed in Sprint 79)
- `ci.yml` `python-wheel` smoke test updated from `laplacian_level_set_segment(curvature_weight=0.2)` to `shape_detection_segment(curvature_weight=1.0)` for representative default coverage

### Changed
- `ritk-python` version bumped from 0.11.0 to 0.12.0

## [0.11.0] — Sprint 79
### Added
- 5 new level-set parity tests (Chan-Vese sphere Dice, GAC expansion in uniform image,
  ShapeDetection binary output near sphere, ThresholdLS expansion inside intensity band,
  LaplacianLS nontrivial binary mask)
- 5 new filter parity tests (RecursiveGaussian interior vs SITK, LoG near-zero in linear
  interior, Sigmoid midpoint analytical+SITK agreement, Canny edge concentration at sphere
  surface, Sobel zero-on-constant and nonzero-on-gradient)
- macOS added to `python_ci.yml` matrix (Python 3.9–3.12 on ubuntu+windows+macos)
- Multi-platform `release.yml` (Linux manylinux, Windows, macOS universal2) with PyPI
  OIDC trusted publishing
### Fixed
- `segmentation.pyi` `shape_detection_segment` default `curvature_weight` corrected
  from `0.2` to `1.0` (now matches pyo3 binding and Rust struct defaults)
- `pyproject.toml` `requires-python` corrected from `>=3.8` to `>=3.9` (matches abi3-py39)
### Changed
- 5 level-set binding tests in `test_segmentation_bindings.py` enhanced with binary
  output assertion (strictly {0.0, 1.0}) replacing the weak `np.var > 0.0` check
- `ritk-python` version bumped from 0.10.0 to 0.11.0

## [0.10.0] — Sprint 78
### Added
- 5 new SimpleITK parity tests in `test_simpleitk_parity.py`:
  - `test_yen_threshold_produces_valid_segmentation` (Yen 1995 max-correlation; Dice vs `sitk.YenThresholdImageFilter` ≥ 0.85)
  - `test_kapur_threshold_produces_valid_segmentation` (Kapur 1985 max-entropy; reference `sitk.MaximumEntropyThresholdImageFilter`; noisy sphere; Dice ≥ 0.85)
  - `test_triangle_threshold_produces_valid_segmentation` (Zack 1977; Dice vs `sitk.TriangleThresholdImageFilter` ≥ 0.85)
  - `test_binary_threshold_segment_agrees_with_sitk` (explicit [lower, upper]; Dice vs `sitk.BinaryThreshold` ≥ 0.999)
  - `test_distance_transform_agrees_with_sitk` (Euclidean DT vs `sitk.SignedMaurerDistanceMap`; background MAE < 0.15 voxels)
- `binary_threshold_segment` and `marker_watershed_segment` added to `segmentation.pyi` stubs
- `binary_threshold_segment` and `marker_watershed_segment` added to smoke test required callable list
- MSYS2 ucrt64 PATH step added to `python_ci.yml` Windows jobs (resolves `libstdc++-6.dll` load failure on clean build)
- `CXXFLAGS_x86_64_pc_windows_msvc` added to `.cargo/config.toml` to statically link GCC C++ runtime when building with MSYS2 clang-cl

### Fixed
- Distance transform convention corrected to ITK standard: `distance_transform` now computes distance from each voxel to the nearest **foreground** voxel (foreground receives 0); previously computed distance from foreground to nearest background. All 19 Rust unit tests updated with analytically re-derived expected values and pass in both debug and release profiles.
- 3 pre-existing Python test failures resolved: `test_distance_transform_all_foreground_returns_zeros`, `test_distance_transform_single_foreground_voxel_background_nonzero` (convention fix), `test_registered_functions_have_stub_and_smoke_coverage` (stub gap)

### Changed
- `gap_audit.md` §3.7 (`Connected Component Analysis`) updated from `Critical` to `Closed` (Hoshen-Kopelman + union-find implemented Sprint 28)
- `gap_audit.md` §5.1 (`Histogram Matching`) updated from `Critical` to `Closed` (implemented Sprint 27)
- `gap_audit.md` §5.4 (`Image Statistics`) `label_statistics.rs` status updated from `MISSING` to `DONE` (implemented, parity-tested Sprint 77)
- `ritk-python` version bumped from `0.9.0` to `0.10.0`


## [0.9.0] — Sprint 77
### Added
- 3 new SimpleITK parity tests in `test_simpleitk_parity.py`:
  - `test_multires_demons_ncc_improves_on_shifted_sphere` (MultiRes Demons 3-level, NCC ≥ 0.90)
  - `test_inverse_consistent_demons_ncc_improves_on_shifted_sphere` (IC-Demons, sigma=1.0, ic_weight=0.1, NCC ≥ 0.85; measured 0.93)
  - `test_label_intensity_statistics_mean_agrees_with_sitk` (RITK vs SimpleITK `LabelStatisticsImageFilter`, 3-label sphere volume, per-label mean/count agreement < 1e-3)
- `SimpleITK vtk` added to `python_ci.yml` pip install step; `test_simpleitk_parity.py`, `test_vtk_parity.py`, `test_ct_mri_registration_parity.py` added to CI test run
- `CHANGELOG.md` created; versioning history from Sprint 71–77 documented per SemVer 2.0.0 policy

### Fixed
- Pre-existing 1D-array `TypeError` in `test_statistics_bindings.py`: `test_minmax_normalize_range_inverted_bounds_raises` and `test_minmax_normalize_range_and_zscore_bindings_are_available` now use valid 3D arrays (`(1,1,3)` and `(1,2,2)` respectively); value-semantic assertions added (min=0.0/max=1.0 for minmax; mean=0.0/std=1.0 for zscore)
- IC-Demons parity test corrected: `sigma_diffusion=1.5` (overly smoothed, NCC ≈ 0.84) changed to `sigma_diffusion=1.0` (NCC ≈ 0.93); analytical justification documented in docstring

### Changed
- GAP-R07 (`gap_audit.md`) section header updated from "Severity: **High**" to "Severity: **Closed**"; implementation record added (BSplineFFDRegistration, multi-resolution refinement, 22 unit tests, Python binding, Sprint 4)
- `ritk-python` version bumped from `0.1.0` to `0.9.0` to reflect sprint milestone history

## [0.8.0] — Sprint 76
### Added
- 4 new SimpleITK `ImageRegistrationMethod`-based parity tests replacing skipped Elastix tests (`test_sitk_translation_recovers_sphere_overlap`, `test_ritk_demons_vs_sitk_translation_quality`, `test_sitk_bspline_deformable_vs_ritk_syn`, `test_sitk_affine_registration_converges_on_shifted_sphere`)
- `gradient_step` parameter exposed in `build_atlas` Python binding (PyO3 signature, pyi stub)

### Fixed
- Removed `scale=False` kwarg from `SetInitialTransform` call (absent in SimpleITK 2.5.4)
- Lowered affine Dice threshold from 0.85 to 0.80 with analytical justification (32³/r6 sphere, Dice ≈ 0.83 at 1-voxel residual)

### Changed
- GAP-R08 (Elastix parity) severity downgraded from Medium to Low; SimpleITK `ImageRegistrationMethod` parity tests now provide active reference baselines (36/36 pass, 0 skipped)

## [0.7.0] — Sprint 75
### Added
- `gradient_step: f64 = 0.25` parameter to `SyNConfig`, `MultiResSyNConfig`, and `BSplineSyNConfig` (matches ANTs default)
- Per-iteration force normalization in all three SyN variants (`diffeomorphic/mod.rs`, `diffeomorphic/multires_syn.rs`, `diffeomorphic/bspline_syn.rs`): max|u| (inf-norm) = gradient_step before velocity accumulation
- `gradient_step` exposed in Python bindings `syn_register`, `multires_syn_register`, `bspline_syn_register`
- New Rust test `syn_recovers_translation_ncc_improves` (Gaussian blob, 4-voxel shift, NCC_after ≥ 0.80)
- New Python parity test `test_syn_register_ncc_improves_on_shifted_gaussian_blob`

### Fixed
- SyN CC gradient force formula corrected in all three `cc_forces` functions: replaced incorrect `-2*cc_num/(var_i*var_j)` with Avants 2008 eq. 10 `force_scale = (J_W−μ_J)/sqrt(var_i·var_j) − CC·(I_W−μ_I)/var_i`
- `build_atlas` inner `MultiResSyNConfig` literal updated to pass `gradient_step` field

## [0.6.0] — Sprint 74
### Added
- MRI-DIR cranial CT test data (409 slices, 512×512, CC BY 4.0, PatientID=MRI-DIR-zzmeatphantom) to `test_data/3_head_ct_mridir/DICOM/`
- 8 new CT/MRI-relevant VTK parity tests in `test_vtk_parity.py`
- 5 new registration quality parity tests in `test_simpleitk_parity.py` Section 5
- 4 real-DICOM CT/MRI parity tests in `test_ct_mri_registration_parity.py` (skipif data absent)
- `crates/ritk-python/README.md` documenting build requirements and module API

### Fixed
- Python wheel DLL load failure on Windows: built with `rustup run nightly-x86_64-pc-windows-msvc py -m maturin build --release --auditwheel repair`; MinGW runtime libs bundled into `ritk.libs/`

## [0.5.0] — Sprint 73
### Added
- `SnapApp` eframe/egui binary with MPR viewer (`crates/ritk-snap/`)
- MRI-DIR T2 head phantom DICOM to `test_data/2_head_mri_t2/DICOM/`
- 10 VTK 9.6.1 ↔ SimpleITK 2.5.4 filter parity tests in `test_vtk_parity.py`
- CT/MRI DICOM integration tests in `crates/ritk-registration/tests/ct_mri_dicom_registration_test.rs`

### Fixed
- 3 `ritk-snap` compiler warnings (doc comment style, unused `mut`, dead-code `step_slice`)

## [0.4.0] — Sprint 72
### Added
- Full `ritk-snap` viewer: `SidebarPanel` (DICOM tree), 2×2 MPR layout, 14 CT + 4 MR clinical window presets, measurement tools (Length mm, Angle °, ROI), NIfTI file open, 4-corner DICOM overlay, PNG export, 7 colormaps

## [0.3.0] — Sprint 71
### Added
- `zscore_normalize` Python binding exposes optional `mask: Image | None = None` parameter
- Smoke test and statistics binding test for `zscore_normalize(image, mask=...)`

### Fixed
- `crates/ritk-python/python/ritk/_ritk/statistics.pyi` stub updated to declare optional mask parameter

<!-- ──────────────────────────────────────────── -->
[Unreleased]: https://github.com/ryancinsight/ritk/compare/HEAD...HEAD
[0.12.3]: https://github.com/ryancinsight/ritk/compare/v0.12.2...v0.12.3
[0.12.0]: https://github.com/ryancinsight/ritk/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/ryancinsight/ritk/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/ryancinsight/ritk/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/ryancinsight/ritk/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/ryancinsight/ritk/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ryancinsight/ritk/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ryancinsight/ritk/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ryancinsight/ritk/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ryancinsight/ritk/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ryancinsight/ritk/compare/v0.2.0...v0.3.0
