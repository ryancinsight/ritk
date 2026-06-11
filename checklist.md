# RITK Sprint Checklist — Active

## Sprint 362 — Architecture Hardening: SSOT · DRY · SRP · DIP · Naming
**Target version**: 0.59.0
ritk-core: 0.9.0 → 0.10.0 | ritk-registration: 0.53.0 → 0.54.0 | ritk-segmentation: 0.1.0 → 0.2.0 | ritk-io: 0.2.0 → 0.3.0

### Track A — Correctness
- [x] FIX-362-01 [patch]: `engine.rs` fake-generic f32 hardcode → `loss.clone().into_scalar().elem::<f64>()` (fake-generic HARD violation; panics on non-f32 backends)
- [x] PERF-362-22 [patch]: Restore Moirai default features so RITK workspace consumers use default parallel execution, Mnemosyne memory surfaces, and Mellinoe branding; verification pending.

### Track B — SSOT Unblock
- [x] SSOT-362-02 [minor]: `ritk-io::ImageFormat` enum + `from_path` resolver; replace CLI `infer_format` and Python `io/mod.rs` if-chains
- [x] DRY-362-03 [patch]: Remove `FftDir` compatibility shim in `filter/fft/convolution/helpers.rs`; update all call sites to `ForwardFft`/`InverseFft` ZSTs

### Track C — DRY/Core
- [ ] DRY-362-04 [minor]: `UnaryImageFilter<Op>` + `UnaryPixelOp` trait; collapse `abs/sqrt/exp/log/square` (5 files, ~570L → ~100L + type aliases); generalize `D=3` → `const D: usize`

### Track D — Registration
- [x] DRY-362-05 [patch]: `ConvergenceFlag` → `optimizer/regular_step_gd/convergence.rs`; re-exported through `regular_step_gd`, `optimizer::mod`; local private enums removed from `regular_step_gd/optimizer.rs` and `adaptive_stochastic_gd.rs`
- [x] DRY-362-06 [patch]: Complete `SamplingConfig` migration — replace `sampling_percentage: Option<f32>` in `MutualInformation` + `CorrelationRatio` + `compute_image/mod.rs`
- [x] DRY-362-07 [minor]: Rename `preprocessing::NormalizationMode` → `IntensityRescaleMode`; resolves name collision with `metric::NormalizationMode`
- [x] DRY-362-08 [patch]: `CacheSlot<T>` newtype + `MutualInformation` migration
- [x] SRP-362-09 [patch]: Split `bspline_ffd/basis.rs` (445L) → `basis/{scalar,cache,evaluate}.rs`
- [x] SRP-362-10 [patch]: Split `dl_registration_loss.rs` → `dl/losses/{lncc,grad,combined,mod}.rs`
- [x] SRP-362-11 [patch]: Extract `regularization/trait_::utils` → `regularization/spatial_ops.rs`; make `pub(crate)`
- [ ] PRIM-362-12 [minor]: `EarlyStoppingPolicy::Enabled { patience, min_improvement }` — bundle orphaned fields into enum variant
- [ ] DIP-362-13 [minor]: `Registration::with_config` DIP fix — `RegistrationCallbackSet` builder decouples engine from concrete callback types

### Track E — Segmentation
- [x] DRY-362-14 [minor]: `HistogramThreshold` sealed trait; blanket `compute<B,D>` + `apply<B,D>` for 6 threshold structs (~150L scaffold eliminated)
- [x] DRY-362-15 [patch]: `smooth_or_borrow(data, dims, sigma) -> Cow<[f64]>` in `level_set/helpers.rs`; collapse 3× repeated Cow conditional
- [x] PRIM-362-16 [patch]: `Connectivity { Six, TwentySix }` enum in `ConnectedComponentsFilter`; remove runtime `assert!`
- [x] SRP-362-17 [patch]: Extract `UnionFind` from `labeling/mod.rs` → `labeling/union_find.rs`

### Track F — IO
- [ ] SRP-362-18 [patch]: Split `dicom/seg/tests/convert.rs` (554L) → 4 test modules
- [ ] SRP-362-19 [patch]: Split `dicom/series.rs` → `series/{types,scan,loader}.rs`; replace `Arc<Mutex>` scan pattern with collect-and-merge

### Track G — CLI
- [ ] SRP-362-20 [major]: `FilterArgs` (46 fields) → `FilterKind` `ValueEnum` + `#[command(flatten)]` per-family structs; `SegmentArgs` same treatment
- [x] DRY-362-21 [patch]: `Backend` alias duplicated in `commands/mod.rs` + `commands/viewer.rs`; viewer uses `super::Backend`
- [x] DRY-362-22 [patch]: `scales: String`, `cpr_points: Vec<String>` deferred parsing → `value_delimiter` typed fields

### Track H — Naming Violations
- [ ] NAMING-362-23 [patch]: `transform_1d/_2d/_3d/_4d` in `bspline/interpolation/` → `transform_points_impl` dispatching on `D` — BLOCKED: duplicate method names on same type across impl blocks; requires [arch] refactor
- [ ] NAMING-362-24 [patch]: `spatial_gradient_2d/_3d`, `spatial_laplacian_2d/_3d` → move to `deformable_field_ops/`, surface only through `dispatch.rs`

### Track I — Primitives
- [ ] PRIM-362-25 [minor]: `IntensityRange { min, max }` validating newtype; adopt in `MinMaxNormalizer.target_{min,max}` and `ZScore` params
- [x] PRIM-362-26 [patch]: Add `// PRECISION:` justification comment in `normalize.rs` f64 accumulator path
- [ ] PRIM-362-27 [minor]: `DicomSeriesInfo` — replace `ArrayString<64>` public fields with `&str` accessor; keep `ArrayString` internal

### Track J — DIP/Arch
- [x] DIP-362-28 [patch]: `wgpu_compat` → `pub(crate)`; file `[arch]` `ExecutionPolicy::max_batch_size()` item
- [x] ARCH-362-29 — Filed [arch] backlog item: `Image<B,T,D>` scalar phantom `PhantomData<T>` — dtype safety, f32 hardcoded throughout; requires architectural migration

**Verification gate** (per Track A completion):
- [x] `cargo clippy -p ritk-registration --all-targets -- -D warnings` → 0 warnings
- [x] `cargo test -p ritk-registration --lib` → all green
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings (Sprint 362 round 2)
- [x] `cargo nextest run -p ritk-core --lib` → 365/365 passed
- [x] `cargo nextest run -p ritk-registration --lib` → 592/592 passed
- [x] `cargo nextest run -p ritk-segmentation --lib` → 375/375 passed
- [x] `cargo nextest run -p ritk-filter --lib` → 689/689 passed
- [x] `cargo nextest run -p ritk-cli` → 200/200 passed

---

## Sprint 361 — Phase 21 Cleanup & Optimization (20 Cycles)
**Target version**: 0.58.0  
ritk-core: 0.8.0 → 0.9.0 | ritk-registration: 0.52.0 → 0.53.0

- [x] CYC-01 [patch]: Fix `ops.rs::gaussian_kernel_1d` bug (1+σ² → 2σ²) + value-semantic FWHM test
- [x] CYC-02 [patch]: Delete 6 duplicate Gaussian kernel functions (n4/dft, frangi, pde wrapper, level_set/helpers, geodesic_active_contour, deconvolution legacy wrappers)
- [x] CYC-03 [patch]: Naming prohibition: `rebuild_image_3d`→`rebuild_image`, `refine_component_3d`→`refine_component`, `laplacian` alias deleted
- [x] CYC-04 [minor]: GaussianSigma in DemonsConfig.sigma_diffusion/fluid (Option<GaussianSigma>), GlobalMiConfig.smoothing_sigmas (Vec<Option<GaussianSigma>>), CmaMiLevelConfig.sigma_mm/coarse_sigma_mm
- [x] CYC-05 [patch]: RegularStepGdConfig derive Copy; `best_x.clone()` → mem::take; Range<i32> redundant clone; SamplingMode enum for use_sampling:bool
- [x] CYC-06 [minor]: VolumeDims for LabelMap.shape, ImageOverlay.dims, MaskOverlay.dims, N4Config.initial_control_points + ritk-io call sites
- [x] CYC-07 [minor]: AffineTransform internal propagation: classical/spatial/{transform,affine,rigid}.rs + global_mi/transforms.rs
- [x] CYC-08 [minor]: CliInverseConsistency enum in ritk-cli (21 bool stubs updated)
- [x] CYC-09 [minor]: CLI sigma validation: checked GaussianSigma construction with anyhow bail in mi.rs, lddmm.rs, smoothing.rs, spatial_impl.rs
- [x] CYC-10 [minor]: PySpacingMode enum replacing use_image_spacing:bool in ritk-python
- [x] CYC-11 [patch]: SRP: demons.rs 448L→152L + normalize.rs 456L→187L (tests extracted)
- [x] CYC-12 [patch]: Delete remaining Gaussian kernel duplicates: level_set/helpers.rs, geodesic_active_contour.rs
- [x] CYC-13 [patch]: Collapse generate_mask_2d_dispatch/3d to generate_mask_generic<D>
- [x] CYC-14 [patch]: Extract CmaMiResult to cma_mi/result.rs
- [x] CYC-15 [patch]: iterate_structure/mod.rs tests already extracted (prior sprint, confirmed)
- [x] CYC-16 [patch]: region_growing/mod.rs 414L → 23L; ConnectedThresholdFilter → connected_threshold.rs; tests → tests.rs
- [x] CYC-17 [patch]: ritk-python/filter/smooth.rs 417L → smooth/ directory (mod.rs, gaussian.rs, diffusion.rs, special.rs)
- [x] CYC-18 [minor]: VolumeDims in deformable_field_ops/* function params (6 files + 21 callers)
- [x] CYC-19 [patch]: Vec::with_capacity — no Vec::new() in hot paths (confirmed no-op)
- [x] CYC-20 [patch]: Full verification gate — clippy 0 warnings, all test suites green

**Verification gate**:
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo test -p ritk-core --lib` → 1647/0/1
- [x] `cargo test -p ritk-registration --lib` → 583/0/1
- [x] `cargo test -p ritk-codecs --lib` → 106/0/0
- [x] `cargo test -p ritk-nrrd --lib` → 23/0/0
- [x] `cargo test -p ritk-io --lib` → 327/0/0
- [x] ritk-core: 0.8.0 → 0.9.0; ritk-registration: 0.52.0 → 0.53.0

---

## Residual Items for Sprint 361

- [x] PRIM-360-01: `GaussianSigma` in `WhiteStripeResult.sigma` + all call sites [minor]
- [x] BOOL-360-02: `DicomAssociationState` for `Association.active: bool` [patch]
- [x] BOOL-360-03: `PixelSignedness` for `signed: bool` in ritk-codecs tests [patch]
- [x] BOOL-360-04: `DcmPresenceFlags` for 7 bools in `ClinicalDistributionSummary` [patch]
- [x] BOOL-360-05: `PyConductanceKind` for `exponential: bool` in Python anisotropic_diffusion [patch]
- [x] BOOL-360-06: `PyDistanceMetric` for `squared: bool` in Python distance_transform [patch]
- [x] BOOL-360-07: `PyVesselPolarity` for `bright_vessels/bright_tubes: bool` in Python vessel filters [patch]
- [x] BOOL-360-08: `PyCleaningPolicy` for `clean_pixel_data/clean_private_tags: bool` in Python anonymize [patch]
- [x] BOOL-360-09: `PyInverseConsistency` for `inverse_consistency: bool` in Python syn multires [patch]
- [x] BOOL-360-10: `PyInitStrategy` for `use_com_init: bool` in Python cma_es [patch]
- [x] SRP-360-11: Split `ritk-macros/src/lib.rs` (895L → ~200L + 3 submodules) [patch]
- [x] SRP-360-12: Split `ritk-python/src/segmentation/levelset.rs` (473L → 6 files) [patch]
- [x] SRP-360-13: Split `ritk-python/src/filter/fft.rs` (465L → 4 files) [patch]
- [x] PRIM-360-14: `VolumeDims` adoption in bspline_ffd function signatures [minor]
- [x] PRIM-360-15: `GaussianSigma` in `CannyEdgeDetector` public API [minor]
- [x] PRIM-360-16: `GaussianSigma` in `LaplacianOfGaussianFilter` public API [minor]
- [x] PRIM-360-17: `GaussianSigma` in `GaussianFilter` sigmas field [minor]
- [x] CAP-360-18: `Vec::with_capacity` in DICOM networking PDU codec (20+ sites) [patch]
- [x] CAP-360-19: `Vec::with_capacity` in remaining compute hot paths — no-op (all early-return guards) [patch]
- [x] VER-360-20: Verification gate passed

### Sprint 360 (×5 continuation) — this session

- [x] FIX-360-C01: `AffineTransform` migration across engine/global_mi/cma_mi call sites [patch]
- [x] FIX-360-C02: `VolumeDims` migration in basis.rs, ritk-python bspline_ffd [patch]
- [x] FIX-360-C03: `ritk-io` useless `.into()` on `RgbaU8` + unused import [patch]
- [x] FIX-360-C04: `tests_canny.rs` `GaussianSigma::new_unchecked` at 3 call sites [patch]
- [x] PRIM-360-C05: `UnsharpMaskFilter.sigmas: Vec<GaussianSigma>` + ritk-snap call sites [minor]
- [x] PRIM-360-C06: `LddmmConfig.kernel_sigma: GaussianSigma` + cli/python/registration call sites [minor]
- [x] PRIM-360-C07: `LNCC.kernel_sigma: GaussianSigma` + test call sites [minor]
- [x] PRIM-360-C08: `CedScratch.cached_sigma: Option<GaussianSigma>` sentinel [patch]
- [x] SRP-360-C09: `interpolation/dispatch.rs` 612L → 407L (tests extracted) [patch]
- [x] SRP-360-C10: `interpolation/kernel/linear/mod.rs` 552L → 134L (tests extracted) [patch]
- [x] SRP-360-C11: `filter/transform/pad.rs` 474L → 329L (tests extracted) [patch]
- [x] SRP-360-C12: `statistics/normalization/histogram_matching.rs` 462L → 183L (tests extracted) [patch]
- [x] SRP-360-C13: `metric/mutual_information` tests_mutual_information.rs [patch]
- [x] SRP-360-C14: `demons/multires.rs` tests extracted [patch]
- [x] SRP-360-C15: `filter/edge/separable_gradient/mod.rs` tests extracted [patch]
- [x] CLONE-360-C16: `BoolStructure::dilate` + `iterate_structure` consuming signatures [patch]
- [x] CLONE-360-C17: `clahe/interpolate.rs` scratch.output `mem::take` [patch]
- [x] CAP-360-C18: `presentation_contexts Vec::with_capacity(32)` [patch]
- [x] ARCH-360-C19: `VolumeDims` promoted to `ritk_core::spatial` (re-exported in ritk-registration) [minor]
- [x] VER-360-C20: Full verification gate — clippy 0, 1612/583/103/23 tests green

**Verification gate (×5 session)**:
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo test -p ritk-core --lib` → 1612/0/1
- [x] `cargo test -p ritk-registration --lib` → 583/0/1
- [x] `cargo test -p ritk-codecs --lib` → 103/0/0
- [x] `cargo test -p ritk-nrrd --lib` → 23/0/0
- [x] ritk-core: 0.7.0 → 0.8.0; ritk-registration: 0.51.0 → 0.52.0
- [x] CHANGELOG.md [0.57.0] section added

---

## Residual Items for Sprint 361

| ID | Description | Priority |
|----|-------------|----------|
| ARCH-361-01 | `LabelMap.shape: [usize; 3]` → `VolumeDims` (now that VolumeDims is in ritk-core) | Medium |
| ARCH-361-02 | `ImageOverlay.dims / MaskOverlay.dims: [usize; 3]` → `VolumeDims` | Medium |
| PRIM-361-03 | `GaussianSigma` in `DiscreteGaussianFilter` variance/sigma params | Low |
| PRIM-361-04 | `GaussianSigma` in `BilateralFilter::new(spatial_sigma, range_sigma)` | Low |
| SRP-361-05 | `filter/bias/n4.rs` (520L) — split remaining operation families | Low |
| SRP-361-06 | `filter/morphology/label_morphology.rs` (448L) — extract tests | Low |
| ARCH-361-07 | `Arc<Mutex<Option<T>>>` → typestate lifecycle in Parzen/LNCC/MI metric structs | [arch] |
| BOOL-361-04 | `inverse_consistency: bool` in CLI `register/mod.rs` — map to `InverseConsistency` enum | Low |
| BOOL-361-05 | `sigma_fixed: f64` / `kernel_sigma: f64` in CLI register args — adopt `GaussianSigma` | Low |
| SRP-361-06 | `compute_image.rs` (499L) — split cache helpers from main compute loop | Low |
| PRIM-361-07 | `GaussianSigma` adoption in `CoherenceConfig` scratch space sigma tracking | Low |
| UPSTREAM-359-03 | `masked_chunked.rs` + `fused.rs` clone-before-slice — blocked by Burn 0.19 lacking `slice_ref` | Blocked |
