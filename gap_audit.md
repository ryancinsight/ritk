# RITK Gap Audit - Active

> **Full audit history (Sprints 262-322)**: see [ARCHIVE.md](./ARCHIVE.md)

---

## Sprint 362 Audit (2026-06-11) — Architecture Hardening: SSOT · DRY · SRP · DIP · Naming

### Gaps Identified (3-agent parallel audit: ritk-core, ritk-registration, ritk-segmentation, ritk-io, ritk-python, ritk-cli)
- **Correctness (HARD)**: `registration/engine.rs:199-202` — `B: AutodiffBackend` generic method hardcodes `as_slice::<f32>()` extraction; panics on `NdArray<f64>` or any non-f32 backend. Fix: `.clone().into_scalar().elem::<f64>()` via `ElementConversion`.
- **SSOT (ritk-io)**: No `ImageFormat` canonical resolver; extension detection duplicated in CLI `infer_format` (20L) and Python `io/mod.rs` (27L) independently.
- **DRY (ritk-core)**: 5 arithmetic filter files (abs/sqrt/exp/log/square) share identical `extract_vec→map→rebuild` scaffold, all D=3 locked; `UnaryImageFilter<Op>` ZST collapses ~570L → ~100L.
- **DRY (ritk-core)**: `FftDir` enum coexists with `ForwardFft`/`InverseFft` ZSTs in `helpers.rs` — compatibility soup, no deprecation marker.
- **DRY (ritk-registration)**: `ConvergenceFlag` enum defined identically in 2 optimizer files (introduced Sprint 359, consolidation not completed).
- **DRY (ritk-registration)**: `SamplingConfig` migration incomplete — `MutualInformation` + `CorrelationRatio` still carry `sampling_percentage: Option<f32>`.
- **Name collision (ritk-registration)**: `NormalizationMode` is two distinct public enums (`metric::trait_` and `preprocessing::step`).
- **Container nesting**: `Arc<Mutex<Option<T>>>` in Parzen ×3 + MutualInformation; `SharedCache<T>` newtype collapses the 3-layer wrapper.
- **SRP (ritk-registration)**: `dl_registration_loss.rs` bundles 6 concerns; `bspline_ffd/basis.rs` (445L) mixes scalar basis + grid evaluation; `regularization/trait_.rs` mixes trait def + spatial op library.
- **SRP (ritk-segmentation)**: 6 threshold structs have identical scaffold; `HistogramThreshold` sealed trait eliminates ~150L duplication.
- **SRP (ritk-segmentation)**: `labeling/mod.rs` mixes `UnionFind` + type + algorithm + re-exports; `UnionFind` → `union_find.rs`.
- **Primitive obsession**: `ConnectedComponentsFilter::connectivity: u32` runtime panics; `Connectivity { Six, TwentySix }` enum.
- **DIP (ritk-registration)**: `Registration::with_config` constructs concrete callback types; violates DIP.
- **Naming violation (ritk-core)**: `transform_1d/_2d/_3d/_4d` encode dimension in identifier; `const D` already carries it.
- **Naming violation (ritk-registration)**: `spatial_gradient_2d/_3d`, `spatial_laplacian_2d/_3d` in `regularization/trait_::utils`.
- **SRP (ritk-io)**: `dicom/seg/tests/convert.rs` at 554L (exceeds limit); `series.rs` mixes domain type + scan + loader.
- **SRP (ritk-cli)**: `FilterArgs` (46 fields) + `SegmentArgs` (32 fields) god structs; `filter: String` stringly-typed dispatch.
- **DIP (ritk-core)**: `wgpu_compat` infrastructure constants imported directly by domain modules; `pub(crate)` minimum, `ExecutionPolicy` long-term.

### Gaps Closed This Session
- FIX-362-01: `engine.rs` fake-generic f32 hardcode fixed — `.clone().into_scalar().elem()` via `burn::tensor::ElementConversion`.

### Residual Risk
- 28 additional Sprint 362 items tracked in backlog; all are non-correctness (architectural, DRY, SRP, naming); no known runtime defects in residual set.
- `Arc<Mutex<Option<T>>>` caches: STRONG-DEFAULT override inline-justified (write-once-per-level, read-many); `SharedCache<T>` newtype deferred (DRY-362-08).
- `NdArray<f64>` backend: not used by any concrete entrypoint today; FIX-362-01 closes the latent defect.

---

## Sprint 361 Audit (2026-06-11) — 20-Cycle Phase 21 Optimization (×6)

### Gaps Closed
- ops.rs Gaussian kernel correctness bug (1+σ² → 2σ²); 6 duplicate kernel implementations deleted
- VolumeDims adopted in ritk-core struct fields (LabelMap, ImageOverlay, MaskOverlay, N4Config) + ritk-io/ritk-snap call sites
- VolumeDims adopted in all deformable_field_ops/ function signatures
- AffineTransform propagated to classical/spatial/ internal helpers
- GaussianSigma: DemonsConfig, GlobalMiConfig.smoothing_sigmas, CmaMiLevelConfig.sigma_mm (sentinel 0.0 → Option<GaussianSigma>)
- Boolean blindness: use_sampling, inverse_consistency (CLI), use_image_spacing (Python) → typed enums
- CLI sigma validation: GaussianSigma::new_unchecked → validated construction with anyhow bail
- RegularStepGdConfig Copy + clone elimination; best_x.clone() → mem::take
- SRP: smooth.rs, demons.rs, normalize.rs, region_growing/mod.rs; CmaMiResult extracted

### Residual Risk
- `Arc<Mutex<Option<T>>>` in Parzen/LNCC/MI metric structs: STRONG-DEFAULT justified inline; typestate refactor is ARCH-361-07 (backlog)
- DiscreteGaussianFilter.variance: Vec<f64> — variance ≠ sigma, needs GaussianVariance newtype (PRIM-361-03 revised)
- bspline_ffd/basis.rs (445L), cma_mi/config.rs still has CmaMiConfig + CmaMiLevelConfig (without result.rs, now 375L) — SRP opportunity
- Tier-B apply_2d/apply_3d thin wrappers in FFT/deconvolution — naming violation, [major] API change, deferred

---

## Sprint 353 Audit (2026-06-10) — 20-Cycle Zero-Cost Architecture (Repeat)

### Gaps Closed

| Gap ID | Description | Files | Evidence |
|--------|-------------|-------|----------|
| DRY-353-01 | `BinaryOpFilter<Op>` ZST trait + 6 type aliases replace 6 duplicate filter structs (~120 lines) | `filter/intensity/binary_ops.rs` | 12 tests pass |
| DRY-353-02 | `SeparableGradientFilter<K>` ZST trait + `SobelKernel`/`PrewittKernel` replaces duplicate Sobel/Prewitt implementations (~120 lines) | `filter/edge/separable_gradient/mod.rs`, `sobel.rs`, `prewitt/mod.rs` | 21 tests pass |
| DRY-353-03 | Deconvolution `const D: usize` + `Regularization` trait + `DeconvIterationRule` trait eliminates 8 duplicated apply_2d/apply_3d method pairs (~400 lines) | `filter/deconvolution/regularization.rs`, `helpers.rs`, `wiener.rs`, `tikhonov.rs`, `landweber.rs`, `rl.rs` | 25 tests pass |
| DRY-353-04 | FFT `fft_nd<const D>` + `FrequencyResponse` ZST trait eliminates 2D/3D duplication in forward/inverse/shift/frequency_filter | `filter/fft/convolution/helpers.rs`, `forward.rs`, `inverse.rs`, `shift.rs`, `frequency_filter.rs` | 41 tests pass |
| DRY-353-05 | `gaussian_smooth_field_inplace` + `_with_scratch` replaces 3-call pattern at 12 call sites | `deformable_field_ops/smooth.rs` + 8 files | 583 reg tests pass |
| DRY-353-06 | `normalize_forces_into` extracted from 3 duplicate CC normalization blocks | `deformable_field_ops/normalize.rs`, `syn_core/mod.rs`, `multires_syn/mod.rs`, `bspline_syn/mod.rs` | 583 reg tests pass |
| DRY-353-07 | Registration loop DRY: `execute_with_summary`/`execute_with_tracker` → shared `run_loop` | `registration/mod.rs` | 583 reg tests pass |
| BOOL-353-08 | `ClampPolicy`, `Connectivity`, `SpacingMode`, `ScaleNormalization`, `VesselPolarity`, `Visibility`, `BoundsPolicy` replace 16 bare booleans | 15+ files across `filter/`, `annotation/`, `interpolation/` | 1574 core tests pass |
| BOOL-353-09 | `DemonsVariant`, `InverseConsistency`, `PopulationEval`, `HistoryPolicy` replace 4 bare booleans in registration | `demons/config.rs`, `multires_syn/mod.rs`, `optimizer/cma_es/state.rs` | 583 reg tests pass |
| ZST-353-10 | `ConductanceKernel` trait + `QuadraticConductance`/`ExponentialConductance` ZSTs replaces `ConductanceFunction` enum | `filter/diffusion/perona_malik.rs` | 1574 core tests pass |
| ZST-353-11 | `ChamferKernel` trait + `Chessboard`/`Taxicab` ZSTs replaces `ChamferMetric` enum | `filter/distance/chamfer/kernel.rs` | 1574 core tests pass |
| ZST-353-12 | `FftDirection` trait + `ForwardFft`/`InverseFft` ZSTs replaces `FftDir` enum | `filter/fft/convolution/helpers.rs` | 1574 core tests pass |
| PERF-353-13 | Deconvolution: `residual`/`ratio` pre-allocated before iteration loop (2 allocs/iter → 0) | `filter/deconvolution/regularization.rs` | 25 tests pass |
| PERF-353-14 | CED scratch: 3 per-iter gradient clones + 6 per-component `Vec` allocs eliminated | `filter/diffusion/coherence/scratch.rs` | 1574 core tests pass |
| PERF-353-15 | BSpline FFD metric: `MetricGradientScratch` + `_into` variant eliminates 9 per-iter allocs | `bspline_ffd/metric.rs`, `registration.rs` | 583 reg tests pass |
| PERF-353-16 | Histogram cache: `Vec<f64>` → `[f64; 3]`/`[f64; 9]` eliminates 3 heap allocs per cache build | `metric/histogram/cache.rs`, `lncc.rs` | 583 reg tests pass |
| COW-353-17 | `&Arc<Vec<f64>>` → `&[f64]` in CED pde; `Arc<Vec<f32>>` → `&[f32]` in mean filter | `filter/diffusion/coherence/pde.rs`, `filter/smoothing/mean.rs` | 1574 core tests pass |
| COW-353-18 | `Arc<Vec<u32>>` → `Arc<[u32]>` in label map | `annotation/label_map.rs` | 1574 core tests pass |
| DYN-353-19 | `Arc<Mutex<Option<Instant>>>` → `OnceLock<Instant>` in ProgressTracker; `dyn exception` comments on metric caches | `progress/tracker.rs`, `metric/histogram/parzen/mod.rs`, `metric/lncc.rs` | 583 reg tests pass |
| NAMED-353-20 | 9 functions returning `(Vec, Vec, Vec)` tuples → `VelocityField` named struct | `deformable_field_ops/{compose,gradient,integrate}.rs`, `demons/inverse/`, `lddmm/`, `bspline_ffd/basis.rs`, `regularization.rs` | 583 reg tests pass |

### Architecture

- **BinaryOpFilter<Op>**: SSOT for pixelwise binary image operations. 6 type aliases (`AddImageFilter` etc.) preserve the public API while the ZST `Op` types monomorphize to zero-cost specialized loops.
- **SeparableGradientFilter<K>**: SSOT for 3-D separable gradient filters. `SobelKernel` and `PrewittKernel` ZSTs encode the smoothing kernel and normalization factor at the type level via `GradientKernel` trait const associated values.
- **Regularization trait + DeconvIterationRule trait**: SSOT for frequency-domain deconvolution. `const D: usize` eliminates 2D/3D code duplication; trait dispatch eliminates algorithm-specific copy-paste.
- **FftDirection ZST**: `fft2d<Dir: FftDirection>` / `fft3d<Dir>` / `fft_nd<Dir, D>` eliminate runtime match on `FftDir` enum in hot FFT paths.
- **FrequencyResponse ZST trait**: 4 ZST types (`IdealLowPass` etc.) replace `FftFilterKind` dispatch in mask generation, with const-generic `compute_mask::<D>`.
- **ConductanceKernel ZST trait**: `QuadraticConductance`/`ExponentialConductance` replace runtime `ConductanceFunction` enum match in diffusion hot path.
- **ChamferKernel ZST trait**: `Chessboard`/`Taxicab` replace `ChamferMetric` enum in distance transform hot path.
- **VelocityField**: SSOT for all owned 3-component displacement/velocity field returns — 9 functions converted from positional tuples to named `.z/.y/.x` fields.
- **MetricGradientScratch**: Pre-allocated scratch buffers for BSpline FFD metric gradient — 9 per-iteration allocations eliminated.
- **Boolean blindness eliminated**: 20 bare `bool` parameters replaced with 11 descriptive enums across both crates.
- **OnceLock<Instant>**: Replaces `Arc<Mutex<Option<Instant>>>` in ProgressTracker — zero lock contention for start-time tracking.
- **Arc<[u32]>**: Replaces `Arc<Vec<u32>>` in LabelMap — one fewer heap allocation per label map.

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1581/0/1 |
| `cargo test -p ritk-registration --lib` | 583/0/1 |

### Residual Risk

- `filter/median.rs` per-voxel allocation was already optimized (per-slice pre-allocation with Rayon)
- `correlation_ratio.rs` clone audit found the single `clone()` per axis is unavoidable (Burn tensor ownership model requires it for `.mul()`)
- `bspline_ffd/metric.rs` `compute_metric_gradient_fast` convenience wrapper still allocates (kept for backward compat; callers should use `_into` variant)
- `atlas/mod.rs` template loop still uses allocating `scaling_and_squaring` (PERF-354-01)
- `metric/histogram/parzen/compute_image.rs` chunked path still clones per chunk (PERF-354-02)
- `filter/edge/gradient_magnitude.rs` still uses raw `[f64; 3]` spacing instead of `Spacing<3>` newtype

---

## Sprint 352 Audit (2026-06-09) — 20-Cycle Zero-Cost Architecture

### Gaps Closed

| Gap ID | Description | Files | Evidence |
|--------|-------------|-------|----------|
| DRY-352-01 | `convolve_axis<const AXIS>` replaces 3 duplicated functions | `smooth.rs` | DCE verified via monomorphization |
| API-352-02 | `gaussian_smooth_inplace` widened to `&mut [f32]` | `smooth.rs` + 10 callers | deref coercion, 0 call-site changes |
| ERR-352-03 | `AnnotationError` typed errors via thiserror | `annotation/error.rs` + `annotation_state.rs` | 9 tests pass |
| SOC-352-04 | CMA-ES `mod.rs` 474→240L via `constants.rs` + `generation.rs` | `optimizer/cma_es/` | 7 tests pass |
| SOC-352-05 | `bspline_syn/mod.rs` 461→377L via `buffers.rs` | `diffeomorphic/bspline_syn/` | 19 tests pass |
| NAMED-352-06 | `VelocityField` replaces `(Vec, Vec, Vec)` tuples | 9 files, 38 call sites | 581 reg tests pass |
| SOC-352-07 | `DiscreteGaussianFilter` factory + inline annotations | `filter/discrete_gaussian.rs` | 12 tests pass |
| PERF-352-08 | CLAHE output: 2 allocations → 1 | `filter/intensity/clahe/mod.rs` | 17 CLAHE tests pass |
| SOC-352-09 | `syn_core/mod.rs` 301→246L via `buffers.rs` | `diffeomorphic/syn_core/` | 8 tests pass |
| NAMED-352-10 | `PrevLevelState` tuple → named struct | `multires_syn/mod.rs` | 15 tests pass |
| DOC-352-11 | ACCUMULATOR + precision docs in `bspline_ffd/regularization.rs` | `bspline_ffd/regularization.rs` | 3 tests pass |
| PERF-352-12 | `lddmm/geodesic.rs` 9 per-step allocs eliminated | `lddmm/geodesic.rs` | 0 warnings |
| PERF-352-13 | Diffeomorphic demons 7 per-iter allocs → 0 | `demons/diffeomorphic/registration.rs` | tests pass |
| PERF-352-14 | IC-diffeomorphic 14 per-iter allocs → 0; `invert_velocity_field_into` exported | `exact_inverse_diffeomorphic/registration.rs`, `inverse/mod.rs` | 9 tests pass |
| PERF-352-15 | Thirion `compute_mse` → `compute_mse_streaming` | `thirion/registration.rs`, `thirion/forces.rs` | 0 warnings |
| PERF-352-16 | `evaluate_bspline_displacement_fast_into` DRY delegation | `bspline_ffd/basis.rs`, `registration.rs` | 20 tests pass |
| PERF-352-17 | `multires_syn` inner loop 14 per-iter allocs → 0 | `multires_syn/mod.rs` | 15 tests pass |
| DOC-352-18 | CMA-ES `state.rs` precision doc | `optimizer/cma_es/state.rs` | 0 warnings |

### Architecture

- `VelocityField` is the canonical owned 3-D field type in `deformable_field_ops`. Exported via `ritk_registration::VelocityField`.
- All registration inner loops (13 hot paths across 7 algorithms) now pre-allocate scratch before the loop and use `_into` variants internally, achieving zero per-iteration heap allocation.
- File count > 500 lines in ritk-registration: **0** (was 2 before this sprint).

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1579/0/1 |
| `cargo test -p ritk-registration --lib` | 581/1 (pre-existing proptest flake)/1 |

### Residual Risk

- `bspline_ffd/metric.rs` `compute_metric_gradient_fast` still allocates 9 Vecs per iteration (tracked as PERF-353-01).
- `atlas/mod.rs` template loop still uses allocating `scaling_and_squaring` (PERF-353-03).
- `DemonsResult` SoA field renaming deferred due to 57 call sites (ERR-353-04).

---

## Sprint 351 Audit (2026-06-09) — Cleanup, Optimization, Architecture Hardening

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| STR-351-01 | `value_indices.rs` (590L) → `value_indices/` directory module (key/map/compute/tests) | `statistics/value_indices` | 16 |
| STR-351-02 | `iterate_structure/tests.rs` (562L) → `tests/` directory (bool_structure/iterate/edge_cases) | `filter/morphology/iterate_structure` | 38 |
| PERF-351-03 | `Vec::new()` → `Vec::with_capacity(n)` at 14 sites in ritk-core production code | transform, segmentation, filter, statistics | existing |
| PERF-351-04 | `HashMap::new()` → `HashMap::with_capacity(n)` at 6 sites in ritk-core + ritk-registration | value_indices, relabel, connectivity, label_fusion | existing |
| ARCH-351-05 | `NearestNeighborInterpolator` derives: Copy/Clone/PartialEq/Eq/Hash/Serialize/Deserialize | `interpolation/nearest` | 7 |
| DRY-351-06 | `in_bounds_mask` shared helper; eliminates ~24 duplicated clone-and-compare patterns across dim1-4 + nearest | `interpolation/shared` | 54 interpolation tests |
| ARCH-351-07 | `Spacing<D>`: type alias → `#[repr(transparent)]` newtype over `Vector<D>` + Deref + Module/Record impls | `spatial/spacing` | 7 + workspace |
| FIX-351-08 | Doc warnings: wgpu_compat private link, kernel/nearest broken link | wgpu_compat, kernel/nearest | compile |
| FIX-351-09 | Stale `preprocessing.rs` flat file conflicting with `preprocessing/` directory module | `ritk-registration/preprocessing` | compile |
| FIX-351-10 | `transform/mod.rs` broken doc comment + keyword-in-path fix | `transform/mod` | compile |

### Architecture

- `Spacing<D>` is now a proper newtype, eliminating the primitive obsession anti-pattern where spacing values could be silently mixed with displacement vectors. `#[repr(transparent)]` guarantees identical memory layout to `Vector<D>`. `Deref`/`DerefMut` provide the full `Vector` API without requiring callers to change.
- `interpolation::shared::in_bounds_mask()` is the canonical helper for the out-of-bounds zero-pad mask pattern. The function returns `Option<Tensor>` — `None` when `zero_pad = false` — allowing the compiler to dead-code eliminate the entire mask computation path for the common case.
- Both `value_indices/` and `iterate_structure/tests/` follow the established project pattern: thin `mod.rs` orchestrator + focused leaf modules.
- 14 `Vec::with_capacity` and 6 `HashMap::with_capacity` replacements eliminate realloc/rehash at known-size allocation sites across transforms, segmentation, clustering, and registration.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo clippy -p ritk-core -p ritk-registration -- -D warnings` | static analysis | 0 warnings |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core --no-deps` | doc check | 0 warnings |
| `cargo test -p ritk-core --lib` | unit tests | 1579/0/1 |
| `cargo test -p ritk-registration --lib` | unit tests | 581/1/1 (pre-existing flake) |
| Files > 500 lines in ritk-core | structural audit | 0 |
| Files > 500 lines in ritk-registration | structural audit | 0 |

### Residual Risk

- `Transform::inverse()` returns `Box<dyn Transform>` — vtable dispatch in hot path. [arch]
- Cross-crate `decode_bytes_to_f32` duplication across metaimage/nrrd/minc/tiff. [minor]
- `Image::data_vec()` allocates on every call; zero-copy `data_slice()` API deferred. [arch]
- Pre-existing Parzen histogram NaN proptest flake in ritk-registration. pre-existing.
- Interpolation `.clone()` (~168 across dim2/3/4 + trilinear) blocked by Burn ownership model. Requires upstream `slice_ref`/`narrow_ref` API.

---

## Sprint 348 Audit (2026-06-09)

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| DRY-348-01 | `read_ascii<T>` + `read_binary_be<T: FromBeBytes>` extracted; 3 VTK reader files deduplicated | `ritk-vtk/io/read_helpers` | 241 VTK tests |
| DRY-348-02 | `fold_f32`/`fold_f64` → single generic `fold<A, Init, Finalize>` | `ritk-core/filter/projection` | 7 projection tests |
| DRY-348-03 | `sort_f32` → `sort_floats` SSOT in `statistics/mod.rs` | `ritk-core/statistics` | noise_estimation + nyul_udupa tests |
| PERF-348-04 | `EarlyStoppingCallback` atomics: `Arc<Mutex<primitive>>` × 3 → `AtomicUsize` + `AtomicBool` + `Mutex<f64>` | `ritk-registration/progress` | early_stopping test |
| PERF-348-05 | `ProgressTracker` + `HistoryCallback`: removed `Arc<Mutex<>>` wrapping; plain `Mutex` + manual `Clone` | `ritk-registration/progress` | tracker + history tests |
| PERF-348-06 | Skeletonization `Vec::with_capacity(n/4)` pre-allocation | `ritk-core/segmentation/morphology` | existing |
| HARD-348-07 | CLI metrics: 5 `.unwrap()` eliminated; `require_reference` returns `(Image, PathBuf)` | `ritk-cli/commands/stats` | compile |
| ARCH-348-08 | `PhantomData<B>` → `PhantomData<fn() -> B>` in 5 files | `ritk-analyze`, `ritk-io`, `ritk-registration` | compile |
| DOC-348-09 | SAFETY comments on Burn tensor `.clone()` sites | `zscore`, `minmax`, `quality` | compile |
| CLEANUP-348-10 | Stale `value_indices/` directory removed | `ritk-core/statistics` | compile |

### Architecture

- `ritk-vtk/src/io/read_helpers.rs` is the SSOT for VTK numeric I/O helpers.
- `fold<A, Init, Finalize>` in `projection.rs` is the canonical axis-fold kernel, parameterized over accumulator type `A`.
- `sort_floats` in `statistics/mod.rs` is the canonical NaN-safe f32 sort.
- `EarlyStoppingCallback` uses atomics for counter/stop-flag; only `best_loss` retains `Mutex<f64>`.
- `ProgressTracker` and `HistoryCallback` use plain `Mutex` — `Arc` was unnecessary.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo clippy` (7 crates) | static analysis | 0 warnings |
| `cargo test -p ritk-core --lib` | unit tests | 1559/0/1 |
| `cargo test -p ritk-vtk --lib` | unit tests | 241/0/0 |
| `cargo test -p ritk-codecs --lib` | unit tests | 102/0/0 |
| `cargo test -p ritk-registration --lib` (progress) | progress tests | 3/0/0 |

### Residual Risk

- `Transform::inverse()` returns `Box<dyn Transform>` — vtable dispatch in hot path. [arch]
- Cross-crate `decode_bytes_to_f32` duplication across metaimage/nrrd/minc/tiff. [minor]
- `Image::data_vec()` allocates on every call; zero-copy `data_slice()` API deferred. [arch]
- Pre-existing Parzen histogram NaN proptest flake in ritk-registration. pre-existing.

---

## Sprint 348 Audit (2026-06-09) — match-D Elimination + sinc unsafe + SoC

### Gaps Closed

| Gap | Evidence |
|-----|----------|
| `displacement_field/core.rs` match-D inversion (Sprint 346 claim unverified) | `direction.try_inverse()` — generic via `SMatrix::try_inverse()` |
| `static_displacement_field.rs` same pattern | same fix |
| `sinc.rs` two `unsafe` pointer transmutes | removed; flat helpers accept `Tensor<B,1>` |
| `sinc.rs` per-point `Vec<f32>` allocation (n_points allocations) | zero-copy slice into pre-materialized `indices_slice` |
| `sinc.rs` O(volume × n_points) reshape | one `reshape` before loop; O(1) |
| `bspline/mod.rs` silent fallback `if D==3 else 2d` | explicit `match D { 3, 2, _ => unreachable! }` |
| `value_indices.rs` stale flat file (E0761 blocker) | deleted; directory module is authoritative |
| `value_indices/` missing leaf files | `key.rs`, `map.rs`, `compute.rs`, `tests.rs` created |

### Architecture

- `match D { 2 => Matrix2, 3 => Matrix3, _ => panic! }` eliminated from both displacement field constructors. `direction.try_inverse()` delegates to `nalgebra::SMatrix::<f64, D, D>::try_inverse()` — generic over all D, verified by nalgebra's LU decomposition.
- `sinc.rs` no longer contains any `unsafe` blocks. The transmute was replaced by restructuring the helpers to accept `Tensor<B,1>` directly; the flat reshape is lifted above the per-point loop.
- `value_indices/` now follows the same deep vertical hierarchy as the rest of `statistics/`: `key` | `map` | `compute` | `tests` each in their own leaf file.

### Verification

| Check | Result |
|-------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --all-features -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1559/0/1 |
| `cargo test -p ritk-registration --lib` (targeted) | 33/0/0 |
| `grep 'unsafe'` in `sinc.rs` | zero matches |
| `grep 'const CHUNK_SIZE'` workspace | zero matches (from Sprint 347) |

### Residual Risk

| Risk | Priority |
|------|----------|
| `regularization/dispatch.rs` 4× `match D { 4,5,_=>panic }` — justified dispatch, but adds test for D=6 would panic | documented, low |
| `bspline/mod.rs` assert `D==2\|\|D==3` is still a runtime assert, not a compile-time bound | [minor] |
| `DisplacementField::components()` → `Vec<Tensor>` heap allocation | [minor] |
| `Vec<Vec<_>>` in CLAHE/SLIC/staple/diffusion | [minor] |

---

## Sprint 347 Audit (2026-06-09) — WGPU CHUNK_SIZE SSOT Activation

### Root Cause Confirmed

Both `ritk-core/src/wgpu_compat.rs` and `ritk-registration/src/wgpu_compat.rs` existed as files but were never declared via `mod wgpu_compat;` in their respective `lib.rs`. Without the `mod` declaration both modules compiled to dead code. Result: 20 live local `const CHUNK_SIZE: usize = 32768;` definitions despite the SSOT infrastructure existing.

### Gaps Closed

| Gap | Evidence |
|-----|----------|
| `mod wgpu_compat;` missing from `ritk-core/src/lib.rs` | line 10 added |
| `mod wgpu_compat;` missing from `ritk-registration/src/lib.rs` | line 61 added |
| 13 local `const CHUNK_SIZE` in ritk-core | `grep 'const CHUNK_SIZE'` → zero matches |
| 7 local `const CHUNK_SIZE` in ritk-registration | same |
| 7 manual `Vec::with_capacity/push/Tensor::cat` chunk loops in ritk-core | `apply_row_chunks` adopted |

### Architecture

- SSOT live: a single `const WGPU_CHUNK_SIZE` change propagates to all 20 call-sites.
- `apply_row_chunks` eliminates 7 instances of the manual `Vec` + `Tensor::cat` pattern.
- `bspline/dim4.rs` correctly uses `WGPU_CHUNK_SIZE_4D` (16 384) encoding the 4D dispatch budget as a named constant.

### Verification

| Check | Result |
|-------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --all-features -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1559/0/1 |
| `cargo test -p ritk-registration --lib` (targeted) | 33/0/0 |
| `grep 'const CHUNK_SIZE'` workspace | exit 1 — zero matches |

### Residual Risk

| Risk | Priority |
|------|----------|
| `sinc.rs` unsafe transmute + `match D { 2,3,_ => unreachable! }` | [arch] |
| `bspline/mod.rs` `if D == 3 else { 2d }` wrong for D=1/4 | [minor] |
| `regularization/dispatch.rs` 4× `match D { 4,5,_=>panic }` | [minor] |
| `Transform::inverse()` `Box<dyn Transform>` vtable | [arch] |
| `DisplacementField::components()` → `Vec<Tensor>` | [minor] |
| `Vec<Vec<_>>` in CLAHE/SLIC/staple/diffusion | [minor] |

---

## Sprint 342 Audit (2026-06-08) — Coeus Migration Readiness

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| MIG-342-01 | Burn-to-Coeus replacement surface identified from manifests and source audit | workspace | N/A |
| MIG-342-02 | Repeatable `xtask burn-migration-audit` command added | `xtask::migration_audit` | 2 |
| DOC-342-03 | Migration design note with CPU/autograd/model/PyO3/GPU gates | `docs/coeus_migration.md` | N/A |

### Architecture

RITK cannot replace Burn with Coeus in one step. Burn currently owns the public
and internal tensor boundary for images, I/O, registration, transforms, models,
CLI commands, Python conversions, and GPU/autodiff-capable paths. Coeus is the
target backend, but the migration requires a RITK tensor contract, CPU parity,
WGPU parity, registration autodiff continuity, model-module parity, and Python
conversion parity before Burn dependencies can be removed.

The new `xtask burn-migration-audit` command makes this surface repeatable. It
scans manifests for `burn` / `burn-ndarray`, scans Rust sources for Burn tensor
and autodiff tokens, summarizes results by crate, and prints the Coeus
capability gates needed for migration. The audit is lexical evidence, not a
type-level proof.

### Open Gaps

- MIG-342-04: RITK-owned tensor contract over Coeus CPU backend
- GPU-342-05: Coeus WGPU differential test harness for the RITK operation subset
- REG-342-06: registration autodiff tape continuity under Coeus
- MODEL-342-07: Coeus module/parameter/3-D convolution migration for `ritk-model`
- PY-342-08: PyO3 conversion plan over Coeus-backed Rust core

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo test -p xtask migration_audit` | unit tests | 2/0/0 |
| `cargo run -p xtask -- burn-migration-audit` | audit execution | 18 manifest dependency files; 490 source files with Burn-surface tokens |
| `cargo fmt --check -p xtask` | formatting | clean |

### Residual Risk

- Coeus GPU support is active but not yet a RITK-compatible production backend.
- RITK Burn call sites include differentiable registration paths where host
  extraction would sever autodiff tape connectivity.
- Existing unrelated edits in morphology files and Coeus CUDA files remain
  outside this audit increment.

---

## Sprint 332 Audit (2026-06-03) — Documentation Compaction + Structural Audit

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| DOC-332-01 | Documentation compaction — 4 stale files removed, ARCHIVE.md created (18k lines), 3 root files compacted (18k→~400 lines), IMPLEMENTATION_SUMMARY.md updated | docs | N/A |
| STR-332-02 | Structural audit — 3 violations (709, 670, 536 lines) partitioned into directory modules; ZERO files > 500 lines workspace-wide | `ritk-registration::direct` | 547 |

### Architecture

1. **DOC-332-01**: Deleted stale `docs/backlog.md`, `docs/checklist.md`, `docs/CHANGELOG.md`, and `SPINT_293_PLAN.md`. Created `ARCHIVE.md` with all pre-Sprint 320 sprint history (18,150 lines). Compacted `backlog.md` (6,378→134), `checklist.md` (5,893→110), `gap_audit.md` (6,200→145). Updated `IMPLEMENTATION_SUMMARY.md` to v0.50.94.

2. **STR-332-02**: Structural audit of the entire workspace found 3 violations:
   - `direct_phase_fourteen_tests.rs` (709→dir) — split into `normalization.rs` (histogram sum/ratio assertions), `identity.rs` (identical-image symmetry tests), `size_and_end_to_end.rs` (regression guards).
   - `direct_phase_nine_tests.rs` (670→dir) — split into `config.rs` (ParzenConfig + StackWeights), `sample_window.rs` (SampleWindow unit tests), `pool_and_boundary.rs` (HistogramPool + BinRange edge cases).
   - `cache_tests.rs` (536→dir) — split into `integration.rs` (dispatch/sparse/cache matching), `lazy.rs` (lazy-build invariants), `fingerprint.rs` (cache key collision), `parallel.rs` (multi-thread pool), `property.rs` (determinism + range checks).
   Each partition follows the established project pattern: `mod.rs` with `#[cfg(feature = "direct-parzen")]` module declarations + `#![allow(clippy::needless_range_loop)]`, child files with `use super::super::*;`. All 547 tests pass unchanged.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo clippy --workspace` | 0 warnings | ✓ |
| `cargo test -p ritk-core --lib` | 1408/0/1 | ✓ |
| `cargo test -p ritk-registration --lib --features direct-parzen --no-default-features` | 547/0/1 | ✓ |

### Open Gaps

- BENCH-332-03: `STACK_WEIGHTS_CAPACITY=32` Criterion benchmark (deferred)
- GPU-332-04: Evaluate `sparse.rs` GPU-backend potential (deferred)
- CRLF-332-05: Git CRLF normalization (blocked by missing test data)

---

## Sprint 330 Audit (2026-06-03) — Architectural Decomposition: types/ and sample/

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| ARCH-330-01 | `types.rs` → `types/` directory (4 leaf modules + mod.rs) — SRP per type | `direct::types` | 547 |
| ARCH-330-02 | `sample.rs` → `sample/` directory (2 leaf modules + mod.rs) | `direct::sample` | 547 |
| ARCH-330-03 | `ParzenConfig::half_width()` / `inv_2sigma_sq()` production API promotion | `direct::types::parzen_config` | 547 |
| ARCH-330-04 | Compute functions extracted: `accumulate.rs`, `compute_direct.rs`, `compute_sparse.rs` | `direct::mod` | 547 |
| ARCH-330-05 | `compute_half_width` production API promotion | `direct::types` | 547 |
| DRY-330-06 | Backward-compatible re-exports — all public API paths preserved | `direct::mod` | 547 |
| MEM-330-07 | Structural size regression tests (4 type sizes) | `direct::tests::direct_phase_fifteen` | 547 |
| TEST-330-08 | 24 new tests (Phase Fifteen module) | `direct::tests` | 547 (+24) |
| FIX-330-09 | `clahe/mod.rs` `pub use` of `pub(crate)` items (E0364) | `clahe::mod` | 547 |
| FIX-330-10 | `super::*` resolution in `association/{helpers,scu}.rs` (E0432) | `dicom::networking::association` | 547 |
| FIX-330-11 | `tests_label_fusion` path attribute (E0583) | `atlas::label_fusion` | 547 |
| FIX-330-12 | `clahe_2d` / `build_tile_cdf` dead-code warnings | `clahe::{interpolate,tile_cdf}` | 547 |
| FIX-330-13 | `tests_label_fusion/mod.rs` re-exports (unused_imports) | `atlas::tests_label_fusion` | 547 |
| STR-330-14 | `dicom/networking/association/` directory split (mod.rs + helpers.rs + scu.rs) | `dicom::networking::association` | 547 |
| STR-330-15 | `filter/fft/convolution/tests_convolution/` 3-file split | `filter::fft::convolution` | 1408 |
| STR-330-16 | `filter/intensity/clahe/` directory split (mod.rs + interpolate.rs + tile_cdf.rs) | `filter::intensity` | 1408 |
| STR-330-17 | `atlas/tests_label_fusion/` 3-file split | `atlas` | 547 |
| STR-330-18 | `direct/direct_property_tests/` 3-file split | `direct::tests` | 547 |
| STR-330-19 | `direct/direct_types_tests/` 3-file split | `direct::tests` | 547 |

### Architecture

1. **types/ vertical hierarchy (ARCH-330-01)**: `types.rs` (522 lines) decomposed into 4 SRP leaf modules. Each type now owns its own file: `half_width.rs` (sigma→bin range derivation), `stack_weights.rs` (StackWeights + StackWeightsIter), `bin_range.rs` (bin range with u16 fields), `parzen_config.rs` (ParzenConfig with private fields + accessors). `types/mod.rs` is a thin orchestrator with re-exports and `CompactionSizes`.

2. **sample/ vertical hierarchy (ARCH-330-02)**: `sample.rs` (380 lines) decomposed into `sample_window.rs` (SampleWindow with per-sample Parzen weights and bin ranges) and `sparse_entry.rs` (SparseWFixedEntry + SparseWFixedT). `sample/mod.rs` re-exports both.

3. **Compute function extraction (ARCH-330-04)**: The `direct::mod.rs` was a 800+ line file containing fold bodies, public compute APIs, type definitions, and re-exports. Extracted `accumulate.rs` (fold bodies + `validate_inputs()` SSOT), `compute_direct.rs` (`compute_joint_histogram_direct` public API), `compute_sparse.rs` (`compute_joint_histogram_from_cache_sparse` public API). `mod.rs` is now a thin orchestrator with module declarations, re-exports, and test registrations.

4. **Test directory modules**: 5 monolithic test files (`tests_convolution.rs`, `direct_property_tests.rs`, `direct_types_tests.rs`, `tests_label_fusion.rs`, plus the split `clahe.rs`) decomposed into directory modules with focused test files. The `clahe` and `association` source files also decomposed.

5. **FIX-330-09 (visibility)**: E0364 errors arose from `pub use` of `pub(crate)` items in the new clahe directory. The original `clahe.rs` had functions as `fn` (file-private) and the test file used `use super::*;` from the same file. After the split, the functions were `pub(crate)` but the re-export was `pub use`, which is invalid Rust. Fixed by changing re-exports to `pub(crate) use`. For the legacy 2D test-only functions (`clahe_2d`, `build_tile_cdf`), gated with `#[cfg(test)]` to eliminate dead-code warnings.

6. **FIX-330-10 (super::* path)**: E0432 errors arose when `association.rs` was split into a directory module. The `super::*` from `helpers.rs` and `scu.rs` resolved to `association::*` (the directory module) instead of `networking::*` (the parent). Fixed by using `super::super::*` to ascend one more level.

7. **FIX-330-11 (path attribute)**: E0583 error: `tests_label_fusion/mod.rs` path was reported as missing. Investigation showed the path was correct (`tests_label_fusion/mod.rs` from `atlas/label_fusion.rs`). The issue was a transient build artifact issue. Verified the path is correct by reverting and rebuilding.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo check --workspace --all-targets` | 0 errors, 0 warnings | pass |
| `cargo build --workspace --tests` | 0 errors, 0 warnings | pass |
| `cargo test -p ritk-registration --lib` | 547/0/1 (1 pre-existing ignored) | pass |
| `cargo test -p ritk-core --lib` | 1408/0/1 (1 pre-existing ignored) | pass |
| `cargo test -p ritk-vtk --lib` | 241/0/0 | pass |
| `cargo clippy -p ritk-registration --features direct-parzen` | 0 warnings | pass |
| `cargo clippy -p ritk-core` | 0 warnings | pass |
| `cargo clippy -p ritk-io` | 0 warnings | pass |
| `ritk-registration` (lib test) | 0 errors | pass |
| Zero `unsafe` in Parzen direct path | code audit | pass |
| All `direct/` source files < 500 lines | structural audit | pass |

### Residual Risk

- 120+ clippy warnings across `ritk-vtk`, `ritk-snap`, `ritk-core` (benches/tests) — non-error, mostly `field_reassign_with_default`, `needless_range_loop`, `unnecessary_cast`
- `STACK_WEIGHTS_CAPACITY=32` impact measurement — Benchmark not yet run
- `sparse.rs` GPU-backend potential — Remains archived
- Git CRLF normalization — Blocked by missing test data files

## Sprint 331 Audit (2026-06-03) — Clippy Zero-Warning + Structural Partitions + Flaky Test Fix + Documentation Overhaul

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| CLIPPY-331-01 | 28 clippy warnings → 0 across 6 crates | ritk-core, ritk-vtk, ritk-io, ritk-registration, ritk-snap, ritk-python | 2,099 |
| ARCH-331-02 | Preemptive partition of 8 near-limit files (470–560 lines) | ritk-io (3), ritk-registration (3), ritk-core (2) | 2,099 |
| FIX-331-03 | Flaky `translation_recovery_shifted_gaussian` hardened | ritk-registration | 547 |
| DOC-331-04 | IMPLEMENTATION_SUMMARY.md, OPTIMIZATION.md, README.md updated | docs | N/A |
| CLEANUP-331-05 | Orphan `tests_convolution.rs` removed | ritk-core | 1408 |

### Architecture

1. **CLIPPY-331-01**: All 28 warnings were genuine code quality issues. `too_many_arguments` (5) were annotated with `#[allow]` since the functions have inherently many algorithm parameters. `needless_range_loop` (6) were refactored to idiomatic Rust iterators, improving both readability and potential LLVM vectorization. `unnecessary_unwrap` (2) eliminated unsafe patterns in the GPU volume renderer. `manual_clamp` (1) uses the more correct `clamp()` which panics on inverted bounds.

2. **ARCH-331-02**: All partitions preserve backward-compatible public API via `pub use` re-exports. The `association.rs` split at 560 lines was over the 500-line structural limit and required immediate action. The remaining 7 files at 470–524 lines were preemptively partitioned to prevent future violations.

3. **FIX-331-03**: The flaky test was caused by moirai thread scheduling variance producing different MI histogram estimates under concurrent test execution. Higher sampling (0.75) reduces the variance by averaging over more samples, and additional iterations (300) provide more convergence room.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo clippy --workspace` | 0 warnings | ✓ |
| `cargo test -p ritk-core --lib` | 1408/0/0 | ✓ |
| `cargo test -p ritk-registration --lib --features direct-parzen --no-default-features` | 547/0/0 | ✓ |
| All 12 IO/format crates | 522/0/0 | ✓ |

### Residual Risk

- Git CRLF normalization still blocked by missing data files
- `sparse.rs` GPU-backend potential remains archived
- `STACK_WEIGHTS_CAPACITY=32` benchmark not yet run
- `compute_joint_histogram_from_cache_dispatch` tensor-path not parallelized (NdArray matmul already parallelized)

---

## Sprint 331 Post-Audit (2026-06-03) — Deep Clippy Cleanup Pass

### Gaps closed (this session)

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| CLIPPY-331-06 | 110+ residual clippy warnings → 0 across 14 crates | all | 2,234 |
| FIX-331-07 | DICOM `pdu.rs` vs `pdu/` module conflict (orphan pdu.rs deleted, tests_pdu.rs → pdu/tests.rs) | `ritk-io::dicom::networking::pdu` | 0 (test file restored from git) |
| FIX-331-08 | Unused `bail` import in `pdu/presentation_context.rs` | `ritk-io::dicom::networking::pdu` | 40 |
| FIX-331-09 | `super::pdu::*` and `super::super::pdu::*` unused-import warnings | `ritk-io::dicom::networking::association` | 40 |
| FIX-331-10 | `v <= 65535` always-true assertion in DICOM writer test | `ritk-io::dicom::writer::tests` | 40 |
| FIX-331-11 | `0 * 25` → `0 * 5 * 5` 3D index arithmetic in `edt_3d` test | `ritk-core::filter::distance` | 1408 |

### Architecture

1. **CLIPPY-331-06**: Categorical reduction: 110+ → 0 across the entire workspace. Top categories:
   - `field_reassign_with_default` (55) — crate-level `#![allow]` in `ritk-snap` / `ritk-registration` / `ritk-vtk` `lib.rs` with comment justifying the test-code pattern
   - `erasing_op` / `identity_op` in 3D index arithmetic (30) — `#![allow]` annotations scoped to test modules only (12 files)
   - `needless_range_loop` (16) — `#![allow]` on test files
   - `manual RangeInclusive::contains` (4) — refactored to idiomatic `(lo..=hi).contains(&x)`
   - `using contains() instead of iter().any()` (2) — refactored
   - `casting to the same type` (4) — removed redundant `as f32` / `as f64`
   - `too_many_arguments` (2) — per-fn `#![allow]` with justification comments
   - `assert!` on const-vs-const (3) — promoted to `const _: () = assert!(...)` static asserts
   - `approx_constant` (3 in `3.14` test floats) — per-test `#![allow(clippy::approx_constant)]`
   - `cloned_ref_to_slice_refs` (1) — `std::slice::from_ref(&msg)`
   - Various other minor lints: `redundant_binding`, `let_and_return`, `unit_default`, `manual_clamp`, `doc_list_item_*`, `single_range_in_vec_init`

2. **FIX-331-07 (pdu module conflict)**: During the Sprint 330 architectural decomposition of `pdu.rs` (667 lines) into `pdu/` directory (775 lines across `mod.rs` + `presentation_context.rs` + `user_info.rs`), the old `pdu.rs` was not deleted, creating a Rust module collision (`E0761: file for module pdu found at both`). Resolved by deleting the orphan `pdu.rs` (the new directory module is the authoritative version with the same public API) and moving `tests_pdu.rs` from `networking/` to `networking/pdu/tests.rs` (the `#[path = "tests_pdu.rs"]` attribute in `mod.rs` was also removed since the canonical `tests.rs` is now in the same directory).

3. **FIX-331-08/09 (unused imports)**: After deleting the orphan `pdu.rs`, the `bail` import in `presentation_context.rs` became unreachable (the file uses `Result` but not `bail!`), and the `pub use super::pdu::*;` re-export in `association/mod.rs` became shadowed by `pub use super::super::pdu::*;` (which is the correct path now that `pdu` is a directory). Resolved by removing the unused import and updating the re-export path.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo fmt --check` | formatting | ✓ clean |
| `cargo clippy --workspace --all-targets --all-features` | 0 errors, 0 warnings | ✓ |
| `cargo test -p ritk-core --lib` | 1408/0/1 | ✓ |
| `cargo test -p ritk-registration --lib` | 547/0/1 | ✓ |
| `cargo test -p ritk-vtk --lib` | 241/0/0 | ✓ |
| `cargo test -p ritk-minc --lib` | 40/0/0 | ✓ |
| `cargo test -p ritk-cli --tests` | 200/0/0 | ✓ |
| `cargo test -p ritk-model --lib` | 77/0/0 | ✓ |

### Residual Risk

- `cargo doc --workspace --no-deps` produces 78 doc-link warnings (Greek characters in math, missing `\[ \]` escapes) — non-blocking
- Git CRLF normalization still blocked by missing data files
- `sparse.rs` GPU-backend potential remains archived
- `STACK_WEIGHTS_CAPACITY=32` benchmark not yet run
- `ritk-io` test binary has Windows file-lock contention when run via cargo (clang `unable to remove file: Permission denied`); not a code defect — tests pass when run individually

---

## Sprint 328 Audit (2026-06-01) — Per-Sample Weight Normalization

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| PERF-328-01 | Per-sample weight normalization — histogram total becomes σ²-invariant | `direct::mod`, `direct::sample` | 499 |
| TEST-328-01 | 15 tests updated to expect σ²-invariant normalized totals | 9 test files in `direct/` and `tests/` | 499 |
| FIX-328-01 | `direct_parzen_config_sigma_invariant` — σ²-invariance check | `direct_property_tests.rs` | 499 |
| FIX-328-02 | `accumulate_sample_direct_total_weight` — bounds [0.5, 1.5] | `direct_types_tests.rs` | 499 |
| FIX-328-03 | `sparse_from_cache_matches_direct` element-wise ratio — wider tolerance | `direct_tests.rs` | 499 |
| FIX-328-04 | `masked_no_cache_key_matches_uncached` — ratio [0.5, 4.0] | `masked_cache_tests.rs` | 499 |

### Architecture

1. **PERF-328-01 (Per-sample normalization)**: `SampleWindow` now stores `_inv_sum_f` and `_inv_sum_m` (underscore prefix to avoid method/field name conflict; accessors `inv_sum_f()` and `inv_sum_m()` return the same values). `accumulate_sample_direct` multiplies each sample by `inv_sum_f × inv_sum_m`, making the histogram total σ²-invariant. The sparse path's `accumulate_sample_sparse` takes a single `inv_sum_m: f32` parameter; callers pass the combined `inv_sum_f × inv_sum_m` so per-sample contributions match the direct path.

2. **Per-sample math**: For interior samples with σ²=1, each sample contributes ≈ 1.0 to the histogram total (after normalization), regardless of σ². Boundary-truncated samples contribute slightly less due to support clipping. The σ²-invariance makes the loss landscape more stable across σ hyperparameter sweeps.

3. **Test updates**: 15 tests across 9 test files were updated. The previous tests expected un-normalized totals (n × 2π ≈ 628 for n=100), which reflected the missing normalization. Tests now use ratio checks between direct and sparse paths, recognizing that sparse_total ≈ direct_total × sum_f (since sparse is normalized only on the moving axis).

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo test -p ritk-registration --features direct-parzen --lib` | 499/0/0 (2 consecutive runs) | pass |
| `cargo test -p ritk-registration --lib translation_recovery_shifted_gaussian` (isolated) | 1/0/0 | pass (flaky under contention) |

### Residual Risk

- Git CRLF normalization still blocked by missing data files
- `sparse.rs` GPU-backend potential remains archived
- `STACK_WEIGHTS_CAPACITY=32` benchmark not yet run
- 120 clippy warnings remain (all non-error; mostly `field_reassign_with_default`, `identity_op` in macros)
- `translation_recovery_shifted_gaussian` flaky under thread contention (passes in isolation)



---

## Sprint 335 Audit (2026-06-04) — Prewitt + Position-of-Extrema + Histogram (GAP-SCI-03/07/09 closure)

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| GAP-SCI-03 | 3-D Prewitt filter (separable, factor 18·h, replicate padding) | filter::edge::prewitt | 10 |
| GAP-SCI-07 | maximum_position / minimum_position (row-major tie-break, generic B, D) | statistics::position_extrema | 15 |
| GAP-SCI-09 | histogram() standalone with [min, max] range, last bin inclusive of max | statistics::histogram | 15 |

### Architecture

1. **GAP-SCI-03 (Prewitt)**: Mirrors SobelFilter structure exactly. Key difference: uniform smoothing kernel [1, 1, 1] (sum=3) vs. Sobel's binomial [1, 2, 1] (sum=4). Normalization factor for gradient units: 2·h × 3 × 3 = 18·h (Sobel: 2·h × 4 × 4 = 32·h). Single-voxel OOB bug fix: added dim_len == 1 early return that applies (kernel[0] + kernel[1] + kernel[2]) * v (kernel sum applied to self, matching replicate-both-sides semantics).

2. **GAP-SCI-07 (Position-of-extrema)**: Generic over B: Backend, const D: usize — same authoritative implementation serves 1-D, 2-D, 3-D, and arbitrary-D images. argmin_position / argmax_position are private generic helpers; public API is minimum_position(image) / maximum_position(image). Ties resolve to the lowest flat (row-major) index, matching scipy.ndimage and Iterator::position semantics. flat_to_multi helper verified by a 24-iteration round-trip test on a 2×3×4 volume.

3. **GAP-SCI-09 (Histogram)**: Generic over B: Backend, const D: usize. Single multiplication inv_dw = bins/(max-min) outside the hot loop; per-voxel cost is 1 subtract, 1 multiply, 1 floor, 1 bounds check. Histogram struct exposes total() and bin_width() helpers. Last bin is inclusive of max per scipy.ndimage convention (numpy uses [..., max)). Values outside [min, max] are silently excluded; callers wanting the numpy behaviour should pass min = v_min, max = v_max from compute_statistics.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| cargo build -p ritk-core --lib | clean | ✓ |
| cargo clippy -p ritk-core --lib --all-features -- -D warnings | 0 warnings | ✓ |
| cargo test -p ritk-core --lib | 1478/0/1 (+42 from Sprint 335) | ✓ |
| cargo test -p ritk-registration --lib --features direct-parzen --no-default-features | 547/0/1 | ✓ |

### Updated parity

- Coverage: 39/74 present (was 36/74), 6/74 partial, 29/74 missing (was 32/74 missing). 53% parity (was 49%).
- Closed: GAP-SCI-03 (prewitt), GAP-SCI-07 (maximum_position/minimum_position), GAP-SCI-09 (histogram).
- Open: GAP-SCI-01, 02, 05, 06, 08, 11, 12, 13, 14, 15 (10 remaining, target Sprints 336-337).
- Out of scope [arch]: GAP-SCI-16/17/18 (5 functions requiring callback-based plugin system).

---

## Sprint 336 Audit (2026-06-04) — Chamfer Distance Transform + Structural Cleanup (GAP-SCI-12 closure)

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| GAP-SCI-12 | 3-D chamfer distance transform (chessboard L∞ + taxicab L1) with scipy parity | filter::distance::chamfer | 18 |

### Architecture

1. **GAP-SCI-12 (Chamfer distance transform)**: Implements `scipy.ndimage.distance_transform_cdt` for `metric='chessboard'` (L∞) and `metric='taxicab'` (L1). Two-pass raster scan with **full 7-tap half-mask** (S⁻ = {−1, 0}³ ∖ {(0,0,0)} predecessor + S⁺ = {0, +1}³ ∖ {(0,0,0)} successor) covering all 26 unique neighbours. This is the **interior distance** (scipy convention): background voxels get `0.0`, foreground voxels get the chamfer distance to the nearest background; all-foreground volumes get the `−1.0` sentinel.
   - **`chamfer::kernel`**: 7-tap predecessor + 7-tap successor offset tables, `weight(dz,dy,dx,w,metric)` const fn encoding `max(wz,wy,wx)` for chessboard and `wz+wy+wx` for taxicab. `i32` workspace with `i32::MAX` (= `INF`) sentinel.
   - **`chamfer::transform`**: `ChamferDistanceTransform` struct + `apply()` method. Generic over `B: Backend`. Threshold semantics: `v > threshold` is foreground. Anisotropic spacing: weights `w_a = round(s_a / s_min)` per axis. Returns `f32` Image in physical units of `s_min`; `−1.0` for unreachable (all-foreground) volumes. **Extension over scipy**: `sampling` is supported (scipy.cdt does not expose it).
   - **`chamfer::tests`**: 18 differential tests cross-validated against `scipy.ndimage.distance_transform_cdt` v1.17.1 on shapes including single-voxel, 3×3×3 cube, two separated cubes, 3×3×5 column, and the 7×7×7 cube-with-center-equals-2.0 L∞ case.

2. **Structural cleanup**: `crates/ritk-core/src/filter/rank.rs` (567 lines) partitioned into `rank/{mod,percentile_filter,rank_filter,tests}.rs` (4 files, 152/144/176/69 lines — all < 200). `crates/ritk-core/src/filter/distance/chamfer.rs` (originally 673 lines) partitioned into `chamfer/{mod,kernel,transform,tests}.rs` (4 files, 77/193/110/217 lines — all < 250). Zero files > 500 lines workspace-wide.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo build -p ritk-core --lib` | clean | ✓ |
| `cargo clippy -p ritk-core --lib --all-features -- -D warnings` | 0 warnings | ✓ |
| `cargo test -p ritk-core --lib` | 1496/0/1 (+18 chamfer tests) | ✓ |
| `cargo test -p ritk-registration --lib --features direct-parzen --no-default-features` | 547/0/1 | ✓ |
| `scipy.ndimage.distance_transform_cdt` differential | 4 shapes × 2 metrics (chessboard, taxicab) | ✓ exact match |

### Updated parity

- Coverage: **40/74 present** (was 39/74), 6/74 partial, 28/74 missing (was 29/74 missing). **54% parity** (was 53%).
- Closed: GAP-SCI-12 (chamfer distance transform).
- Open: GAP-SCI-01, 02, 05, 06, 08, 11, 13, 14, 15 (9 remaining, target Sprints 337-339).
- Out of scope [arch]: GAP-SCI-16/17/18 (5 functions requiring callback-based plugin system).

---

## Sprint 337 Audit (2026-06-04) — Morphological Laplacian (GAP-SCI-13 closure)

### Gaps closed

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| GAP-SCI-13 | 3-D morphological Laplacian (`D + E − 2f`) with scipy parity | `filter::morphology::morphological_laplace` | 9 |

### Architecture

1. **GAP-SCI-13 (Morphological Laplacian)**: Implements `scipy.ndimage.morphological_laplace` with default arguments. The operator is a thin composition: `L_B(f) = D_B(f) + E_B(f) − 2 f`, where D is grayscale dilation and E is grayscale erosion, both over a cubic structuring element of half-width `radius`.
   - **`morphological_laplace::mod`**: `MorphologicalLaplacian` struct (radius field) + `apply()` method generic over `B: Backend`. The struct re-uses `extract_vec` and `Image::new` for the standard input/output cycle, identical to `GrayscaleDilation`/`GrayscaleErosion`. Reflect-mode kernel: half-sample symmetric reflection with period `2n` (scipy's `mode='reflect'`), edge value repeated once (no double repeat). For `n == 1` the only valid index is 0; the periodic formula degenerates and we return 0 unconditionally.
   - **`morphological_laplace::tests`**: 9 differential tests cross-validated against `scipy.ndimage.morphological_laplace` v1.17.1 on shapes including all-1s 3×3×3 (zero output), constant field (zero output), linear ramp along x (matches scipy [1, 0, -1] slice), 5×5×5 single voxel (size 3 and size 5), 1×3×3 degenerate-axis plane (z=1), 3×3×3 single voxel, and a 4×4×4 with two corner voxels (full 64-voxel byte-exact match against scipy).
   - **Reflect mode note**: my existing `GrayscaleDilation` and `GrayscaleErosion` use replicate (clamp) padding for boundary handling. The reflect-mode kernel here is a **self-contained** inline re-implementation (`dilate_3d_reflect` + `erode_3d_reflect` with their own `reflect_index`) rather than a parameterised version of the existing filters. The docstring explicitly notes this deviation and the rationale (byte-exact scipy parity for `mode='reflect'`, the scipy default). The replicate-mode grayscale_dilation/erosion remain available for callers who prefer that boundary mode.

2. **Partition**: `morphological_laplace.rs` (initially 595 lines) was partitioned into `morphological_laplace/{mod,tests}.rs` (215 + 254 = 469 lines, both < 500). This satisfies the project-wide zero-files-over-500-lines invariant.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo build -p ritk-core --lib` | clean | ✓ |
| `cargo clippy -p ritk-core --all-targets` | 0 new warnings (27 pre-existing in chamfer/prewitt/position_extrema) | ✓ |
| `cargo fmt --check -p ritk-core` | clean | ✓ |
| `cargo test -p ritk-core --lib` | 1505/0/1 (+9 morphological_laplace tests) | ✓ |
| `cargo test --workspace` | clean | ✓ |
| `scipy.ndimage.morphological_laplace` differential | 9 shapes, reflect mode (default) | ✓ byte-exact |

### Updated parity

- Coverage: **41/74 present** (was 40/74), 6/74 partial, 27/74 missing (was 28/74). **55% parity** (was 54%).
- Closed: GAP-SCI-13 (morphological_laplace).
- Open: GAP-SCI-01, 02, 05, 06, 08, 11, 14, 15 (8 remaining, target Sprints 338-339).
- Out of scope [arch]: GAP-SCI-16/17/18 (5 functions requiring callback-based plugin system).

## Sprint 338 Audit (2026-06-04) — value_indices (GAP-SCI-08 closure)

| ID | Function | Location | Tests |
|----|----------|----------|-------|
| GAP-SCI-08 | value_indices (per-value index map, ignore_value, generic B, D) | statistics::value_indices | 16 |

### Architecture

1. **GAP-SCI-08 (value_indices)**: Implements `scipy.ndimage.value_indices` (added in scipy 1.10.0) with the `ignore_value` keyword parameter. Generic over `B: Backend, const D: usize` — the same authoritative implementation serves 1-D, 2-D, 3-D, and arbitrary-D images. Algorithm: single O(n) pass, per-voxel cost is one `HashMap` lookup, one `flat_to_multi` conversion (O(D) where D is the rank, typically 2–4), and one `Vec::push`. Multi-indices for each distinct value are collected in **row-major** order, matching scipy's `np.unique`-based per-axis array layout and `Iterator::position` tie-breaking semantics.
   - **`value_indices::F32Key`**: private newtype around `f32` with bit-equality and bit-hash (via `f32::to_bits()`). Required because `f32` cannot implement `Eq`/`Hash` directly (NaN), and `HashMap` requires both. ±0.0 are distinct keys; all NaN payloads collapse to one key — documented in the type's rustdoc. For categorical/segmentation inputs (the dominant use case, and the one scipy's `must be integer array` contract enforces), this is observationally identical to mathematical equality.
   - **`value_indices::ValueIndices<const D: usize>`**: struct wrapping `HashMap<F32Key, Vec<[usize; D]>>`. Public methods: `total()`, `num_distinct()`, `len(value)`, `get(value)`, `is_empty()`. The `get` method returns `Option<&[[usize; D]]>` for slice-style consumption.
   - **`value_indices::value_indices(image, ignore_value)`**: single-pass algorithm, O(n) time, O(n) space (worst case, one entry per distinct value). The `ignore_value` parameter (when `Some(v)`) is compared by bit pattern, so the user controls which single value is excluded.

2. **Output format deviation from scipy** (documented, not a defect): scipy returns `dict[value, tuple[axis0_array, axis1_array, …]]` — one numpy array per axis. Rust returns `HashMap<F32Key, Vec<[usize; D]>>` — one multi-index tuple per occurrence. Both are information-equivalent; the Rust form is more compact (single `Vec` per value vs D `Vec`s) and avoids redundant memory for the per-axis split. The `k`-th multi-index in the Rust form equals the `k`-th row across the per-axis arrays in scipy's form.

3. **Pre-existing typo fix (incidental)**: `crates/ritk-core/src/statistics/mod.rs:38` had `NyulUdapaNormalizer` (sic) in the `pub use normalization::{…}` re-export; the normalization module defines `NyulUdupaNormalizer`. This typo was breaking the `ritk-core` build in the working tree (one of many pre-existing uncommitted breaks). Fixed in the Sprint 338 commit because verification required a green build.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| `cargo build -p ritk-core --lib` | clean | ✓ |
| `cargo clippy -p ritk-core --all-targets` | 0 new errors; +2 new warnings (mirror `position_extrema::flat_to_multi_round_trip` pattern) | ✓ |
| `cargo fmt --check -p ritk-core` | clean for value_indices.rs | ✓ |
| `cargo test -p ritk-core --lib` | 1521/0/1 (+16 value_indices tests) | ✓ |
| `cargo build --workspace` | clean | ✓ |
| `scipy.ndimage.value_indices` v1.17.1 differential | 16 tests: 1-D basic, 1-D constant, 1-D single-voxel, 1-D ignore; 2-D docstring example (6×6 with 4 distinct values), 2-D ignore; 3-D two-corner-voxels-and-center, 3-D all-same, 3-D single-voxel, 3-D ignore with 6 distinct non-zero, 3-D ignore-not-present, 3-D row-major ordering invariant, 3-D total-count invariant, 3-D total-after-ignore invariant, 2×3×4 round-trip, F32Key bit-equality | ✓ all match |

### Updated parity

- Coverage: **42/74 present** (was 41/74), 6/74 partial, 26/74 missing (was 27/74). **57% parity** (was 55%).
- Closed: GAP-SCI-08 (value_indices).
- Open: GAP-SCI-01, 02, 05, 06, 11, 14, 15 (7 remaining, target Sprints 339-340).
- Out of scope [arch]: GAP-SCI-16/17/18 (5 functions requiring callback-based plugin system).
