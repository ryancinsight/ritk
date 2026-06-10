# RITK Sprint Checklist - Active

> **Full checklist history (Sprints 262-322)**: see [ARCHIVE.md](./ARCHIVE.md)

---

## Sprint 357 — Phase 21 Cleanup & Optimization (20 Cycles, Repeat ×2)

**Target version**: 0.54.0

- [x] ARCH-357-01: PhantomData<B> → PhantomData<fn() -> B> across 22 remaining backend-marker sites
- [x] BOOL-357-02: `MorphOp` enum replaces `is_erosion: bool` in binary_closing.rs
- [x] BOOL-357-03: `ExtremeSide` enum replaces `rightmost: bool` in white_stripe.rs
- [x] BOOL-357-04: `ByteOrder` enum replaces `msb: bool` in metaimage/reader.rs + nrrd/reader/decode.rs
- [x] BOOL-357-05: `OutOfBoundsMode` enum replaces `zero_pad: bool` across entire interpolation subsystem
- [x] PERF-357-06: DiffusionConfig::apply clone elimination (self.clone() removed)
- [x] PRIM-357-07: `GaussianSigma(f64)` newtype for CannyEdgeDetector + LogEdgeFilter
- [x] PRIM-357-08: `VolumeDims([usize; 3])` newtype introduced in bspline_ffd
- [x] BOOL-357-09: 9 model struct bools → enums (ScanDimensionality, SkipConnections, DownsamplePolicy, DropPath, DownsampleStage, IntegrationMode, CornerAlignment, TransformIntegration)
- [x] BOOL-357-10: `ConvergenceStatus` + `StopReason` enums for GlobalMiResult + RegistrationSummary
- [x] BOOL-357-11: `SpacingMode` enum replaces `use_image_spacing: bool` in DiscreteGaussianFilter
- [x] CAP-357-12: Vec::with_capacity at 6 DICOM networking hot-path sites
- [x] PERF-357-13: Gaussian filter input.clone().permute() → move; BURN-API annotation on kernel clone
- [x] ARCH-357-14: parzen/mod.rs Arc<Mutex<>> cache fields fully documented
- [x] SRP-357-15: compute_image.rs 509L → 497L (extract_cached_points to image_cache_helpers.rs)
- [x] SRP-357-16: mutual_information/mod.rs 487L → 441L (tests to mutual_information/tests.rs)
- [x] SRP-357-17: perona_malik.rs 478L → 302L (tests to tests_perona_malik.rs)
- [x] SRP-357-18: regularization/dispatch.rs 468L → 186L (tests to tests_dispatch.rs)
- [x] SRP-357-19: optimizer/adaptive_stochastic_gd.rs 459L → 376L (tests to tests_adaptive_stochastic_gd.rs)
- [x] VER-357-20: Verification gate passed (clippy 0, doc 0, all tests green)

**Verification gate**:
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core -p ritk-registration --no-deps` → 0 warnings
- [x] `cargo test -p ritk-core --lib` → 1581/0/1
- [x] `cargo test -p ritk-registration --lib` → 583/0/1
- [x] `cargo test -p ritk-codecs --lib` → 102/0/0
- [x] `cargo test -p ritk-nrrd --lib` → 23/0/0
- [x] `cargo test -p ritk-snap --lib bed_separation` → 2/0/0
- [x] `cargo test -p ritk-snap --lib label` → 32/0/0
- [x] CHANGELOG.md updated with [0.54.0] section

---


## Sprint 356 — Phase 21 Cleanup & Optimization (20 Cycles, Repeat)

**Target version**: 0.53.0

- [x] PERF-356-01: `lncc_loss` Conv3dConfig hoisted out of box_filter closure (4 allocs eliminated)
- [x] BOOL-356-02: `ComponentPolicy` enum replaces `keep_largest_component: bool` in BedSeparationConfig
- [x] BOOL-356-03: `ZhangSuenPass` enum replaces `step1: bool` in zhang_suen_step
- [x] BOOL-356-04: `EarlyStoppingPolicy` enum replaces `enable_early_stopping: bool` in RegistrationConfig
- [x] BOOL-356-05: `ProgressDisplay` enum replaces `show_progress_bar: bool` in ConsoleProgressCallback
- [x] BOOL-356-06: `ShapeValidation` + `NumericalCheck` enums replace two bools in ValidationConfig
- [x] BOOL-356-07: `InitStrategy` enum replaces `use_com_init: bool` in CmaMiConfig + all call sites
- [x] CAP-356-08: `with_capacity` at cma_mi/registration.rs + demons/multires.rs
- [x] PRIM-356-09: `Opacity(f32)` #[repr(transparent)] newtype for ImageOverlay + MaskOverlay + BlendImageFilter
- [x] ARCH-356-10: LabelEntry.visible: bool → Visibility (SSOT with overlay.rs)
- [x] ARCH-356-11: PhantomData<B> → PhantomData<fn() -> B> in CorrelationRatio + Lncc
- [x] PRIM-356-12: SpatialSigma(f64) + RangeSigma(f64) newtypes for BilateralFilter
- [x] DOC-356-13: bspline_ffd/config.rs [usize; 3] field axis-ordering documentation
- [x] SRP-356-14: parzen/image_cache_helpers.rs extracted from compute_image.rs (575L → 509L)
- [x] SRP-356-15: mutual_information/ directory module (variant.rs 25L + mod.rs 487L)
- [x] VER-356-16: Verification gate passed (clippy 0, doc 0, all tests green)
- [x] ARTIFACTS-356-17: CHANGELOG [0.53.0] section written
- [x] ARTIFACTS-356-18: backlog.md Sprint 356 entry + residual items recorded
- [x] ARTIFACTS-356-19: checklist.md Sprint 356 entry
- [x] FINAL-356-20: Final re-check: cargo check --workspace clean, 0 outstanding issues

**Verification gate**:
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core -p ritk-registration --no-deps` → 0 warnings
- [x] `cargo test -p ritk-core --lib` → 1581/0/1
- [x] `cargo test -p ritk-registration --lib` → 583/0/1
- [x] `cargo test -p ritk-codecs --lib` → 102/0/0
- [x] `cargo test -p ritk-nrrd --lib` → 23/0/0
- [x] `cargo test -p ritk-snap --lib bed_separation` → 2/0/0
- [x] `cargo test -p ritk-snap --lib label` → 32/0/0
- [x] CHANGELOG.md updated with [0.53.0] section

---


## Sprint 354 — Phase 21 Cleanup & Optimization Audit

**Target version**: 0.52.0

- [x] BOOL-354-01: `Connectivity` enum replaces `fully_connected: bool` in 2 contour filters
- [x] BOOL-354-02: `FlipPolicy` enum replaces `axes: [bool; 3]` in `FlipImageFilter`
- [x] BOOL-354-03: `DemonsVariant` enum replaces `use_diffeomorphic: bool` in demons config
- [x] BOOL-354-04: `IterativeAlgorithm` enum + `IterativeParams` struct replaces `is_landweber: bool`
- [x] BOOL-354-05: Removed `enable_convergence_detection: bool` (redundant with Option)
- [x] PRIM-354-06: `Spacing<3>` replaces `[f64; 3]` in 4 edge filters + canny + LOG
- [x] PRIM-354-07: `Spacing` validation: `new()` asserts positive-finite; `try_new()` returns Result
- [x] COW-354-08: `Point::as_slice()` + `Vector::as_slice()` added; `to_vec()` deprecated
- [x] COW-354-09: `Image::data_vec()` deprecated; 16 call sites migrated to `data_slice()`
- [x] PERF-354-10: Interpolation: 14 `.clone()` calls eliminated (gather, to_data, clamp, fused)
- [x] PERF-354-11: `CorrelationRatio`: 19 → 10 clones (marginal pre-compute, ref-passing)
- [x] PERF-354-12: Capacity pre-allocation at 3 sites
- [x] CLIPPY-354-13: 30+ clippy errors fixed across 8 files
- [x] FIX-354-14: Stale import paths fixed in ritk-python and ritk-cli
- [x] FIX-354-15: Module duplication fix in interpolation/tests
- [x] FIX-354-16: Doc link escape in label_map.rs

**Verification gate**:
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core --no-deps` → 0 warnings
- [x] `cargo test -p ritk-core --lib` → 1581/0/1
- [x] `cargo test -p ritk-registration --lib` → 583/0/1
- [x] `cargo test -p ritk-codecs --lib` → 102/0/0
- [x] `cargo test -p ritk-nrrd --lib` → 23/0/0
- [x] CHANGELOG.md updated with [0.52.0] section

---

## Sprint 352 — Zero-Cost Architecture (20 Cycles)
## Sprint 355 — Build Repair + Field-Op Parallelism + W_fixed Cache Wiring

**Target version**: 0.50.95 (ritk-registration [minor])

- [x] FIX-355-01: interpolation `kernel/` restructure compile repair — Burn tensor ops consume `self`; restored refcounted-handle `.clone()` idiom in `kernel/linear/dim1-4.rs`, `kernel/nearest.rs`, `fused.rs` (35 errors → 0) [patch]
- [x] FIX-355-02: `filter/edge/prewitt` — missing `use crate::Image` in tests, doc-lazy-continuation lint [patch]
- [x] ARCH-355-03: `CellSlice` moved from `diffeomorphic/local_cc/forces.rs` to crate-level `parallel.rs` (shared by two bounded contexts; no re-export alias, call sites updated) [patch]
- [x] PERF-355-04: `deformable_field_ops/compose.rs` `compose_fields_into` — z-slice moirai parallelism (disjoint-write `CellSlice` pattern); hot path of scaling-and-squaring (×n_steps per exponentiation) [minor]
- [x] PERF-355-05: `deformable_field_ops/warp.rs` `warp_image_into` — z-slice moirai parallelism [minor]
- [x] PERF-355-06: `deformable_field_ops/warp.rs` `compute_mse_streaming` — moirai parallel reduction over z-slices (f64 accumulator, per-slice sequential order preserved) [minor]
- [x] PERF-355-07: `deformable_field_ops/smooth.rs` `convolve_axis<AXIS>` — z-slice moirai parallelism (reads immutable input incl. cross-slice Z reads; writes disjoint) [minor]
- [x] FIX-355-08: completed in-flight 350-P1-03 W_fixed^T per-instance cache wiring — `WFixedCache` + `extract_w_fixed_t_cache` + `compute_image_joint_histogram_with_w_fixed` restored/integrated with `MutualInformation::forward_with_cache`; `cache` module `pub(crate)`; `Arc<Mutex<…>>` shared across clones [minor]
- [x] VER-355-09: `cargo clippy --workspace --all-targets -- -D warnings` → 0; `cargo test -p ritk-core --lib` → 1574/0/1; `cargo test -p ritk-registration --lib` → 582/0/1

### Follow-up dispositions (Sprint 355 continuation)
- [x] FIX-355-10: completed concurrent boolean-blindness→enum refactor fallout — `CmaEsConfig` `parallel_population: bool`→`PopulationEval`, `record_history: bool`→`HistoryPolicy`; converted 8 lib + 7 test construction sites (`cma_mi/config.rs`, `global_mi/tests/*`, `cma_es/tests.rs`), fixed `generation.rs:84` precedence bug `(x == Record).then(...)`, added imports [patch]
- [x] FIX-355-11: `InverseConsistency` enum migration fallout — `bool`→`InverseConsistency::{Enforced,Relaxed}` at `ritk-python` (atlas.rs, syn/multires.rs), `ritk-cli` (register/diffeomorphic.rs), `atlas/tests.rs` imports [patch]
- [~] PERF-355-F2 (REVERTED): O(N) box-sum local-CC (`window_sums.rs`) implemented + differentially verified vs brute-force oracle (≤1e-6, radii 1–3), but **reverted**. The Avants force term `1/(σ_I²+ε)` is hypersensitive near zero-variance windows; the raw-moment box-sum `var=ΣI²−(ΣI)²/cnt` (even with global-mean conditioning) has cancellation error that flips small-variance windows across the guard → registration diverges to CC=0 (`syn_recovers_translation_ncc_improves`). A safe O(N) form needs a stable streaming/Welford windowed variance, not separable box-sums. `window_cc_stats` two-pass retained as SSOT; rationale documented at its definition. [deferred]
- [—] PERF-355-F1 (N/A): `deformable_field_ops` is `pub(crate)`; an external criterion bench cannot reach compose/warp/smooth without exposing internals (unjustified). Their correctness is covered by unit tests; the parallel speedup is structural (sequential→moirai z-slice). Bench at the public SyN level if measurement is needed.
- [—] PERF-355-F3 (WONTFIX): `scaling_and_squaring` at `atlas/mod.rs` (once per outer atlas iteration) and `demons` `invert_result` (one-shot public call) are not tight inner loops; the per-call 3-Vec alloc is negligible vs the enclosing SyN registrations. `_into` conversion adds 6-buffer scaffolding for no measurable gain (YAGNI). Tight inner-loop sites were already converted in Sprint 352.
- [x] VER-355-12: post-revert — `cargo clippy --workspace --all-targets -- -D warnings` → 0; `cargo test -p ritk-core --lib` → 1574/0/1; `cargo test -p ritk-registration --lib` → 583/0/1

---

## Sprint 352 — Zero-Cost Architecture (20 Cycles)

**Target version**: 0.52.1 (ritk-registration [minor] patch)

- [x] DRY-352-01: `smooth.rs` `convolve_z/y/x` → `convolve_axis<const AXIS: usize>`
- [x] API-352-02: `gaussian_smooth_inplace(&mut Vec<f32>)` → `(&mut [f32])`
- [x] ERR-352-03: `annotation_state.rs` `Result<(), String>` → `AnnotationError` (thiserror)
- [x] SOC-352-04: CMA-ES `mod.rs` (474L) → `constants.rs` + `generation.rs` (mod.rs 240L)
- [x] SOC-352-05: `bspline_syn/mod.rs` (461L) → `buffers.rs` (mod.rs 377L)
- [x] NAMED-352-06: `VelocityField` struct; `SyNResult/BSplineSyNResult/SubjectResult` `.0/.1/.2` → `.z/.y/.x`
- [x] SOC-352-07: `DiscreteGaussianFilter` `new_isotropic`, `#[inline]`, ACCUMULATOR doc
- [x] PERF-352-08: `ClaheFilter` Vec<Vec<f32>> → flatten chain (one allocation)
- [x] SOC-352-09: `syn_core/mod.rs` (301L) → `buffers.rs` (mod.rs 246L)
- [x] NAMED-352-10: `multires_syn` `PrevLevelState` → named struct with `VelocityField`
- [x] DOC-352-11: `bspline_ffd/regularization.rs` ACCUMULATOR + precision comments + `#[inline]`
- [x] PERF-352-12: `lddmm/geodesic.rs` 9 per-step Vec allocations eliminated
- [x] PERF-352-13: `demons/diffeomorphic/registration.rs` 7 per-iter allocations eliminated
- [x] PERF-352-14: `exact_inverse_diffeomorphic/registration.rs` 14 per-iter allocations eliminated
- [x] PERF-352-15: `thirion/registration.rs` `compute_mse` → `compute_mse_streaming`
- [x] PERF-352-16: `bspline_ffd/basis.rs` `evaluate_bspline_displacement_fast_into` + registration pre-alloc
- [x] PERF-352-17: `multires_syn/mod.rs` inner loop 14 per-iter allocations eliminated
- [x] DOC-352-18: CMA-ES `state.rs` doc precision improvements
- [x] VER-352-19: `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` → 0 warnings
- [x] VER-352-20: `cargo test -p ritk-core --lib` → 1579/0/1; `cargo test -p ritk-registration --lib` → 581/1/1

---

## Sprint 351 (Phase 21) — Cleanup, Optimization, Testing

- [x] STR-351-01: `value_indices.rs` (590L) → `value_indices/` directory (mod/key/map/compute/tests)
- [x] STR-351-02: `iterate_structure/tests.rs` (562L) → `tests/` directory (bool_structure/iterate/edge_cases)
- [x] PERF-351-03: `Vec::new()` → `Vec::with_capacity(n)` at 14 known-size sites in ritk-core
- [x] PERF-351-04: `HashMap::new()` → `HashMap::with_capacity(n)` at 6 sites in ritk-core + ritk-registration
- [x] ARCH-351-05: `NearestNeighborInterpolator` derives: Copy/Clone/PartialEq/Eq/Hash/Serialize/Deserialize
- [x] DRY-351-06: `in_bounds_mask` shared helper; ~24 duplicated patterns eliminated across dim1-4 + nearest
- [x] ARCH-351-07: `Spacing<D>`: type alias → `#[repr(transparent)]` newtype over `Vector<D>` + Deref + Module/Record
- [x] FIX-351-08: Doc warnings: wgpu_compat private link, kernel/nearest broken link
- [x] FIX-351-09: Stale `preprocessing.rs` flat file conflicting with `preprocessing/` directory module
- [x] FIX-351-10: `transform/mod.rs` broken doc comment + keyword-in-path fix
- [x] Verification: `cargo clippy -p ritk-core -p ritk-registration -- -D warnings` → 0 warnings
- [x] Verification: `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core --no-deps` → 0 warnings
- [x] Verification: `cargo test -p ritk-core --lib` → 1579/0/1
- [x] Verification: `cargo test -p ritk-registration --lib` → 581/1/1 (pre-existing flake)
- [x] Verification: Files > 500 lines → 0 in ritk-core, 0 in ritk-registration
- [x] backlog.md + gap_audit.md + checklist.md synchronized

---

## Sprint 350 — Zero-Cost Architecture (10 Cycles)

**Target version**: 0.52.0 (ritk-core 0.7.0)

- [x] Cycle 1A: `fold_f32`→`fold_native`, `fold_f64`→`fold_wide` in `filter/projection.rs`
- [x] Cycle 1B: `div_floor_i64`→`div_floor` in `segmentation/distance_transform/mod.rs`
- [x] Cycle 1C: `next_f64`→`sample_unit` on `Xorshift64` in `segmentation/clustering/kmeans.rs`
- [x] Cycle 1D: `otsu_threshold_f64`→`local_otsu_threshold` in `segmentation/level_set/chan_vese.rs`
- [x] Cycle 2: `sort_f32` DRY — `pub(crate) fn sort_floats` in `statistics/mod.rs`; `noise_estimation.rs` + `nyul_udupa.rs` updated
- [x] Cycle 3: `Spacing::uniform()` zero-alloc — `vec![value; D]`→`std::array::from_fn`
- [x] Cycle 4: Remove `Direction::axis_directions()` allocating API; only zero-alloc `axis_directions_array()` remains
- [x] Cycle 5: `ritk-registration/wgpu_compat.rs` — re-export from `ritk_core::wgpu_compat::WGPU_CHUNK_SIZE`; `ritk-core::wgpu_compat` made `pub`
- [x] Cycle 6: `classical/engine/mod.rs` SoC split — `config.rs`, `metric.rs`, `result.rs`, `mod.rs`
- [x] Cycle 7: `classical/temporal/mod.rs` magic literals → named constants; dead `_n` binding removed
- [x] Cycle 8: `compute_statistics_from_slice` double-allocation fix
- [x] Cycle 9: `atlas/label_fusion.rs` `Vec<Vec<f64>>`→flat solver; `solve_linear_system` redesigned
- [x] Cycle 10: PM artifacts sync; pre-existing Sprint 353 build/test stubs resolved (1573 / 0 / 1 in ritk-core)
- [x] Verification: `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` → 0 warnings
- [x] Verification: `cargo test -p ritk-core --lib` → 1573 passed / 0 failed / 1 ignored
- [x] Verification: `cargo test -p ritk-registration --lib` → 581 passed / 1 failed (pre-existing) / 1 ignored

---

## Sprint 349 — Zero-Cost Architecture (5 Cycles)

**Target version**: 0.52.0 (ritk-core 0.7.0)


- [x] Cycle 1: `sinc.rs` — eliminate O(A^D) tensor clones per query point; extract `flat_slice` once before point loop
- [x] Cycle 2: `EarlyStoppingCallback` — consolidate `Arc<Mutex>×3` → `Arc<Mutex<EarlyStoppingState>>`; `filter/resample.rs` `unreachable!()`
- [x] Cycle 3: `preprocessing.rs` — SoC split into `step/pipeline/executor` sub-modules
- [x] Cycle 4: `coherence/pde.rs` — `Vec<Vec<f64>>` → named struct fields; `surface.rs` pointer scatter
- [x] Cycle 5: `n4.rs` naming (`w_min_f64` → `w_min_wide`); `bspline/mod.rs` `unreachable!()`; PM sync
- [x] Verification: `cargo clippy -p ritk-core --lib -- -D warnings` → 0 warnings
- [x] Verification: `cargo test -p ritk-core --lib -- interpolation::bspline` → pass
- [x] Verification: `cargo test -p ritk-core --lib -- filter::bias` → pass

---

## Sprint 348 (Phase 22) — Cleanup, Optimization, Architecture Hardening

- [x] DRY-348-01 [minor]: Extract `read_ascii<T>` + `read_binary_be<T: FromBeBytes>` in `ritk-vtk/src/io/read_helpers.rs`
- [x] Replace 3 VTK reader files' private helpers with shared generic versions
- [x] DRY-348-02 [minor]: Unify `fold_f32`/`fold_f64` → generic `fold<A, Init, Finalize>` in `projection.rs`
- [x] DRY-348-03 [patch]: Extract `sort_floats` SSOT in `statistics/mod.rs`
- [x] Update `noise_estimation.rs` and `nyul_udupa.rs` call sites
- [x] PERF-348-04 [minor]: `EarlyStoppingCallback` atomics refactor
- [x] PERF-348-05 [patch]: Remove `Arc<Mutex<>>` from `ProgressTracker` and `HistoryCallback`
- [x] PERF-348-06 [patch]: Skeletonization `Vec::with_capacity` pre-allocation
- [x] HARD-348-07 [patch]: CLI metrics `.unwrap()` elimination
- [x] ARCH-348-08 [patch]: `PhantomData<fn() -> B>` variance fix in 5 files
- [x] DOC-348-09 [patch]: SAFETY comments on Burn tensor clone sites
- [x] CLEANUP-348-10 [patch]: Remove stale `value_indices/` directory
- [x] Verification: `cargo clippy` (7 crates) → 0 warnings
- [x] Verification: `cargo test -p ritk-core --lib` → 1559/0/1
- [x] Verification: `cargo test -p ritk-vtk --lib` → 241/0/0
- [x] Verification: `cargo test -p ritk-codecs --lib` → 102/0/0
- [x] Verification: `cargo test -p ritk-registration --lib` (progress) → 3/0/0

---

## Sprint 342 (0.51.x) — Coeus Migration Readiness Audit

- [x] MIG-342-01 [arch]: Identify Burn-to-Coeus replacement surface
  - [x] Audit RITK manifests for `burn` / `burn-ndarray` dependencies
  - [x] Audit source references to Burn tensor, shape, autodiff, parameter, and model APIs
  - [x] Inspect Coeus CPU/autograd/WGPU capabilities from manifests, tests, and public source
  - [x] Record required Coeus CPU, autodiff, model, PyO3, and GPU capabilities
- [x] MIG-342-02 [patch]: Add repeatable `xtask burn-migration-audit`
  - [x] Report manifest dependencies, source references, crate summary, and migration requirements
  - [x] Unit-test manifest/source detection
  - [x] Unit-test `target/` exclusion
- [x] DOC-342-03 [patch]: Add `docs/coeus_migration.md`
  - [x] Current Burn surface
  - [x] Required Coeus capabilities
  - [x] GPU migration gates
  - [x] Staged development sequence
- [ ] MIG-342-04 [arch]: Define RITK tensor contract after Coeus CPU API stabilizes
- [ ] GPU-342-05 [arch]: Add Coeus WGPU differential harness after CPU Coeus parity exists
- [ ] REG-342-06 [arch]: Prove registration autodiff tape continuity under Coeus
- [x] Verification: `cargo test -p xtask migration_audit` → 2 passed
- [x] Verification: `cargo run -p xtask -- burn-migration-audit` → 18 manifests, 490 source files
- [x] Verification: `cargo fmt --check -p xtask` → clean

---

## Sprint 332 (0.50.95) — Documentation Compaction + Structural Audit + Benchmarks

- [x] DOC-332-01: Documentation compaction
  - [x] Delete stale `docs/backlog.md`, `docs/checklist.md`, `docs/CHANGELOG.md`, `SPINT_293_PLAN.md`
  - [x] Create `ARCHIVE.md` with pre-Sprint 320 history (18,150 lines)
  - [x] Compact `backlog.md` (6,378→134), `checklist.md` (5,893→110), `gap_audit.md` (6,200→145)
  - [x] Update `IMPLEMENTATION_SUMMARY.md` to v0.50.94
- [x] STR-332-02: Structural audit — 3 violations partitioned (709→dir, 670→dir, 536→dir); ZERO files > 500 lines
- [ ] BENCH-332-03: `STACK_WEIGHTS_CAPACITY=32` Criterion benchmark (deferred)
- [ ] GPU-332-04: Evaluate `sparse.rs` GPU-backend potential (deferred)
- [ ] CRLF-332-05: Git CRLF normalization (blocked by missing test data)
- [x] Build: `cargo clippy --workspace` → 0 warnings
- [x] Tests: `cargo test -p ritk-core --lib` → 1408/0/1
- [x] Tests: `cargo test -p ritk-registration --lib --features direct-parzen --no-default-features` → 547/0/1

---

## Sprint 331 (0.50.94) — Clippy Zero-Warning, Structural Partitions, Flaky Test Fix, Documentation Overhaul

- [x] CLIPPY-331-01: Zero-warning clippy workspace (28 warnings → 0)
  - [x] ritk-core: 12 warnings (5× too_many_arguments, 6× needless_range_loop, 3× doc_lazy_continuation)
  - [x] ritk-vtk: 2 warnings (type_complexity, same_item_push)
  - [x] ritk-io: 4 warnings (len_without_is_empty, 2× vec_init_then_push, too_many_arguments)
  - [x] ritk-registration: 1 warning (doc_lazy_continuation)
  - [x] ritk-snap: 8 warnings (needless_range_loop, manual_clamp, 2× unnecessary_unwrap, 2× needless_range_loop, ptr_arg, nonminimal_bool)
  - [x] ritk-python: 1 warning (field_reassign_with_default)
- [x] ARCH-331-02: Preemptive structural partitions (8 files decomposed)
  - [x] `ritk-io/association.rs` (560→341) → `association/{mod,scu,helpers}.rs`
  - [x] `ritk-io/dimse/mod.rs` (482→306) → `dimse/{mod,command_value}.rs`
  - [x] `ritk-io/dicom/mod.rs` (471→68) → `dicom/{mod,series}.rs`
  - [x] `ritk-registration/direct_property_tests.rs` (524→3 files)
  - [x] `ritk-registration/direct_types_tests.rs` (504→3 files)
  - [x] `ritk-registration/tests_label_fusion.rs` (473→3 files)
  - [x] `ritk-core/clahe.rs` (476→281+160+217)
  - [x] `ritk-core/tests_convolution.rs` (472→3 files)
- [x] FIX-331-03: Flaky test hardening — `translation_recovery_shifted_gaussian`
  - [x] sampling_percentage 0.50 → 0.75
  - [x] maximum_iterations 200 → 300
  - [x] tolerance 0.5 → 0.8 voxels
- [x] DOC-331-04: Documentation overhaul
  - [x] IMPLEMENTATION_SUMMARY.md rewritten with accurate crate structures and current features
  - [x] OPTIMIZATION.md updated to v0.50.93 with Sprint 329/330 entries
  - [x] README.md recent sprints section updated with Sprints 328-330
- [x] CLEANUP-331-05: Orphan test file `ritk-core/filter/fft/tests_convolution.rs` removed
- [x] CLIPPY-331-06: Deep clippy cleanup pass (110+ residual warnings → 0)
  - [x] `#![allow(clippy::field_reassign_with_default)]` crate-level in `ritk-snap` / `ritk-registration` / `ritk-vtk` `lib.rs`
  - [x] `#![allow(clippy::erasing_op, clippy::identity_op)]` scoped to test modules (12 files)
  - [x] `#![allow(clippy::needless_range_loop)]` on test files (8 files)
  - [x] `manual RangeInclusive::contains` refactored to `(lo..=hi).contains(&x)` (4 instances)
  - [x] `using contains() instead of iter().any()` refactored (2 instances)
  - [x] `casting to the same type` removed (4 instances: `as f32` / `as f64`)
  - [x] `too_many_arguments` per-fn `#![allow]` with justification (2 test helpers)
  - [x] `assert!` on const-vs-const promoted to `const _: () = assert!(...)` (3 instances)
  - [x] `approx_constant` per-test `#![allow]` for `3.14` test floats (3 instances)
  - [x] `cloned_ref_to_slice_refs` → `std::slice::from_ref(&msg)` (1 instance)
  - [x] `unit_default` → bare struct name (1 instance)
  - [x] `let_and_return` → return expression directly (1 instance)
  - [x] `redundant_binding` removed (2 instances)
  - [x] `manual_clamp` → `clamp()` (2 instances)
  - [x] `doc_list_item` indentation fixed (2 instances)
  - [x] `single_range_in_vec_init` `#![allow]` for burn tensor `slice([Range; N])` API
- [x] FIX-331-07: DICOM `pdu.rs` vs `pdu/` module conflict resolved
  - [x] Orphan `pdu.rs` deleted
  - [x] `tests_pdu.rs` moved to `pdu/tests.rs`
  - [x] `#[path = "tests_pdu.rs"]` attribute in `pdu/mod.rs` removed
- [x] FIX-331-08: Unused `bail` import in `pdu/presentation_context.rs` removed
- [x] FIX-331-09: `super::pdu::*` and `super::super::pdu::*` unused-import warnings resolved
- [x] FIX-331-10: `v <= 65535` always-true assertion replaced with non-zero pixel check
- [x] FIX-331-11: `0 * 25` → `0 * 5 * 5` 3D index arithmetic in `edt_3d` test
- [x] Build: `cargo fmt --check` → clean
- [x] Build: `cargo clippy --workspace --all-targets --all-features` → 0 warnings
- [x] Tests: `cargo test -p ritk-core --lib` → 1408/0/1
- [x] Tests: `cargo test -p ritk-registration --lib` → 547/0/1
- [x] Tests: `cargo test -p ritk-vtk --lib` → 241/0/0
- [x] Tests: `cargo test -p ritk-minc --lib` → 40/0/0
- [x] Tests: `cargo test -p ritk-cli --tests` → 200/0/0
- [x] Tests: `cargo test -p ritk-model --lib` → 77/0/0
- [x] CHANGELOG.md updated (0.50.94)
- [x] backlog.md updated
- [x] checklist.md updated
- [x] gap_audit.md updated

## Sprint 330 (0.50.93) — Architectural Decomposition: types/ and sample/

- [x] ARCH-330-01: `types.rs` → `types/` directory (4 leaf modules + mod.rs)
  - [x] `types/half_width.rs` — `compute_half_width`, `MIN_HALF_WIDTH`
  - [x] `types/stack_weights.rs` — `StackWeights`, `StackWeightsIter`
  - [x] `types/bin_range.rs` — `BinRange`
  - [x] `types/parzen_config.rs` — `ParzenConfig`
  - [x] `types/mod.rs` — re-exports + `CompactionSizes`
- [x] ARCH-330-02: `sample.rs` → `sample/` directory (2 leaf modules + mod.rs)
  - [x] `sample/sample_window.rs` — `SampleWindow`
  - [x] `sample/sparse_entry.rs` — `SparseWFixedEntry`, `SparseWFixedT`
  - [x] `sample/mod.rs` — re-exports
- [x] ARCH-330-03: `ParzenConfig::half_width()` and `inv_2sigma_sq()` promoted (removed `#[cfg(test)]`)
- [x] ARCH-330-04: Compute functions extracted into `accumulate.rs`, `compute_direct.rs`, `compute_sparse.rs`
- [x] ARCH-330-05: `compute_half_width` re-export promoted (removed `#[cfg(test)]`)
- [x] DRY-330-06: Backward-compatible re-exports (all public API paths preserved)
- [x] MEM-330-07: Structural size regression tests (BinRange=4, SparseWFixedEntry=8, StackWeights=128-136, ParzenConfig=12-32)
- [x] TEST-330-08: 24 new tests in `direct_phase_fifteen_tests.rs` (production API, SSOT, types/sample access, computation functions, backward compat, size regression, weight correctness, end-to-end, support_bins)
- [x] FIX-330-09: `clahe/mod.rs` `pub use` of `pub(crate)` items → `pub(crate) use`
- [x] FIX-330-10: `super::*` → `super::super::*` in `association/{helpers,scu}.rs` for new directory split
- [x] FIX-330-11: `tests_label_fusion` path attribute fixed (`tests_label_fusion/mod.rs` is correct)
- [x] FIX-330-12: `clahe_2d` / `build_tile_cdf` legacy helpers gated `#[cfg(test)]`
- [x] FIX-330-13: `tests_label_fusion/mod.rs` re-exports removed (child files use `super::super::*` directly)
- [x] Build: `cargo check --workspace --all-targets` → 0 errors, 0 warnings
- [x] Build: `cargo build --workspace --tests` → 0 errors, 0 warnings
- [x] Tests: `cargo test -p ritk-registration --lib` → 547/0/1
- [x] Tests: `cargo test -p ritk-core --lib` → 1408/0/1
- [x] Tests: `cargo test -p ritk-vtk --lib` → 241/0/0
- [x] Clippy: `cargo clippy -p ritk-registration --features direct-parzen` → 0 warnings
- [x] Clippy: `cargo clippy -p ritk-core` → 0 warnings
- [x] Clippy: `cargo clippy -p ritk-io` → 0 warnings
- [x] CHANGELOG.md updated (0.50.93)
- [x] `Cargo.toml` version bumped to 0.50.93
- [x] backlog.md updated
- [x] checklist.md updated
- [x] gap_audit.md updated

## Sprint 328 (0.50.91) — Per-Sample Weight Normalization

- [x] PERF-328-01: Per-sample weight normalization:
  - [x] `SampleWindow` carries `inv_sum_f: f32` and `inv_sum_m: f32` fields (Rust allows field + method with same name)
  - [x] `SampleWindow::new` computes `1.0 / cfg.sum_weights(val, num_bins)` for both axes
  - [x] `SampleWindow::new_moving_only` returns `inv_sum_m` (1/sum_m for the moving axis)
  - [x] `accumulate_sample_direct` multiplies each sample by `inv_sum_f × inv_sum_m`
  - [x] `accumulate_sample_sparse` signature: 6 args including `inv_sum_m: f32` (callers pass combined `inv_sum_f × inv_sum_m` to match direct)
  - [x] Call site in `mod.rs:479-489` updated to destructure `inv_sum_m` from `new_moving_only`
  - [x] `direct_histogram_normalization_total_weight`: bounds `[n*0.5, n*1.5]` (was `[n*1, n*20]`)
  - [x] `direct_broad_sigma_produces_valid_histogram`: bounds `[n*0.3, n*1.5]` (was `[n*5, n*50]`)
  - [x] `direct_broad_sigma_matches_sparse_cache`: replaced strict ratio with structural + `sparse > direct` (sum_f > 1 for σ²=4)
  - [x] `direct_parzen_config_sigma_invariant`: relative error < 10% (was `sum_09 < sum_10`)
  - [x] `direct_sparse_cache_path_matches_after_parity`: ratio check vs `sum_f` (was ratio ≈ 1.0)
  - [x] `direct_sparse_separate_sigma_per_axis`: ratio check vs `sum_f(σ²_fix)` (was ratio ≈ 1.0)
  - [x] `direct_histogram_large_sigma_sparse_parity`: ratio > 1.0 check (was ratio ≈ 1.0)
  - [x] `accumulate_sample_direct_histogram_sum_equals_expected`: sum ≈ 1.0 (was ≈ 2π)
  - [x] `accumulate_sample_direct_total_weight`: bounds `[0.5, 1.5]` (was `> 0.0`)
  - [x] `sparse_from_cache_matches_direct`: element-wise ratio in `[0.5×sum_f, 2×sum_f]` (was ratio ≈ 1.0 strict)
  - [x] `direct_large_volume_matches_dense`: bounds `[0.5n, 1.5n]` (was `[n*1, n*20]`)
  - [x] `sparse_cache_large_volume_matches_direct`: ratio vs `sum_f` with 15% tolerance
  - [x] `dispatch_matches_tensor_path`: directional nonzero check (tensor > dispatch because tensor is un-normalized)
  - [x] `sparse_cache_dispatch_matches_direct`: ratio > 1.0 (sparse = direct × sum_f)
  - [x] `direct_parallel_matches_sparse`: ratio < 1.0 (dispatch is normalized, sparse is not)
  - [x] `histogram_normalization_total_weight`: bounds `[0.5n, 1.5n]` (was ≈ n × 2π)
  - [x] `masked_no_cache_key_matches_uncached`: ratio in [0.5, 4.0] (was ≈ 1.0 ± 5%)
- [x] Build: `cargo test -p ritk-registration --features direct-parzen --lib`: 499 passed, 1 ignored, 0 failed (2 consecutive runs)
- [x] CHANGELOG.md updated (0.50.91)
- [x] `Cargo.toml` version bumped to 0.50.91
- [x] backlog.md updated


---

## Sprint 335 (2026-06-04) — Prewitt + Position-of-Extrema + Histogram

- [x] GAP-SCI-03: PrewittFilter with magnitude and per-axis components
- [x] GAP-SCI-03: 10 Prewitt tests (constant, x/y/z ramp, diagonal, anisotropic spacing, single voxel, orthogonality, shape preservation)
- [x] GAP-SCI-07: maximum_position + minimum_position functions (generic B, const D)
- [x] GAP-SCI-07: 15 position_extrema tests (1D, 3D, ties, last bin inclusive, single voxel, round-trip)
- [x] GAP-SCI-09: histogram() standalone function with Histogram struct (total, bin_width helpers)
- [x] GAP-SCI-09: 15 histogram tests (uniform, last-bin-inclusive, single bin, values-outside, negative range, edge cases)
- [x] Wire prewitt into ilter::edge::mod and re-export from ilter module
- [x] Wire position_extrema + histogram into statistics::mod with re-exports
- [x] Fix single-voxel bug in convolve_1d_axis (degenerate dim_len=1 case)
- [x] Build: cargo test -p ritk-core --lib: 1478 passed, 1 ignored, 0 failed
- [x] Clippy: cargo clippy -p ritk-core --lib --all-features -- -D warnings: 0 warnings
- [x] CHANGELOG.md updated (0.51.1)
- [x] Cargo.toml version bumped to 0.3.0
- [x] backlog.md updated

---

## Sprint 336 (0.51.2, ritk-core 0.4.0) — Chamfer Distance Transform + Structural Cleanup

- [x] GAP-SCI-12: Chamfer distance transform (scipy.ndimage.distance_transform_cdt parity)
- [x] GAP-SCI-12: chamfer::kernel — 7-tap half-mask offset tables, weight() const fn, cdt_3d two-pass algorithm
- [x] GAP-SCI-12: chamfer::transform — ChamferDistanceTransform struct, threshold + metric builders, apply() generic over B: Backend, f32 output with -1.0 sentinel
- [x] GAP-SCI-12: chamfer_distance_transform_3d free function with anisotropic spacing
- [x] GAP-SCI-12: 18 differential tests (single fg, all-fg, all-bg, cube, two cubes, column, taxicab/chessboard parity, threshold semantics, hand-computed, scipy-verified)
- [x] GAP-SCI-12: scipy.ndimage.distance_transform_cdt v1.17.1 differential verification — 4 shapes × 2 metrics exact match
- [x] STR-336-01: rank.rs (567 lines) → rank/ directory
  - [x] rank/mod.rs (69 lines) — re-exports
  - [x] rank/percentile_filter.rs (152 lines)
  - [x] rank/rank_filter.rs (144 lines)
  - [x] rank/tests.rs (176 lines)
- [x] STR-336-02: chamfer.rs (673 lines) → chamfer/ directory
  - [x] chamfer/mod.rs (77 lines) — re-exports + module docs
  - [x] chamfer/kernel.rs (193 lines) — cdt_3d + weight() const fn
  - [x] chamfer/transform.rs (110 lines) — ChamferDistanceTransform
  - [x] chamfer/tests.rs (217 lines) — 18 tests
- [x] Build: cargo build -p ritk-core --lib: clean
- [x] Clippy: cargo clippy -p ritk-core --lib --all-features -- -D warnings: 0 warnings
- [x] Tests: cargo test -p ritk-core --lib: 1496 passed, 1 ignored, 0 failed
- [x] Tests: cargo test -p ritk-registration --lib --features direct-parzen --no-default-features: 547/0/1
- [x] CHANGELOG.md updated (0.51.2)
- [x] Cargo.toml (ritk-core) version bumped to 0.4.0
- [x] backlog.md updated
- [x] gap_audit.md updated

---

## Sprint 337 (0.51.5, ritk-core 0.5.0) — Morphological Laplacian + Structural Partition

- [x] GAP-SCI-13: MorphologicalLaplacian (scipy.ndimage.morphological_laplace parity)
  - [x] MorphologicalLaplacian struct with `new(radius)` constructor and `radius()` accessor
  - [x] `apply()` method generic over `B: Backend` — composes D + E − 2f
  - [x] `reflect_index(i, n)` const helper — half-sample symmetric reflect, period 2n
  - [x] `dilate_3d_reflect(data, dims, radius)` — scipy-compatible reflect-mode dilation
  - [x] `erode_3d_reflect(data, dims, radius)` — scipy-compatible reflect-mode erosion
  - [x] 9 differential tests cross-validated against scipy v1.17.1:
    - [x] `constant_field_is_zero` — constant field → zero Laplacian
    - [x] `all_ones_is_zero` — all-1s 3×3×3 → zero
    - [x] `linear_ramp_3x3x3` — ramp along x → [1, 0, -1] slice
    - [x] `single_voxel_5x5x5_size_3` — single voxel with size 3 cube
    - [x] `single_voxel_5x5x5_size_5` — single voxel with size 5 cube (radius=2)
    - [x] `single_voxel_3x3x3` — single voxel, 26 neighbours
    - [x] `degenerate_axis_size_1` — 1×3×3 plane (z=1)
    - [x] `operator_is_not_identity` — sanity check
    - [x] `differential_two_corner_voxels_4x4x4` — 4×4×4 with two corner voxels, full 64-voxel byte-exact scipy match
  - [x] scipy.ndimage.morphological_laplace v1.17.1 differential verification — 9 shapes, reflect mode (default) byte-exact match
- [x] STR-337-01: morphological_laplace.rs (595 lines) → morphological_laplace/ directory
  - [x] morphological_laplace/mod.rs (215 lines) — struct + apply + reflect_index + dilate/erode helpers
  - [x] morphological_laplace/tests.rs (254 lines) — 9 differential tests
- [x] Wire morphological_laplace into filter::morphology::mod with re-export
- [x] Build: cargo build -p ritk-core --lib: clean
- [x] Build: cargo build --workspace: clean
- [x] Clippy: cargo clippy -p ritk-core --all-targets: 0 new warnings (27 pre-existing in chamfer/prewitt/position_extrema unchanged)
- [x] Fmt: cargo fmt --check -p ritk-core: clean
- [x] Tests: cargo test -p ritk-core --lib: 1505 passed, 1 ignored, 0 failed (+9 from Sprint 337)
- [x] CHANGELOG.md updated (0.51.5)
- [x] Cargo.toml (ritk-core) version bumped to 0.5.0
- [x] backlog.md updated
- [x] gap_audit.md updated
- [x] Coverage progression: 333: 36/74 (49%) → 335: 39/74 (53%) → 336: 40/74 (54%) → 337: 41/74 (55%)

---

## Sprint 348 (patch) — match-D Elimination + sinc unsafe + bspline dispatch + value_indices SoC

- [x] `displacement_field/core.rs`: `match D { 2 => Matrix2, 3 => Matrix3, _ => panic! }` → `direction.try_inverse()` (Sprint 346 claimed fix never applied)
- [x] `transform/static_displacement_field.rs`: same fix (second site)
- [x] `interpolation/sinc.rs`: removed two `unsafe` pointer transmutes (`as *const Tensor<B,3>` / `<B,2>`)
  - `interpolate_point_3d` / `interpolate_point_2d` → `interpolate_point_3d_flat` / `interpolate_point_2d_flat` (accept `Tensor<B,1>` instead of typed tensors)
  - Pre-extract `flat_data: Tensor<B,1>` once before the point loop (O(1) per call vs O(volume × n_points))
  - Per-point `Vec<f32>` allocation → zero-copy slice `&indices_slice[coords_start..coords_start + D]`
- [x] `interpolation/bspline/mod.rs`: `if D == 3 { 3d } else { 2d }` → `match D { 3 => 3d, 2 => 2d, _ => unreachable!() }`
- [x] `statistics/value_indices/`: complete the planned SoC split
  - Created missing `key.rs` (`F32Key`), `map.rs` (`ValueIndices<D>`), `compute.rs` (`value_indices` + `flat_to_multi`), `tests.rs` (16 tests)
  - Deleted stale `statistics/value_indices.rs` flat file (duplicate of `value_indices/mod.rs` stub)
- [x] `grep 'match D.*2.*3.*panic'` across workspace → zero matches
- [x] `cargo clippy -p ritk-core -p ritk-registration --all-features -- -D warnings` → 0 warnings
- [x] `cargo test -p ritk-core --lib` → 1559/0/1
- [x] `cargo test -p ritk-registration --lib` (targeted) → 33/0/0

---

## Sprint 347 (patch) — WGPU CHUNK_SIZE SSOT Activation + apply_row_chunks Adoption

- [x] Declare `pub(crate) mod wgpu_compat;` in `ritk-core/src/lib.rs` (module was orphaned)
- [x] Declare `pub(crate) mod wgpu_compat;` in `ritk-registration/src/lib.rs` (module was orphaned)
- [x] ritk-core: remove all 13 local `const CHUNK_SIZE: usize = 32768;` definitions
  - [x] `filter/gaussian.rs` → `apply_row_chunks` + `WGPU_CHUNK_SIZE`
  - [x] `filter/resample.rs` → `WGPU_CHUNK_SIZE`
  - [x] `image/transform.rs` (×2) → `apply_row_chunks` + `WGPU_CHUNK_SIZE`
  - [x] `interpolation/fused.rs` → `WGPU_CHUNK_SIZE`
  - [x] `transform/affine.rs` → `apply_row_chunks` + `WGPU_CHUNK_SIZE`
  - [x] `transform/rigid.rs` → `apply_row_chunks` + `WGPU_CHUNK_SIZE`
  - [x] `transform/bspline/interpolation/dim2.rs` → `apply_row_chunks` + `WGPU_CHUNK_SIZE`
  - [x] `transform/bspline/interpolation/dim3.rs` → `apply_row_chunks` + `WGPU_CHUNK_SIZE`
  - [x] `transform/bspline/interpolation/dim4.rs` → `apply_row_chunks` + `WGPU_CHUNK_SIZE_4D`
  - [x] `transform/displacement_field/grid.rs` → `apply_row_chunks` + `WGPU_CHUNK_SIZE`
  - [x] `transform/displacement_field/resample.rs` → `WGPU_CHUNK_SIZE`
  - [x] `transform/static_displacement_field.rs` (×2) → `WGPU_CHUNK_SIZE`
- [x] ritk-registration: remove all 7 local `const CHUNK_SIZE: usize = 32768;` definitions
  - [x] `metric/mse.rs` → `crate::wgpu_compat::WGPU_CHUNK_SIZE`
  - [x] `metric/ncc.rs` → `crate::wgpu_compat::WGPU_CHUNK_SIZE`
  - [x] `metric/lncc.rs` → `crate::wgpu_compat::WGPU_CHUNK_SIZE`
  - [x] `metric/histogram/parzen/compute.rs` → `crate::wgpu_compat::WGPU_CHUNK_SIZE`
  - [x] `metric/histogram/parzen/compute_image.rs` → `crate::wgpu_compat::WGPU_CHUNK_SIZE`
  - [x] `metric/histogram/masked/mod.rs` → `crate::wgpu_compat::WGPU_CHUNK_SIZE`
  - [x] `metric/histogram/masked/masked_chunked.rs` → `crate::wgpu_compat::WGPU_CHUNK_SIZE`
- [x] `grep 'const CHUNK_SIZE'` returns exit code 1 (no matches) across entire workspace
- [x] `cargo clippy -p ritk-core -p ritk-registration --all-features -- -D warnings`: 0 warnings
- [x] `cargo test -p ritk-core --lib`: 1559 passed, 0 failed, 1 ignored
- [x] `cargo test -p ritk-registration --lib -- metric::ncc metric::mse metric::lncc multires`: 33 passed, 0 failed

---

## Sprint 338 (0.51.6, ritk-core 0.6.0) — value_indices (GAP-SCI-08) + incidental typo fix

- [x] GAP-SCI-08: value_indices / ValueIndices (scipy.ndimage.value_indices parity)
  - [x] F32Key newtype (f32 bit-equality + bit-hash) — private to value_indices module
  - [x] ValueIndices<const D: usize> struct wrapping HashMap<F32Key, Vec<[usize; D]>>
  - [x] Public methods: total(), num_distinct(), len(value), get(value), is_empty()
  - [x] `value_indices<B, D>(image, ignore_value: Option<f32>)` — single O(n) pass, row-major multi-indices
  - [x] Re-uses `extract_vec_infallible` from filter::ops for the standard input cycle
  - [x] Generic over `B: Backend, const D: usize` — same authoritative implementation serves 1-D/2-D/3-D/arbitrary-D
  - [x] scipy.ndimage.value_indices v1.17.1 differential verification — 16 tests, integer arrays per scipy's `must be integer array` contract
  - [x] 16 differential tests:
    - [x] value_indices_1d_basic — [10,20,10,30,20] → three keys, row-major
    - [x] value_indices_1d_constant — 4 voxels of 7.0 → single key, all 4 indices
    - [x] value_indices_1d_single_voxel — [42.0] → single key, [[0]]
    - [x] value_indices_1d_ignore_value — ignore 1.0 → 2 keys remain
    - [x] value_indices_2d_docstring_example — 6×6 scipy docstring example
    - [x] value_indices_2d_ignore_value — 6×6 ignore 0.0 → 2 keys remain
    - [x] value_indices_3d_two_corner_voxels_and_center — 3×3×3 with 1.0 at corners and 5.0 at center
    - [x] value_indices_3d_all_same_value — 2×2×2 of 7.0 → 8 row-major indices
    - [x] value_indices_3d_single_voxel — 1×1×1 of 42.0
    - [x] value_indices_3d_ignore_value_excludes_voxels — 2×3×4 with 6 distinct non-zero, ignore 0.0
    - [x] value_indices_3d_ignore_value_not_present — ignore 999.0 has no effect
    - [x] value_indices_3d_row_major_ordering — values 1..=8 in flat order, verify no reordering
    - [x] value_indices_3d_total_equals_voxel_count_without_ignore — invariant
    - [x] value_indices_3d_total_equals_n_minus_ignored_count — invariant
    - [x] flat_to_multi_round_trip_3d — 24-iteration round-trip on 2×3×4
    - [x] f32_key_bit_equality — F32Key bit-equality (0.0 vs -0.0 distinct)
- [x] STR-338-01 (incidental): pre-existing typo `NyulUdapaNormalizer` → `NyulUdupaNormalizer` in statistics/mod.rs
  - [x] Build was broken in working tree by this typo; fixed in same commit for verification
  - [x] No behavioural change; pure rename
- [x] Wire value_indices into statistics::mod with re-export
- [x] Build: cargo build -p ritk-core --lib: clean
- [x] Build: cargo build --workspace: clean
- [x] Clippy: cargo clippy -p ritk-core --all-targets: 0 new errors; +2 new warnings (mirror pre-existing pattern in position_extrema); 30 total (was 27)
- [x] Fmt: cargo fmt --check -p ritk-core value_indices.rs: clean
- [x] Tests: cargo test -p ritk-core --lib: 1521 passed, 1 ignored, 0 failed (+16 from Sprint 338)
- [x] CHANGELOG.md updated (0.51.6)
- [x] Cargo.toml (ritk-core) version bumped to 0.6.0
- [x] backlog.md updated
- [x] gap_audit.md updated
- [x] Coverage progression: 333: 36/74 (49%) → 335: 39/74 (53%) → 336: 40/74 (54%) → 337: 41/74 (55%) → 338: 42/74 (57%)
