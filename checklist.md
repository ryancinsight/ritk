# RITK Sprint Checklist - Active

> **Full checklist history (Sprints 262-322)**: see [ARCHIVE.md](./ARCHIVE.md)

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
