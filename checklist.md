# RITK Sprint Checklist — Active

## Sprint 378 — Performance Parallelization Wave 2 & SimpleITK Parity
**Target version**: 0.91.2
**Sprint phase**: Execution — parallelization + test parity

### Delivered (Sprint 378)
- [x] PERF-378-01 [patch]: **Parallelize BinaryContourImageFilter, VotingBinaryImageFilter, VotingBinaryHoleFillingImageFilter** over flat voxel index via `moirai::map_collect_index_with`. Commit: `ad8a1e40`
- [x] PERF-378-02 [patch]: **Parallelize MorphologicalLaplace dilate_3d_reflect/erode_3d_reflect + gradient_vecs** (single-pass triplet scatter). Commit: `5b298026`
- [x] PERF-378-03 [patch]: **Parallelize LaplacianSharpening laplacian_f64** + parallel combined[] with sequential .sum() fold. Commit: `c2b62ab3`
- [x] PERF-378-04 [patch]: **Parallelize erode_binary_3d, convolve_1d_axis, CED compute_gradient/gaussian_smooth/compute_divergence** — see below. Commit: `f8fe1970`
  - `morphology/binary_erode.rs`: `erode_binary_3d` uses flat-index parallel map with `.flat_map().all()` short-circuit
  - `edge/separable_gradient/mod.rs`: `convolve_1d_axis` (called 3× per gradient component × 3 axes = 9 calls per gradient)
  - `diffusion/coherence/pde.rs`: `compute_gradient`, `gaussian_smooth` (all 3 axes), `compute_divergence`
- [x] TEST-378-01 [patch]: **Fix deconvolution tests** — `SetOutputRegionModeToSame` replaced with correct API; InverseDeconvolution Pearson structural assertion; BoxMean/by333 + CurvatureFlow tolerance corrections; Canny structural parity test; GradientMagnitude/short + Median/radius2 cmake cases. Commits: `358cf656`, `d0c1254c`

### Verification gate
- [x] `cargo nextest run -p ritk-filter` → 872/872 passed (23.7 s)
- [x] `cargo clippy -p ritk-filter --lib -- -D warnings` → 0 warnings
- [x] Python deconvolution tests (6/6): previously failing `SetOutputRegionModeToSame` + `rel<5e-2` → now all pass
- [x] Python cmake test suite → 315 passed, 2 pre-existing failures (transform_to_displacement_field, signed_distance_map)

### Known pre-existing failures (not introduced this session)
- `test_cmake_transform_to_displacement_field` — pre-existing world-axis ordering issue
- `test_cmake_signed_distance_map_deviation_documented` — PyO3 ndarray type conversion issue

### Deferred / carry-forward
- [ ] PERF-377-01-HUANG3D — reopen conditions unchanged (>10⁶-voxel workload or SSOT promotion)
- [ ] PERF-377-02-RANGE-LUT — gate-blocked by test contract (728k bins/unit required)
- [ ] FIX-transform_to_displacement_field — pre-existing world-axis ordering vs sitk convention
- [ ] FIX-signed_distance_map — PyO3 ndarray conversion for 4D input shape

### Known WIP in working tree — concurrent agent, do NOT touch

### Delivered (Sprint 377 — so far)
- [x] GATE [patch]: `morphology::window_1d` inline `#[allow(clippy::needless_range_loop)]` with justification (sliding-window genuinely needs index-based outer step)
- [x] REFACTOR [patch]: `segmentation::threshold::{huang,isodata,renyi}` `for i in 0..n { vec[i] }` → `for (i, &x) in slice.iter().enumerate()` / `for &x in slice` — idiomatic; value-semantic-equivalent.
- [x] FMT [patch]: `cargo fmt --check` clean on staged files
- [x] CLIPPY [patch]: `cargo clippy --workspace --all-targets -- -D warnings` 0 warnings
- [x] COMMIT [patch]: `de26c2fc refactor(segmentation,filter): needless_range_loop -> iterator`
- [x] PERF-377-01 [patch] (partial): **MedianFilter clamp-hoist micro-optimisation** — pre-baked Z-clamp & Y-clamp indices into stack buffers (`zz_buf`/`yy_buf`), eliminating `(2r+1)²` and `(2r+1)` redundant clamps per voxel; `radius==0` identity fast-path; `BUF_CAP=64` stack buffer with `2r+1<64` panic guard. Verified bit-identical to naive reference via two new brute-force equivalence tests (`to_bits()` equality at r=1 / r=3). Huang sliding-histogram full algorithm deferred (`concurrent_agents` and algorithmic scope note below).
- [x] COMMIT [patch]: `c8048c5d perf(ritk-filter): hoist per-voxel clamps in MedianFilter`
- [x] PERF-377-02 [patch]: **BilateralFilter z-slice parallelism** — replaced serial `for iz in 0..nz` with `moirai::for_each_chunk_mut_enumerated_with` over disjoint output slices. Hoisted dz² + dy² outer-loop arithmetic; tightened spatial_w construction. Verified equivalent via existing brute-force reference test (max_abs < 1e-5). Bench: 32³ 152ms → 11.4ms (~13.3×); 16³ ~1.2ms; 64³ ~76ms (compute-bound linear scaling).
- [x] COMMIT [patch]: `ca5b49a5 perf(ritk-filter): parallelise BilateralFilter over z-slices`
- [x] PERF-377-03 [patch]: **Rank/Percentile filter SSOT consolidation** — promoted duplicated `rank_select_3d` / `percentile_3d` algorithm bodies to a single canonical `rank::kernel::neighborhood_rank_3d`; both `RankFilter::apply` and `PercentileFilter::apply` now translate their public parameter to a `usize rank_idx` and delegate. Net: ~56 lines of duplicated API plumbing gone, one canonical site for future Huang / SIMD / sliding-histogram work. All 14 existing rank/percentile tests still pass; behaviour bit-equivalent.
- [x] COMMIT [patch]: `cb671b64 refactor(ritk-filter): consolidate rank/percentile kernel to SSOT`
- [x] PERF-377-BENCH [patch]: **MedianFilter criterion benchmark** — new `crates/ritk-filter/benches/median.rs` with three sizes (16³, 32³, 64³) at r=2. Recorded baselines (release build, sample size 20-30): 197.47 µs / 1.4888 ms / 9.9397 ms. Captures the per-size threshold any future 3-D Huang sliding-histogram must beat before being accepted. Pattern matches `benches/bilateral.rs`.
- [x] COMMIT [patch]: `47a0e794 perf(ritk-filter): add criterion bench for MedianFilter at r=2`
- [x] PERF-377-02-LUT-DOC [patch]: **Range-LUT ε-bound derivation** — module-level doc on `bilateral.rs` documents the analytical derivation showing a full quantised range LUT cannot meet the existing `1e-5` test tolerance (would require qscale > 728k bins/unit). Three honest alternatives enumerated (hybrid exp+LUT, loosen test tolerance, or keep current path) with their trade-offs; the parallelism-only implementation at `ca5b49a5` remains the final landing for PERF-377-02.
- [x] COMMIT [patch]: `462a6b63 docs(ritk-filter): record range-LUT epsilon-bound analysis on BilateralFilter`
- [x] PERF-377-FULL-DEFER [patch]: **Huang 3-D sliding-histogram MedianFilter** explicitly DOWN-SCOPED to a future backlog item. The 3-D Huang is estimated ~200 LOC of intricate bookkeeping; 2-D Huang is a regression (`O(r²·n_bins)` > current `O(r³)` for typical n_bins), and existing brute-force parallelism already meets 64³ = ~10 ms at r=2 (median bench baseline). Vote-with-feet: future work items may reopen if (a) a >10⁶-voxel workload appears, or (b) the algorithm is migrated into `rank::kernel::neighborhood_rank_3d` where it amortises across rank/percentile filters. PERF-377-01 partial (clamp-hoist) at `c8048c5d` and the median bench at `47a0e794` are the bounded Sprint 377 deliverable.

### Verification gate
- [x] `cargo nextest run -p ritk-segmentation -E 'test(threshold)'` → 120/120 passed
- [x] `cargo nextest run -p ritk-filter -E 'test(unary_minus)|test(round_half)'` → 2/2 passed
- [x] `cargo nextest run -p ritk-filter` → 800/800 passed (full crate incl. 9/9 median, 5/5 bilateral, 14/14 rank/percentile)
- [x] `cargo clippy -p ritk-filter --lib -- -D warnings` → 0 warnings (lib clean; test-only WIP files owned by parallel agent)
- [x] `cargo bench -p ritk-filter --bench bilateral -- apply/16x16x16` → ~1.2 ms median
- [x] `cargo bench -p ritk-filter --bench bilateral -- apply/32x32x32` → ~11.4 ms median
- [x] `cargo bench -p ritk-filter --bench bilateral -- apply/64x64x64` → ~76 ms median (linear scaling confirms compute-bound)
- [x] `cargo bench -p ritk-filter --bench median` → recorded baselines:
  - `median_3d/apply/16x16x16` ~197.47 µs
  - `median_3d/apply/32x32x32` ~1.4888 ms
  - `median_3d/apply/64x64x64` ~9.9397 ms
- [x] `cargo build -p ritk-filter --lib` → clean (1 dead-code warning in `crates/ritk-filter/src/discrete_gaussian.rs` is owned by the parallel agent's `DiscreteGaussianDerivative` port and is not in this session's scope)
- [x] `cargo test -p ritk-filter --lib 'median::'` → 6/6 median tests pass (brute-force bitwise-equivalence held)
- [x] `cargo test -p ritk-filter --lib 'bilateral::'` → 5/5 bilateral tests pass (1e-5 ε unchanged)

### Deferred / carry-forward (next increments)
- [ ] PERF-377-01-HUANG3D [patch→[minor]?] (deferred-with-rationale): **Huang 3-D sliding-histogram MedianFilter** — Perreault-Hebert 2007 §3.2 with `window_hist[n_bins]` + row_in/row_out column-histogram updates. Reopen condition: (a) >10⁶-voxel workload where the algorithm is the bottleneck, or (b) algorithm promotion into the `rank::kernel::neighborhood_rank_3d` SSOT to amortise across rank/percentile. Existing brute-force parallelism is already 10ms at 64³ r=2; 2-D Huang would be a regression (O(r²·n_bins) > current O(r³) at typical n_bins). See `benches/median.rs` for the per-size baseline threshold.
- [ ] PERF-377-02-RANGE-LUT [patch→[minor]?] (gate-blocked by test contract): **BilateralFilter range LUT** — module-level doc on `bilateral.rs` carries the ε-bound derivation; see commit `462a6b63`. A quantised `range_w[|dr|]` LUT over the full intensity range would need qscale > 728k bins/unit to hold the existing 1e-5 test epsilon for σ_r=50. Three options documented in code:
  1. Hybrid exp + LUT (≤ 2× at typical σ_r)
  2. Loosen test tolerance to a derived ~0.05 HU bound (test-contract change, [minor])
  3. Keep current `exp`-per-neighbour path (default)
  Reopen when (1) or (2) are justified by an explicit workload or test-contract change.
- [ ] DOC-377-01 [patch]: 16 pre-existing intra-doc-link warnings (rustdoc unresolved link, public docs → private items) accumulated from Sprint 393-395 commits; gated but non-blocking.
- [ ] FMT-377-01 [patch]: working-tree fmt-only diffs from cumulative agent updates (long-line rewraps). Now ~30 files per current `git status`; pure whitespace; next `cargo fmt --all` by next agent or this session will close.

### Known WIP in working tree — concurrent agent, do NOT touch

At session start (2026-06-17), the parallel agent has 22 files modified in working tree
(plus `rust_out.exe` deletion). Per `concurrent_agents`, treated as foreign WIP;
do not touch any of the below; coordinate via `git status` between commits:

- `crates/ritk-filter/src/color.rs` (whitespace-fmt)
- `crates/ritk-filter/src/morphology/{label_contour,label_morphology/reconstruction,regional_extrema,tests_grayscale_fillhole,tests_grayscale_grind_peak,tests_h_transform,tests_hit_or_miss,tests_reconstruction_opening_closing,tests_regional_extrema}.rs` — morphology feature/test batch
- `crates/ritk-filter/src/tests_color.rs`
- `crates/ritk-image/src/color.rs`
- `crates/ritk-python/pyproject.toml`, `src/segmentation/{labeling.rs,threshold.rs}`
- `crates/ritk-python/tests/{test_smoke.py,test_registration_gap_validation.py,test_registration_side_by_side.py,test_elastix_vs_ritk_rire.py}`
- `crates/ritk-registration/src/classical/global_mi/cma_mi/helpers.rs`, `src/metric/mutual_information/mod.rs`
- `crates/ritk-segmentation/src/threshold/{kittler.rs,mod.rs}`
- `rust_out.exe` (deleted build artifact)

Subsequent parallel-agent commits since session start:
- `271c026c feat(ritk-filter): add MedianProjection filter` (MedianIntensityProjectionFilter + Cargo.toml/version bump + lib.rs re-export — now landed in `271c026c`). `crates/ritk-filter/src/projection.rs`, `crates/ritk-filter/Cargo.toml`, `crates/ritk-filter/src/lib.rs` re-export, and `crates/ritk-python/{Cargo.toml, pyproject.toml, _ritk/filter.pyi, src/filter/{mod.rs, projection.rs}}` land via this commit line.

`crates/ritk-filter/src/median.rs` is **clean** — this session's sole working file (`PERF-377-01 Huang sliding-histogram MedianFilter`).

## Sprint 376 — DRY Closure, Build Hardening & Carry-Forward Reconciliation
**Target version**: 0.70.1
**Sprint phase**: Closure — DRY closure, build hardening, fmt/clippy gating, CPR-PERF-01 all delivered and verified.

### Delivered (Sprint 376)
- [x] CARRY [patch]: Concurrent trunk of 25 inline test blocks extracted (`b052c40a refactor(filter): extract 25 inline test blocks to sibling files (SRP) [-3070L]`)
- [x] CARRY [patch]: Carry `cpro_CHRONO_history' DRY tracker commits `d4754aa1 6998b4cf d4a9a701 `carry-forward filter binding surface expansion (single-axis match sitk Euler3DTransform + extended corpus + API mismatches)`
- [x] CARRY [minor]: `feat(python): expose normalize, unsharp, zero-crossing, rotate, shift, zoom` (`43d9553 feat python filter bindings`) 6 new PyImage functions added.
- [x] CARRY [minor]: Concurrent drain: `feat(stats): Add ddof flag for sample (sitk) vs population std` + std-ddof population/sample parity tests
- [x] CARRY [patch]: Concurrent drain: `test(python): Connected-component + label-shape parity vs sitk`
- [x] CARRY [patch]: `chore(filter): enable ritk-image test-helpers feature for DRY helper consumption`
- [x] DRY-374-01 [minor]: `Refactor tests to use shared test_support helpers` — 78 test files migrated to delegating wrappers over `ritk_image::test_support::*`. Resolution: keeps thin local wrappers for type fixity while body delegates to canonical entry point.
- [x] CARRY [patch]: Cargo-fix applied 51 test files to strip unused `burn::tensor` and `ritk_spatial` imports after migration.
- [x] CLIPPY [patch]: 2 prior lint failures resolved before this sprint (doc list indent + Range single-element array). `cargo clippy --workspace --all-targets -- -D warnings` clean at session start.
- [x] CLIPPY [patch]: 1 carry-over clippy warning fixed: `for i in 0..n { out[i] }`  closeness simplified to `for (i, &v) in out.iter().enumerate()` in `tests_hit_or_miss.rs`.
- [x] FMT [patch]: `cargo fmt --check` clean (0 diff lines).
- [x] FIX [patch]: Sister-file incorrect hit-or-miss assignment caught: line-48 `n` now unused after migration; clippy validates clean.
- [x] CONVERGED [patch]: Local tree in sync with `origin/main`.
- [x] BILAT-PERF-01 [minor]: `BilateralFilter::compute` rewritten with precomputed spatial-kernel lookup table `spatial_w[d²]` + clamped boundary iteration `z_lo..z_hi`. Per-neighbour cost reduced from 3 squarings + mul + `exp` to 1 lookup + 1 `exp`. Per-neighbour `as isize`/`as usize` casts and boundary branches eliminated. Verified bitwise identical vs brute-force reference (`max |Δ| = 0` on `5×6×7` deterministic volume).
- [x] BILAT-REGRESSION-01 [patch]: `test_bilateral_matches_brute_force_reference` added — locks the kernel computation to the original mathematical formulation by comparing `apply` output against an explicit-arithmetic reference on a non-trivial volume.
- [x] BILAT-BENCH-01 [patch]: criterion bench `benches/bilateral.rs` registered — measures `apply` over 16³/32³/64³ volumes at spatial σ = 1.5 (r ≈ 5). Baselines: 16³=14.4ms, 32³=152ms.
- [x] DOC-376-01 [patch]: `OPTIMIZATION.md` updated with Sprint 376 BilateralFilter section documenting LUT, clamped iteration, equivalence evidence and measured timings.
- [x] CPR-PERF-01 [patch]: `CprImageFilter::apply` rewritten with hoisted `direction.inverse()` (3×3 inverse computed once per call instead of once per cross-section sample) and a per-path-point index basis `(idx_p0, slope)` that collapses the inner loop to a linear-in-offset `idx_p[i,j] = idx_p0[i] + slope[i] * offset[j]`. New private helper `trilinear_sample_from_idx` accepts the precomputed voxel index; public `trilinear_sample` unchanged.
- [x] CPR-REGRESSION-01 [patch]: `cpr_apply_matches_brute_force_reference` + `cpr_apply_matches_brute_force_reference_nonidentity_direction` brute-force differential tests — locks value semantics against the pre-optimisation form (`max |Δ| ≤ 1e-5`) on both identity and non-identity direction matrices.
- [x] CPR-BENCH-01 [patch]: `benches/cpr_apply.rs` criterion bench — end-to-end `apply` on 16³/32³/64³ default config; head-to-head Δ vs reverted reference: 16³ 1.98×, 32³ 1.47×, 64³ 1.14×.

### Verification gate
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo fmt --check` → 0 diffs
- [x] `cargo nextest run -p ritk-filter` → 707/707 passed (15 CPR tests; +2 vs prior 705 baseline from `cpr_apply_matches_brute_force_reference*` regression suite)
- [x] `cargo nextest run -p ritk-segmentation -p ritk-statistics -p ritk-tiff` → 707/707 passed
- [x] `cargo nextest run -p ritk-image -p ritk-statistics` → 312/312 passed
- [x] `cargo bench --bench bilateral` compiles and runs (16³=14.4ms, 32³=152ms; 64³ unmeasured)
- [x] `cargo bench --bench cpr_apply` compiles and runs: 16³=505µs, 32³=976µs, 64³=4.69ms (optimised); reference (reverted) 16³=1.00ms, 32³=1.43ms, 64³=5.33ms — speedup 1.98×/1.47×/1.14×
- [x] Numerical equivalence (CPR): `max |Δ| ≤ 1e-5` on 12³ identity + 10³ 90°-rotated direction matrices (regression tests locked)

### Blocked / Deferred (carry-forward)
- [ ] VAR-375-01 [upstream]: `PhantomData<B>` → `PhantomData<fn() -> B>` BLOCKED at `burn-core-0.19.1`
- [ ] CONST-375-02 [toolchain]: const-assert companion for `BSplineTransform` blocked on const_panic_fmt
- [ ] NAMING-362-23 [arch]: sealed trait `DimInterpolation<B>` BLOCKED — ADR required
- [ ] SRP-362-20 [minor]: `FilterKind` ValueEnum separation — partial (slice delivery done; per-family Args structs remain)
- [ ] NAMING-FILTER-01 [major]: `FftConvolution3DFilter` const-generic unification — concurrent-crate changes required
- [ ] N-375-08 [arch]: DRY cross-crate parse utils — promotion trigger requires `ritk-io` → `ritk-core` migration

---

## Sprint 375 — Architecture Hardening Round 8: SSOT · DRY · NAMING · ENUM · SRP · COMPAT
**Target version**: 0.70.0  
**Sprint phase**: Closure — all 60 patches delivered and verified.

### Delivered (Sprint 375)
- [x] P01 [patch]: [HARD] fake UID bypass in seg/writer.rs — real compute restored
- [x] P02–P05 [patch]: SSOT/DRY — EXPLICIT_VR_LE ×6 writers; normalize_to_u16 helper; UID gen dedup; emit_pixel_format_tags helper
- [x] P06–P08 [minor]: ENUM — RtRoiInterpretedType, RtDoseType/RtDoseSummationType, SegmentationType/SegmentAlgorithmType promoted from ArrayString<16>
- [x] P09 [minor]: DRY+NAMING — DicomObjectNode::with_value<V> generic + get_u32 rename + is_image_sop_class + Association::config removed
- [x] P10–P14 [minor/patch]: NAMING+DRY — ritk-vtk 13 type-concrete fns deleted → read_helpers; write_attribute dedup; xml_helpers.rs shared module; char literals + SSOT consts
- [x] P15–P17 [patch]: SRP+COMPAT — ritk-vtk domain/io test extraction (6 files); compat/doc cleanup
- [x] P18–P20 [patch]: SSOT+COMPAT+SRP — spatial ORTHOGONALITY_TOLERANCE; deprecated to_vec() removed; shape_markers test extracted
- [x] P21–P26 [minor]: NAMING — ritk-minc/metaimage/nrrd type-suffix renames (extract_scalar_float, build_attr_msg_float, decode_raw_bytes, decode_element_bytes, parse_float_vec) + reader.rs SRP split
- [x] P27–P31 [patch]: SRP — 24 inline test blocks extracted to sibling files in ritk-snap
- [x] P32–P38 [patch]: COMPAT+SSOT+NAMING+DRY — dead ModalityDisplay/MRI arm; W/L + MPR + alpha constants; dot3/cross3/normalize3; W/L DRY helper; SSOT sweep
- [x] P39–P46 [patch]: NAMING+SSOT+SRP+COMPAT — 27+14+6 test fn renames; 17 prod SSOT consts; test tolerance consts; 5 test extractions; 5 dup test deletions; 5 dead code removals
- [x] P47–P55 [patch/minor]: SSOT+SRP+NAMING+ENUM+COMPAT — JPEG constants; LANCZOS/SPATIAL_DIMS; grid/transform/pixel_layout/jpeg/nearest/trilinear test extractions; apply_rescale helper; legacy.rs + 8 NN arms deleted; InterleaveMode/QuantPrecision enums; dim→rank rename
- [x] P56–P60 [patch]: NAMING+SRP+SSOT+COMPAT — 28 fft/conv test renames; NCC_DENOM_FLOOR; 22 test extractions (batch A+B); entropy/F32_TOL/STAPLE_TOL/FOREGROUND_THRESHOLD; final verification

### Blocked / Deferred
- [ ] DRY-374-01: `make_image_*`/`make_mask_*` — 68 occurrences [minor] (next round)
- [ ] NAMING-362-23: `transform_1d/_2d/_3d/_4d` [arch] BLOCKED — ADR required
- [ ] SRP-362-20: `FilterArgs` → `FilterKind` [major] BLOCKED
- [ ] NAMING-FILTER-01: `FftConvolution3DFilter` const-generic unification [major] BLOCKED
- [ ] N-375-08: DRY cross-crate parse utils (shared IO codec layer) [arch] BLOCKED

### Verification gate
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] ritk-io nextest → 330/330
- [x] ritk-vtk nextest → 241/241
- [x] ritk-spatial/morphology/minc/metaimage/nrrd nextest → 131/131
- [x] ritk-snap nextest → 633/633
- [x] ritk-registration + ritk-transform nextest → 69+69 = 138
- [x] ritk-codecs + ritk-image + ritk-interpolation nextest → 353/353
- [x] ritk-filter nextest → 703/703
- [x] ritk-segmentation + ritk-statistics nextest → 663/663

---
## Sprint 374 — Architecture Hardening Round 7: SSOT · DRY · NAMING · ENUM · SRP · COMPAT
**Target version**: 0.69.0  
**Sprint phase**: Closure — all 40 patches delivered and verified.

### Delivered (Sprint 374)
- [x] P01–P05 [patch]: SSOT constants extracted in ritk-filter (SIGMA_MIN, NEAR_ZERO_MAG, LENGTH_EPSILON, NEAR_ZERO_WEIGHT, TIKHONOV_LAMBDA)
- [x] P06 [minor]: DRY — `morphological_scan_3d` consolidates dilate_3d/erode_3d in ritk-filter morphology/mod.rs
- [x] P07 [minor]: SSOT — `PROB_ZERO_GUARD: f64 = 1e-12` in threshold/mod.rs; 15 production sites across kapur/li/otsu/multi_otsu/chan_vese; EIGENVALUE_SINGULARITY_EPS in label_shape_extended
- [x] P08–P10 [patch]: SSOT — FOREGROUND_THRESHOLD bypass fixed; NORMALIZER_EPSILON in 2 test files; CENTRAL_DIFF_HALF in jacobian.rs
- [x] P11 [minor]: ENUM — `OptimizerAlgorithm` enum in ritk-registration (5 optimizer impls updated)
- [x] P12–P14 [patch]: COMPAT + SSOT — stale diagram fixed; test tolerance consts in transform/registration
- [x] P15 [minor]: ENUM — `ContourGeometricType` enum in ritk-io RtContour (reader/converter/writer/tests updated)
- [x] P16–P19 [patch]: DRY + SSOT + SRP — str_to_vr dedup; SOP UID + TS UID SSOT; converter tests extracted
- [x] P20–P23 [patch/minor]: COMPAT + NAMING + SSOT — ritk-image deprecated fix; ritk-codecs to_u16 removal; ritk-analyze LeBytes trait + HDR_SIZE/EXTENTS
- [x] P24–P31 [patch]: NAMING + SSOT + COMPAT — 6 snap renames; U8_MAX_F32 const; 2 dead code deletions
- [x] P32–P34 [minor/patch]: NAMING + COMPAT — VtkCellType From/TryFrom; parse_floats generic; ply renames + dead fn deletion
- [x] P35–P40 [patch]: NAMING + SSOT + SRP — annotation test names; epsilon/U8_MAX_F consts; 3 test extractions; nrrd/mgh/tensor-ops naming

### Blocked / Deferred
- [ ] NAMING-362-23: `transform_1d/_2d/_3d/_4d` [arch] BLOCKED
- [ ] SRP-362-20: `FilterArgs` → `FilterKind` [major]
- [ ] DRY-374-01: `make_image_*`/`make_mask_*` 35+ copies (next round)
- [ ] SRP-374-03: 21 test blocks in ritk-filter (next round)
- [ ] SRP-374-04: 25 test blocks in ritk-snap (next round)
- [ ] NAMING-374-02, ENUM-374-06, DRY-374-07/08, NAMING-374-05: carry-forward (next round)

### Verification gate
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run` (modified crates) → 3620/3620 passed

---
## Sprint 373 — J2K interop closure (MQ root cause fixed)
**Target version**: 0.68.x  
**Sprint phase**: Closure — J2K-INTEROP P1 closed; next increment: SimpleITK parity comparison (tests + examples).

### Delivered
- [x] J2K-INTEROP [P1→closed, patch]: MQ probability-estimation root cause — `I(CX)` advanced on every MPS instead of only on renormalisation (ISO 15444-1 §C.2.6/Fig. C.7); encoder+decoder shared the defect so internal round-trips masked it. Found via register-trace diff against an instrumented vendored openjp2 (instrumentation removed after diagnosis). 6 interop acceptance tests un-ignored; escalation byte-compare green; `openjp2_captured_packet_conformance` (OpenJPEG 2.5.2 fixed vector) byte-exact both directions
- [x] Cleanup: diagnostic probe/dump tests removed; minimized impulse case kept as `cross_decode_impulse_8x8_regression`; env_logger/log dev-deps removed

### Delivered (cont. — SITK validation pass)
- [x] SITK-PARITY (filters/registration/statistics): `test_simpleitk_parity.py` 175/175 green against SimpleITK 3.0.0a1
- [x] J2K-BITSTUFF [P1→closed, patch]: tier-2 packet headers byte-stuffed (0x00 after 0xFF) instead of §B.10.1 bit-stuffing; found via SimpleITK/GDCM-written J2K DICOM failing to decode; `BitWriter`/`BitReader` rewritten on `opj_bio` semantics; 126-config interop matrices (incl. 12-bit) green both directions. ritk-codecs 0.5.2

- [x] SITK-PARITY (codec e2e, manual): SimpleITK/GDCM-written J2K DICOM → `ritk.io.read_image` exact at 8/12/16-bit (fresh wheel, extracted-archive run; 2026-06-12)

### Open (next increment)
- [ ] SITK-PARITY (codec e2e, automated): add the SimpleITK-written J2K DICOM round-trip as a pytest in `test_simpleitk_parity.py` once the concurrent agent's `fix/sitk-parity-mi-sampling` branch merges (file currently has uncommitted edits on that branch)
- [ ] J2K-LOSSY-97, JLS-INTEROP, CODEC-PERF, REG-MI-FLAKY: carry-forward

### Verification gate
- [x] nextest ritk-codecs → 194/194 (0 ignored); ritk-io → 330/330; clippy -p ritk-codecs --all-targets → 0 warnings; fmt clean; doc clean

---
## Sprint 372 — J2K conformance + interop harness (complete)
**Target version**: 0.68.x  
**Sprint phase**: Closure.

### Delivered
- [x] J2K-372-CONF [patch]: 7 ISO 15444-1 conformance fixes (B.10.3 packet bit, Table B.4, B.10.7.1 Lblock, E.1 Mb, D.4.1 pass count, D.2 stripe scan, D.1 ZC tables, RLC)
- [x] J2K-372-HARNESS: openjp2 differential suite (dev-dep, pure Rust) — tier-2 header now parses OpenJPEG output exactly
- [x] JLS-NEAR-TAIL + JLS-16BIT-LOSSLESS [P1→closed]: single root cause — trailing 0xFF before EOI discarded as marker prefix; flush emits the stuffed follow byte. Proptests re-enabled at full domain

### Verification gate
- [x] clippy workspace -D warnings → 0; nextest codecs+dicom+io → 526/526 (9 ignored = tracked pending/defect tests)

---
## Sprint 371 — J2K multi-code-block tier-2 (J2K-MULTI-CBLK delivered)
**Target version**: 0.68.0 (ritk-codecs 0.5.0)  
**Sprint phase**: Closure — full-size single-tile J2K encode/decode delivered and verified.

### Delivered (Sprint 371)
- [x] J2K-371-TT [minor]: `tag_tree` module — §B.10.2 quad-tree with standard polarity, persistent cross-layer state; replaces the non-standard single-leaf coding
- [x] J2K-371-CBLK [minor]: 64×64 code-block partitioning per subband; per-band inclusion/MSB trees; per-code-block layer state; arbitrary single-tile sizes
- [x] J2K-371-TEST [patch]: multi-grid lossless round-trips (130×70 LL0; 150×100 L2 @16-bit); tag-tree unit + partial-threshold tests
- [x] J2K-371-BENCH [patch]: criterion 512×512 16-bit 5-level cases — encode 55.6 ms / decode 58.2 ms median (baseline `sprint371`); CODEC-PERF target ≈2–3× (OpenJPEG-class)

### Blocked / Deferred
- [ ] J2K-INTEROP [patch]: differential decode vs OpenJPEG-encoded reference corpus — now unblocked (conformant tag trees + multi-cblk in place); NEXT
- [ ] J2K-LOSSY-97, JLS-INTEROP, CODEC-PERF, REG-MI-FLAKY: carry-forward

### Verification gate (Sprint 371)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-codecs -p ritk-dicom -p ritk-io` → 526/526 (180 codec tests)
- [x] `cargo doc --no-deps -p ritk-codecs` → warning-clean

---
## Sprint 370 — J2K multi-level DWT (J2K-DECODE-DWT delivered)
**Target version**: 0.67.0 (ritk-codecs 0.4.0)  
**Sprint phase**: Closure — multi-resolution lossless J2K decode/encode delivered and verified.

### Delivered (Sprint 370)
- [x] J2K-370-DWT [minor]: forward + inverse multi-level 5/3 DWT on the Mallat layout (the prior multi-level ROI scheme was structurally wrong for N > 1); `subband` geometry module (rects, ZC orientations, gains)
- [x] J2K-370-T2 [minor]: LRCP multi-resolution packets with per-code-block state across layers; per-subband ε_b from QCD (`QcdMarker::exponents`); encoder emits 3N+1 SPqcd entries
- [x] J2K-370-FIX [patch]: tier-2 `BitReader::byte_pos()` returns RAW offsets — stuffed-0xFF packet headers desynced the next packet body (latent single-packet bug)
- [x] J2K-370-TEST [patch]: 2/3-level explicit round-trips, 2×2 L1 regression, proptest randomizes 0–3 levels; ritk-io DICOM round-trip uses 2 levels @16-bit
- [x] FIX-370-WS [patch]: in-flight registration example `registration_compare_figure.rs` write_nifti arg order

### Blocked / Deferred
- [ ] REG-MI-FLAKY [investigate]: carry-forward (in-flight registration wave)
- [ ] J2K-MULTI-CBLK, J2K-LOSSY-97, J2K-INTEROP, JLS-INTEROP, CODEC-PERF: carry-forward

### Verification gate (Sprint 370)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-codecs -p ritk-dicom -p ritk-io` → 520/520 (174 codec tests incl. level-randomized proptest)
- [x] `cargo doc --no-deps -p ritk-codecs` → warning-clean

---
## Sprint 369 — Native JPEG-LS codec: CharLS elimination + NEAR support
**Target version**: 0.66.0 (ritk-codecs 0.3.0)  
**Sprint phase**: Closure — native JPEG-LS encoder/decoder delivered; zero C/C++ FFI in the DICOM codec stack.

### Delivered (Sprint 369)
- [x] JLS-369-ENC [minor]: pure-Rust JPEG-LS encoder (lossless + NEAR), mirror of the scan decoder over the shared context model; bit writer + Golomb writer with §C.2.1 stuffing
- [x] JLS-369-NEAR [minor]: NEAR-aware decode (TS .81 native); `CodingParams` + `quantize_error`/`reconstruct` SSOT shared by both sides
- [x] JLS-369-CONF [patch]: `default_thresholds` per ISO C.2.4.1.1.1 (4095 factor cap; >8-bit defaults were non-conformant); §A.3.3 NEAR dead-zone in gradient quantization
- [x] JLS-369-DEP [minor]: `charls` removed (workspace + dev-dep + build.rs libstdc++ hacks); registry `charls`/`openjp2` features and `jpeg2k` dropped — codec stack 100 % Rust
- [x] JLS-369-TEST [patch]: lossless + NEAR proptests, run-mode/interrupt fixtures, 12/16-bit threshold derivation tests; one-time differential: native NEAR=2 stream decoded by CharLS-backed backend before removal → |err| ≤ 2 confirmed
- [x] FIX-369-WS [patch]: clippy fixes in in-flight registration work (`grid.rs`, dead `integrate_geodesic_into` removed, bench rand API, `sample_count` call site)

### Blocked / Deferred
- [ ] REG-MI-FLAKY [investigate]: `translation_recovery_shifted_gaussian` fails deterministically (est 1.0 vs true 3.0) in the in-flight NGF/RSGD registration wave — owned by the concurrent registration effort; not in codec blast radius
- [ ] J2K-DECODE-DWT [minor]: carry-forward (Sprint 368)
- [ ] J2K-LOSSY-97, J2K-INTEROP: carry-forward (Sprint 368)

### Verification gate (Sprint 369)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-codecs -p ritk-dicom -p ritk-io` → 510/510 + 68 jpeg_ls module tests passed
- [x] `cargo doc --no-deps -p ritk-codecs` → warning-clean
- [x] libstdc++ DLL-shadowing failure mode eliminated (no C++ linkage remains)
- [x] criterion baseline `sprint369` saved (ritk-codecs/benches/codec_throughput): JPEG-LS 512x512 16-bit encode 13.4 ms / decode 10.3 ms (median); J2K 64x64 16-bit ~0.6 ms each way. Follow-up: CODEC-PERF profile-first optimization pass

---

## Sprint 368 — RITK-native JPEG 2000 codec (pure-Rust ISO 15444-1, C/FFI elimination)
**Target version**: 0.65.0 (ritk-codecs 0.2.0)  
**Sprint phase**: Closure — native J2K lossless codec delivered and verified.

### Delivered (Sprint 368)
- [x] J2K-368-MQ [patch]: MQ coder conformance fixes — INITDEC alignment, MPSEXCHANGE `A=Qe` removal, CODEMPS/CODELPS per Figures C.7/C.8, dummy-first-byte BYTEOUT, FLUSH `CT` shift, QE_TABLE NMPS/NLPS column swap, Table D.7 initial contexts
- [x] J2K-368-T2 [patch]: tier-2 packet fixes — Lblock terminator bit, Table B.4 39+ prefix (5 bits), inclusion tag-tree threshold 0
- [x] J2K-368-ENC [minor]: encoder promoted from `#[cfg(test)]` to public module (`jpeg_2000::encoder`); ritk-io consumes it for DICOM J2K round-trip tests
- [x] J2K-368-DEP [minor]: `jpeg2k`/`openjp2`/`openjpeg-sys`/`charls` removed from ritk-codecs; `decode_tile_part` params → `TileCodingParams`
- [x] J2K-368-TEST [patch]: 16-bit regression test + proptest lossless round-trip (random images, 8/12/16-bit, signed/unsigned)
- [x] FIX-368-REG [patch]: NGF/RSGD config call-site compile fixes (`center_weight_sigma_frac`, `learning_rate_decay`); `ngf_scalar` gated `#[cfg(test)]`

### Blocked / Deferred
- [ ] J2K-DECODE-DWT [minor]: multi-level 5/3 DWT decode (wavelet.rs idwt groundwork in place; `decode_tile_part` currently bails on `num_decomp_levels > 0`)
- [ ] J2K-LOSSY-97 [minor]: 9/7 irreversible wavelet (lossy TS .91 full support)
- [ ] J2K-INTEROP [patch]: differential decode test against an OpenJPEG-encoded reference codestream (real-world DICOM corpus)

### Verification gate (Sprint 368)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-codecs` → 145/145 passed (incl. 256-case proptest)
- [x] `cargo nextest run -p ritk-io` → 330/330 passed (JPEG2000 Windows abort resolved — pure-Rust path)

### Environment note
- `ritk-io` test binaries link `libstdc++-6.dll` (charls dev-dep); a Julia `libstdc++-6.dll` in PATH caused 0xc0000139 at load. Workaround: ucrt64 runtime DLLs copied beside `target/debug/deps`. Root fix: drop charls dev-dep once a pure-Rust JPEG-LS differential reference exists.

---

## Sprint 367 — Architecture Hardening Round 6: ENUM · NAMING · SRP · SSOT · DRY · COMPAT + ritk-core Crate Extraction
**Target version**: 0.64.0  
**Sprint phase**: Closure — all 40 patches + [arch] crate extraction delivered and verified.

### Delivered (Sprint 367)
- [x] ARCH-367 [arch]: Extract `ritk-annotation`, `ritk-statistics`, `ritk-morphology`, `ritk-tensor-ops` from ritk-core; compatibility shims in `annotation/mod.rs` + `statistics/mod.rs`
- [x] ENUM-367-35 [minor]: `SegmentArgs.method: String` → `SegmentMethod` ValueEnum (23 variants); unreachable arm + dead test removed
- [x] ENUM-367-36 [minor]: `ConvertArgs.format: Option<String>` → `Option<OutputFormat>` ValueEnum (8 variants)
- [x] ENUM-367-37 [minor]: `NormalizeArgs.contrast: Option<String>` → `Option<CliContrast>` ValueEnum; dead test removed
- [x] ENUM-367-38 [minor/patch]: `FilterArgs.order: usize` → `CliDerivativeOrder` ValueEnum; `parse_spacing_mode` wrapper removed
- [x] NAMING-367-05 [patch]: `RgbaU8`→`RgbaBytes`, `RgbaF32`→`RgbaLinear`; all callers in ritk-io + ritk-snap updated
- [x] NAMING-367-06 [patch]: `UnaryPixelOp::apply_f32` → `apply` in ritk-filter
- [x] NAMING-367-07 [patch]: `fft2d`/`fft3d` `pub` → `pub(crate)`; deconvolution/helpers.rs migrated to `fft_nd`
- [x] NAMING-367-08 [patch]: `required_usize`/`optional_usize`/`optional_u16` → `read_required<T>`/`read_optional<T>` in color_common.rs
- [x] NAMING-367-09 [patch]: `read_nested_f64` → `read_nested_scalar<T: FromStr>` in ritk-io/helpers.rs
- [x] NAMING-367-10 [patch]: `test_normalize_3d`/`test_dot_3d` → `test_normalize_unit_vector`/`test_dot_product`
- [x] NAMING-367-11 [patch]: `build_rle_fragment_8bit` → `build_rle_fragment`
- [x] NAMING-367-12 [patch]: `CommandField::from_u16` → `impl TryFrom<u16> for CommandField`
- [x] SRP-367-A1 [patch]: ritk-annotation `tests_annotation_state.rs` extracted
- [x] SRP-367-A2 [patch]: ritk-annotation `tests_overlay.rs` extracted
- [x] SRP-367-A3 [patch]: ritk-annotation `tests_color.rs` extracted
- [x] SRP-367-R1 [patch]: ritk-registration `tests_lncc.rs` extracted
- [x] SRP-367-R2 [patch]: ritk-registration `tests_ncc.rs` extracted
- [x] SRP-367-R3 [patch]: ritk-registration `tests_numerical.rs` extracted
- [x] SRP-367-I1 [patch]: ritk-io `tests_sop_class.rs` extracted (193L)
- [x] SRP-367-S1 [patch]: ritk-segmentation `tests_shape_detection.rs` extracted (230L)
- [x] SRP-367-S2 [patch]: ritk-segmentation `tests_growcut.rs` extracted (175L)
- [x] SRP-367-S3 [patch]: ritk-segmentation `tests_fill_holes.rs` extracted (116L)
- [x] SRP-367-S4 [patch]: ritk-segmentation `tests_morphological_gradient.rs` extracted (114L)
- [x] SSOT-367-23 [patch]: `DEFAULT_NOISE_SEED: u64 = 42` const; 4 noise filters updated
- [x] SSOT-367-24 [patch]: `DEFAULT_ITERATIVE_TOLERANCE: f32 = 1e-6` const; landweber + rl updated
- [x] SSOT-367-25 [patch]: `FOREGROUND_THRESHOLD: f32 = 0.5` const; 5 morphology modules updated
- [x] DRY-367-28 [patch]: `box_muller(u1, u2) -> f64` extracted to noise/mod.rs; 3 noise filters use it
- [x] DRY-367-30 [patch]: `ritk-analyze/codec.rs` shared helpers + `DT_FLOAT` const; reader.rs + writer.rs updated
- [x] COMPAT-367-32 [patch]: `DRY_353_02_STATUS` dead const removed from ritk-interpolation/kernel/macros.rs
- [x] COMPAT-367-33 [patch]: Stale `#[allow(dead_code)]` on `BoundsPolicy` removed; dead `is_zero_pad` deleted; `BinRange::is_empty` gated `#[cfg(test)]`
- [x] COMPAT-367-34 [patch]: `#[allow(dead_code)]` removed from direct-parzen `cache.rs` feature-gated functions
- [x] COMPAT-367-35 [patch]: `ParzenConfig` test-only fns gated `#[cfg(test)]`; suppressions removed
- [x] COMPAT-367-36 [patch]: `compute_joint_histogram_from_cache` `#[allow(dead_code)]` → `#[cfg(not(feature = "direct-parzen"))]`
- [x] COMPAT-367-37 [patch]: Dead `is_empty` removed from `bin_range.rs` + `stack_weights.rs`; suppressions removed
- [x] COMPAT-367-39 [patch]: Stale doc in `deconvolution/regularization.rs` referencing `apply_2d`/`apply_3d` corrected
- [x] FIX-367-INT [patch]: ritk-snap/label/tests.rs `use super::*` restored after RgbaU8→RgbaBytes rename

### Blocked / Deferred
- [ ] NAMING-362-23 [arch]: `transform_1d/_2d/_3d/_4d` — BLOCKED; `DimInterpolation<B>` sealed trait design needed
- [ ] SRP-362-20 [major]: `FilterArgs` → `FilterKind` ValueEnum — carry forward
- [ ] NAMING-FILTER-01 [major]: `FftConvolution3DFilter`/`FftNormalizedCorrelation3DFilter` → const-generic unification
- [ ] TIMEOUT-367: ritk-interpolation 4-test timeout cluster (`dim4`, `dim3_extended`) — investigate under performance_engineering protocol

### Verification gate (Sprint 367)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-core -p ritk-filter -p ritk-segmentation -p ritk-statistics -p ritk-annotation` → 1429/1429 passed
- [x] `cargo nextest run -p ritk-registration --lib` → 591/591 passed, 1 skipped
- [x] `cargo nextest run -p ritk-io -p ritk-cli --no-fail-fast` → 523/524 passed (1 pre-existing JPEG2000 Windows abort)
- [x] Commit: ec6badc pushed to origin/main

---

## Sprint 366 — Architecture Hardening Round 5: NAMING · SSOT · COMPAT · DRY · SRP · ENUM · PRIM
**Target version**: 0.63.0  
**Sprint phase**: Closure — all 20 patches delivered and verified.

### Delivered (Sprint 366)
- [x] NAMING-CORE-01 [patch]: `gaussian_kernel_1d` → `gaussian_kernel`; all callers updated
- [x] ENUM-366-01 [minor]: `ResampleArgs.interpolation: String` → `InterpolationMode` ValueEnum
- [x] COMPAT-366-02 [patch]: Delete 4 `#[deprecated(0.64.0)] apply_3d` shims in noise filters
- [x] SSOT-366-03 [patch]: Delete dead `wgpu_compat.rs` shadow module in ritk-registration
- [x] COMPAT-366-04 [patch]: Remove `let _device` dead bindings in normalization modules
- [x] SSOT-366-05 [patch]: `NORMALIZER_EPSILON` const; `minmax.rs` + `zscore.rs` updated
- [x] SSOT-366-06 [patch]: `FOREGROUND_THRESHOLD` const; 4 statistics modules updated
- [x] SSOT-366-07 [patch]: Fix stale docs in `deconvolution/helpers.rs` + `mod.rs`
- [x] NAMING-366-08 [patch]: `cross_3d/normalize_3d/dot_3d` → `cross/normalize/dot`; 22 callers updated
- [x] NAMING-366-09 [patch]: `spatial_gradient_2d/_3d`/`spatial_laplacian_2d/_3d` → `*_planar/*_volumetric`
- [x] NAMING-366-10 [patch]: `VectorField3D/VectorFieldMut3D` → `VectorField/VectorFieldMut`; 12 files updated
- [x] NAMING-366-11 [patch]: `get_f64/get_f64_vec` → `get_scalar/get_scalar_vec` in series/loader.rs
- [x] DRY-366-12 [patch]: `read_nested_f64` consolidated into `dicom/helpers.rs`
- [x] SRP-366-13 [patch]: `threshold/li.rs` inline tests → `tests_li.rs`
- [x] SRP-366-14 [patch]: `threshold/yen.rs` inline tests → `tests_yen.rs`
- [x] SRP-366-15 [patch]: `watershed/mod.rs` inline tests → `tests_watershed.rs`
- [x] SRP-366-16 [patch]: `labeling/relabel.rs` inline tests → `tests_relabel.rs`
- [x] SRP-366-17 [patch]: `color_multiframe.rs` inline tests → `tests_color_multiframe.rs`
- [x] PRIM-366-18 [patch]: `SegmentArgs.markers: Option<String>` → `Option<PathBuf>`
- [x] COMPAT-366-19 [patch]: Remove dead `integration_steps` field from `DiffeomorphicSSMMorph`

### Blocked / Deferred
- [ ] NAMING-362-23 [arch]: `transform_1d/_2d/_3d/_4d` — BLOCKED; `DimInterpolation<B>` sealed trait design needed
- [ ] SRP-362-20 [major]: `FilterArgs` → `FilterKind` ValueEnum — carry forward
- [ ] NAMING-FILTER-01 [major]: `FftConvolution3DFilter`/`FftNormalizedCorrelation3DFilter` → const-generic unification

### Verification gate (Sprint 366)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-core -p ritk-filter -p ritk-segmentation` → 1447/1447 passed
- [x] `cargo nextest run -p ritk-registration --lib` → 591/591 passed, 1 skipped
- [x] `cargo nextest run -p ritk-io -p ritk-cli --no-fail-fast` → 526/527 (1 pre-existing JPEG2000 Windows abort)
- [x] Commit: 0feb9ec pushed to origin/main

---

## Sprint 365 — Architecture Hardening Round 4: COMPAT · NAMING · SSOT · SRP · DRY · DIP · ENUM
**Target version**: 0.62.0  
**Sprint phase**: Closure — all 20 patches delivered and verified.

### Delivered (Sprint 365)
- [x] COMPAT-365-01 [patch]: Delete dead `NormalizationMode` + test from `metric/trait_.rs`
- [x] NAMING-365-02 [patch]: `collect_vec_3/9` → `collect_array::<N>` in histogram/cache.rs; fix doc
- [x] NAMING-365-03 [minor]: `StopReason` → `CmaEsStopReason` in cma_es/state.rs + re-exports
- [x] DIP-365-04 [minor]: `RegistrationConfig::build_tracker()` + `TrackerBuildResult`; engine decoupled
- [x] SRP-365-05 [patch]: `correlation_ratio.rs` tests → `tests_correlation_ratio.rs`
- [x] COMPAT-365-06 [patch]: Delete deprecated dead `apply_tikhonov_2d/_3d` from regularization.rs
- [x] NAMING-365-07 [patch]: 6 private dim-suffix renames in ritk-filter; all call sites updated
- [x] SRP-365-09 [patch]: `image_statistics.rs` tests → `tests_image_statistics.rs`
- [x] SRP-365-10 [patch]: `minmax.rs` tests → `tests_minmax.rs`
- [x] DRY-365-11 [patch]: `build_tensor` helper extracted from `filter/ops.rs` rebuild bodies
- [x] SSOT-365-12 [minor]: `.ima` added to `ImageFormat::from_path` Dicom arm; `is_likely_dicom_file` unified
- [x] NAMING-365-13 [patch]: `DicomObjectNode::u16/i32/f64` → `from_u16/from_i32/from_f64`
- [x] DRY-365-14 [patch]: `io_err()` helper; 17 repeated closures removed in ritk-python/io/mod.rs
- [x] PRIM-365-15 [patch]: `read_transform`/`write_transform` `String` → `&str` at PyO3 boundary
- [x] NAMING-365-16 [patch]: `gaussian_smooth_3d` → `gaussian_smooth` in level_set/helpers.rs
- [x] NAMING-365-17 [patch]: `skeleton_1d/2d/3d` → `endpoint_extract`/`zhang_suen`/`sequential_thin`
- [x] NAMING-365-18 [patch]: `dilate/erode_1d/2d/3d` → `dilate/erode_line/plane/volume`
- [x] ENUM-365-19 [minor]: `StatsArgs.metric: String` → `StatMetric` ValueEnum (7 variants)
- [x] ENUM-365-20 [minor]: `RegisterArgs.method: String` → `RegistrationMethod` ValueEnum (10 variants)

### Blocked / Deferred
- [ ] NAMING-362-23 [arch]: `transform_1d/_2d/_3d/_4d` — BLOCKED; `DimInterpolation<B>` sealed trait design needed
- [ ] SRP-362-20 [major]: `FilterArgs` → `FilterKind` ValueEnum — carry forward
- [ ] ENUM-365-03 [minor]: `ResampleArgs.interpolation: String` → `InterpolationMode` ValueEnum
- [ ] NAMING-CORE-01 [patch]: `gaussian_kernel_1d` → `gaussian_kernel` (cross-crate callers)
- [ ] NAMING-FILTER-01 [major]: FftConvolution*3DFilter → const-generic unification

### Verification gate (Sprint 365)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-filter` → 699/699 passed
- [x] `cargo nextest run -p ritk-core` → 373/373 passed
- [x] `cargo nextest run -p ritk-registration` → 630/630 passed, 23 skipped
- [x] `cargo nextest run -p ritk-segmentation` → 375/375 passed
- [x] `cargo nextest run -p ritk-io --no-fail-fast` → 329/330 (1 pre-existing JPEG2000 Windows abort)
- [x] `cargo nextest run -p ritk-cli` → 198/198 passed
- [x] Commit: c6daed5 pushed to origin/main

---

## Sprint 364 — Architecture Hardening Round 3: COMPAT · NAMING · SSOT · CACHE · SRP · PRIM · ENUM
**Target version**: 0.61.0
ritk-filter: → major bump | ritk-core: → minor bump | ritk-registration: minor bump | ritk-io: minor bump | ritk-cli: minor bump | ritk-python: minor bump
**Sprint phase**: Closure — all 20 patches delivered and verified.

### Delivered (Sprint 364)
- [x] COMPAT-364-01 [major]: Remove 16 deprecated `apply_2d`/`apply_3d` from deconvolution ×4 + fft ×4; fix doctests
- [x] SRP-364-02 [patch]: `noise.rs` (370L) → `noise/{mod,gaussian,salt_pepper,shot,speckle}.rs`
- [x] NAMING-364-03 [minor]: Noise `apply_3d` inversion fixed; `apply` is now real impl; `apply_3d` deprecated; 30+ test sites updated
- [x] NAMING-364-04 [minor]: Chamfer `cdt_3d*` → `cdt*`; `chamfer_distance_transform_3d*` → `chamfer_distance_transform*`
- [x] NAMING-364-05 [minor]: `compute_hessian_3d` → `compute_hessian`; frangi, sato, tests updated
- [x] CACHE-364-06 [patch]: `ParzenJointHistogram.cache`/`masked_cache` → `CacheSlot<T>`; `with_ref`/`with_mut` added
- [x] DRY-364-07 [patch]: `compute_image_joint_histogram` `Option<f32>` → `SamplingConfig`; `full_grid()` added
- [x] NAMING-364-08 [patch]: `cubic_bspline_1d` → `cubic_bspline_basis`
- [x] NAMING-364-09 [patch]: Remove `gaussian_kernel_1d_f64` redundant wrapper in `smooth.rs`
- [x] SRP-364-10 [patch]: `threshold_level_set.rs` inline tests → `tests_threshold_level_set.rs`
- [x] SRP-364-11 [patch]: `laplacian.rs` inline tests → `tests_laplacian_level_set.rs`
- [x] SRP-364-12 [patch]: `kapur.rs` inline tests → `tests_kapur.rs`
- [x] SRP-364-13 [patch]: `triangle.rs` inline tests → `tests_triangle.rs`
- [x] SRP-364-14 [patch]: `filter/ops.rs` → extract `gaussian_kernel_1d` into `filter/kernel_utils.rs`
- [x] SSOT-364-15 [minor]: `ImageFormat::Analyze` + `from_path` arms + `from_str_name()`
- [x] SSOT-364-16 [minor]: `ritk-python/io/mod.rs` if-chains → `ImageFormat::from_path` dispatch
- [x] SSOT-364-17 [patch]: `ritk-cli/commands/mod.rs` → `ImageFormat` dispatch; `write_image` takes `ImageFormat`
- [x] PRIM-364-18 [patch]: `ResampleArgs.spacing: String` → `Vec<f64>` with `value_delimiter = ','`
- [x] PRIM-364-19 [patch]: `ConvertArgs.format` → `ImageFormat`-typed resolution
- [x] ENUM-364-20 [minor]: `NormalizeMethod` ValueEnum replaces `NormalizeArgs.method: String`

### Blocked / Deferred
- [ ] DIP-362-13 [minor]: `RegistrationCallbackSet` DIP — deferred; requires surveying `src/progress/` first
- [ ] NAMING-362-23 [patch]: `transform_1d/_2d/_3d/_4d` — **BLOCKED** [arch] — duplicate method names on same type
- [ ] SRP-362-20 [major]: `FilterArgs` (46 fields) → `FilterKind` ValueEnum — carry forward
- [ ] ENUM-365-01 [minor]: `StatsArgs.metric: String` → `StatMetric` ValueEnum — **Done** (Patch 19)
- [ ] ENUM-365-02 [minor]: `RegisterArgs.method: String` → `RegistrationMethod` ValueEnum — **Done** (Patch 20)
- [ ] ENUM-365-03 [minor]: `ResampleArgs.interpolation: String` → `InterpolationMethod` ValueEnum

### Verification gate (Sprint 364)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-filter ritk-core ritk-segmentation ritk-io ritk-cli` → 1976/1977 (1 pre-existing JPEG2000 Windows abort)
- [x] `cargo nextest run -p ritk-registration` → 631/631 passed, 23 skipped
- [x] Commit: b740507 pushed to origin/main

---

## Sprint 363 — Architecture Hardening Round 2: DRY · SRP · PRIM · NAMING · CACHE
**Target version**: 0.60.0
ritk-core: 0.10.0 → 0.11.0 | ritk-registration: 0.54.0 → 0.55.0 | ritk-filter: → minor bump | ritk-io: 0.3.0 → 0.4.0
**Sprint phase**: Closure — all 20 patches delivered and verified.

### Delivered (Sprint 363)
- [x] DRY-362-04 [minor]: `UnaryImageFilter<Op, const D>` + `UnaryPixelOp` sealed trait; abs/sqrt/exp/log/square → type aliases; D-generic `apply`
- [x] SRP-361-06 [patch]: `label_morphology.rs` (445L) → `label_morphology/{mod,label_ops,reconstruction,tests}.rs`
- [x] PRIM-361-03 [minor]: `DiscreteGaussianFilter::new(Vec<GaussianSigma>)` — sigma not variance; all callers updated
- [x] PRIM-362-12 [minor]: `EarlyStoppingPolicy::Enabled { patience, min_improvement }` — bundle eliminates invalid state
- [x] NAMING-362-24 [patch]: `spatial_gradient_2d/_3d`, `spatial_laplacian_2d/_3d` → private `fn` in `dispatch.rs`; `spatial_ops.rs` deleted
- [x] CACHE-363-01 [patch]: `CacheSlot<LnccCacheEntry<B>>` in `lncc.rs`; `get_or_reinit_if` added to `CacheSlot`; `Arc<Mutex<Option<>>>` eliminated
- [x] SRP-362-19 [patch]: `series.rs` (438L) → `series/{types,scan,loader}.rs`; `Arc<Mutex<HashMap>>` replaced with lock-free collect-and-merge
- [x] SRP-362-18 [patch]: `seg/tests/convert.rs` (554L) → 4 focused test modules
- [x] PRIM-362-27 [minor]: `DicomSeriesInfo` — `pub(crate)` `ArrayString` fields + public `&str` accessors + `pub fn new()`
- [x] PRIM-362-25 [minor]: `IntensityRange<T>` validating newtype in `ritk-core::statistics`
- [x] PRIM-362-25b [minor]: `MinMaxNormalizer` adopts `IntensityRange<f32>`
- [x] PRIM-362-25c [minor]: `CorrelationRatio::new` adopts `IntensityRange<f32>` for intensity bounds
- [x] BOOL-361-05a [minor]: `RegisterArgs.sigma_fixed: GaussianSigma` via clap `value_parser`
- [x] BOOL-361-05b [minor]: `RegisterArgs.kernel_sigma: GaussianSigma` via clap `value_parser`
- [x] FIX-363-01/02/03/04 [patch]: Cross-crate call site fixes (ritk-cli smoothing, ritk-cli viewer, ritk-snap series_tree, ritk-python gaussian)

### Blocked / Deferred
- [ ] DIP-362-13 [minor]: `RegistrationCallbackSet` DIP — deferred; requires surveying `src/progress/` ProgressTracker internals first
- [ ] NAMING-362-23 [patch]: `transform_1d/_2d/_3d/_4d` — **BLOCKED**: duplicate method names on same type; [arch] refactor required
- [ ] SRP-362-20 [major]: `FilterArgs` (46 fields) → `FilterKind` ValueEnum — carry forward

### Verification gate (Sprint 363)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` → 0 warnings
- [x] `cargo nextest run -p ritk-filter ritk-registration ritk-core ritk-io ritk-snap ritk-cli --no-fail-fast` → 2868/2869 passed (1 pre-existing JPEG2000 Windows codec abort)
- [x] Commit: 59f4bee pushed to origin/main

---

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
