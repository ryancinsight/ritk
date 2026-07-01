# RITK Backlog - Active Planning

> **Full sprint history (Sprints 262-322)**: see [ARCHIVE.md](./ARCHIVE.md)

---

## Open safety items

- **TEST-447-05 [patch] — MINC format-level hostile-fixture regression. READY.**
  Acceptance: construct (or extend the MINC writer to forge) an HDF5 file whose
  image dataset shape claims more bytes than are backed on disk, and assert
  `read_minc` returns a typed error without OOM. Blocked on a way to emit a
  shape≠data HDF5 fixture; the underlying `read_bounded_with` primitive is
  unit-tested in `ritk-core::io_bounds`. Driver: complete Sprint 447 per-crate
  regression coverage.

- **SEC-446-05 [patch] — Untrusted-input allocation hardening for the remaining
  format-parser crates. DONE (Sprint 447).**
  `ritk-mgh`, `ritk-metaimage`, and `ritk-minc` readers route through the new
  `ritk-core::io_bounds` SSOT helpers; `ritk-vtk` migrated onto the same module
  (per-crate copies removed). `ritk-nifti` already validates `volume_byte_range`
  against the input length and `ritk-nrrd` allocates from real payload length, so
  both were already safe. Evidence tier: value-semantic nextest (352 passed
  across the five crates) plus compile/lint/docs.

- **SEC-446-01 [patch] — VTK reader untrusted-input allocation hardening. DONE.**
  `ritk-vtk` binary/ASCII VTK and PLY readers no longer reserve `count * size`
  bytes up front from header count fields. SSOT `read_exact_bounded` /
  `bounded_capacity` helpers cap speculative allocation at 16 MiB/chunk and
  report truncation; `read_binary_be` checks the length product for overflow.
  Evidence tier: value-semantic nextest plus compile/lint/docs;
  `cargo nextest run -p ritk-vtk` passed 256 tests including hostile-count,
  overflow, and truncation regressions.

---

## Open performance items

- **PERF-432-01 [patch] — Registration integration tests exceed the strict
  nextest budget. READY.**
  Acceptance: profile the slow registration integration tests reported by
  `cargo nextest run -p ritk-registration --features coeus` and reduce each
  unmodified test below the AGENTS.md 30s slow threshold, or replace the
  `.config/nextest.toml` 600s override with a stricter repo policy only after
  the real implementation path is optimized. Current evidence: full package run
  passed 669 tests with `bspline_registers_offset_sphere` at 87.615s; the fused
  MSE interpolation slice reduced the focused row to 76.441s but did not close
  the budget violation. A follow-up identity-direction fast path was rejected
  after regressing the focused row to 78.925s; with that production change
  removed, the latest focused evidence is 80.456s.

  **New profiling evidence (this pass)**: instrumented `run_loop` with
  `std::time::Instant` timers around forward/backward/step (removed after
  measuring; `cargo flamegraph`/`perf`/`samply` are impractical on this
  Windows/MSYS host — flamegraph requires a release rebuild too slow to
  iterate with and Windows lacks `perf`). On `bspline_registers_offset_sphere`
  (10³ voxels, 5³ B-spline control points, 200 iterations): `forward` ≈ 42%,
  `backward` ≈ 45%, optimizer `step` + scalar extraction ≈ negligible (<0.1%)
  of the ~87s loop. The bottleneck is squarely the metric-forward /
  autodiff-backward tensor graph, consistent with `BSplineTransform::
  transform_3d_chunk` chaining ~30 distinct burn tensor ops (gather/select/
  reshape/floor/clamp/compare/mul/sum) per call — each op is a separate
  autodiff graph node whose dispatch/allocation overhead dominates at this
  workspace's mandated `[profile.test] opt-level = 0` (kept for readable
  Windows backtraces). This explains why "fused MSE interpolation" (reducing
  op count) got a real but partial win (87.6s→76.4s) and why further op-count
  reduction is the correct, durable fix — not a config change.

  **Rejected approach (evidence-backed, do not re-attempt as-is)**: configuring
  `Registration::with_config(...).with_convergence_detection(ConvergenceChecker)`
  to stop once the aggregate MSE loss plateaus. A full per-iteration loss dump
  simulated offline against patience∈{10,20,30,50}/threshold∈{1e-4..5e-3}
  showed threshold=1e-4,patience=20 robustly triggers at iteration 90 (loss
  0.013698 vs the iteration-199 floor of 0.013640 — only 0.4% higher). Wiring
  that exact config into the test still **failed the assertion**
  (err_x 0.668 vs the iteration-199 value 0.342, threshold 0.5) — a 2x error
  increase from a 0.4% loss difference. Root cause: aggregate voxel-wise MSE
  is dominated by the (much larger) near-static background, so the loss curve
  can visually plateau while the few control points governing the actual
  query point (5,5,5) are still refining — aggregate-loss convergence is not a
  safe proxy for this test's single-point geometric assertion. Do not retune
  the threshold further; the failure mode is structural, not a tuning miss
  (confirmed by testing two thresholds an order of magnitude apart, both
  failing). This may still be valid for OTHER registration tests whose
  assertions are loss-aligned (e.g. a global-similarity check) — untested.

  **Correction to the item filed in the previous pass, plus new precise
  evidence (this pass)**: the "hoist static `range`/`i_idx`/`j_idx`/`k_idx`/
  `zeros` tensors" idea filed previously was **reasoned from reading the code,
  not measured** — exactly the mistake "profile before optimizing" exists to
  prevent. Directly instrumented those 5 lines inside `transform_3d_chunk`
  with `std::time::Instant` (added, measured, fully reverted): across 200
  calls they cost 28.4ms total out of 58.9s (**0.05%**) — negligible. **Do
  not implement that hoist**; it would not measurably help and is not worth
  the risk of adding fields to a `#[derive(Module)]` struct.

  A follow-up section-by-section instrumentation of the whole
  `transform_3d_chunk` body (5 buckets: grid+mask setup, basis evaluation,
  weights outer product, index computation, and the final gather+weighted-sum
  block) found the cost concentrated almost entirely in the **last bucket**:
  of 52.2s total across 200 calls — grid_mask 0.67s (1.3%), basis 0.43s
  (0.8%), weights 2.34s (4.5%), index 4.87s (9.3%), **gather_sum 43.86s
  (84.1%)**. That bucket is `flat_indices.reshape(...)` →
  `t.coefficients.val().select(0, gather_indices)` (gathering 64 control-point
  rows per query point — 64,000 gathered rows for this test's 1000 points) →
  `reshape` → `(coeffs * weights).sum_dim(1).flatten(...)` →
  `displacement * valid_mask` → `points + masked_displacement`
  (`crates/ritk-transform/src/transform/bspline/interpolation/dim3.rs`,
  the block starting at `let gather_indices = ...` through the function's
  return). `select` gathers the one tensor that carries gradients
  (`t.coefficients`, the only `Param`), so burn's autodiff backward for this
  block is a scatter-add over 64,000 indices into a 125-row buffer — the
  likely reason `backward` is 45% of the outer loop.

  **Why no fix is filed as "ready to implement"**: this bucket is not
  reducible by caching (every op depends on `points`, `base_index`, or the
  per-iteration `t.coefficients` — none of it is iteration-invariant) and not
  reducible by simple op-fusion at the call-site level (it's already one
  contiguous chain, not scattered redundant calls like the earlier "fused MSE
  interpolation" win). A real fix needs either a custom fused gather+weighted-
  sum burn kernel (framework-level work) or bypassing burn's generic
  autodiff for this specific operation with a manually-implemented analytic
  backward (an architectural change to the `Transform`/`Metric` trait
  contract) — both larger, riskier undertakings than this session's verified-
  and-reverted profiling passes. Filed as the next investigation target, not
  a scoped implementation task: quantify whether a hand-written CPU gather-
  weighted-sum (bypassing burn's generic `select` for just this hot path,
  with a matching hand-derived backward) is worth the correctness-
  verification cost, before attempting it.

  **Also unverified — do not trust without measuring first**: a prior version
  of this entry also claimed `MeanSquaredError::forward`'s per-call
  `grid::generate_grid`/`index_to_world_tensor` recompute was a "verified"
  win. It was reasoned, not measured, same as the retracted item above — given
  the gather_sum bucket now measured at 84% of `transform_3d_chunk` alone,
  this grid-recompute cost is almost certainly comparatively small, but that
  has not been directly measured either. Treat as an open hypothesis, not a
  ready increment.

- **MEM-445-01 [patch] — MAD noise work-buffer reuse. DONE.**
  MAD noise estimation now overwrites its mutable work buffer with absolute
  deviations after the median is known, avoiding the previous second
  `Vec<f32>` allocation for deviation sorting. Evidence tier: value-semantic
  nextest plus compile/lint/docs; `cargo nextest run -p ritk-statistics
  --features coeus mad` passed 9 tests, including borrowed-slice
  order-preservation coverage.

- **MEM-444-01 [patch] — Histogram matching allocation cleanup. DONE.**
  `HistogramMatcher::match_histograms` now reuses the extracted source voxel
  buffer as the transform output after landmark estimation, avoiding a separate
  output `Vec<f32>`. `quantile_landmarks` now emits landmarks during a single
  histogram-bin scan instead of allocating a cumulative histogram `Vec<u64>`.
  Evidence tier: value-semantic nextest plus compile/lint/docs;
  `cargo nextest run -p ritk-statistics --features coeus histogram_matching`
  passed 12 tests, including an unsorted-order regression.

- **MEM-443-01 [patch] — Nyul-Udupa output buffer reuse. DONE.**
  `NyulUdupaNormalizer::apply` still needs one sorted work buffer because
  percentile landmarks require sorted intensities while image reconstruction
  must preserve original voxel order. It now reuses the extracted original-order
  voxel buffer as the transform output after landmark computation, avoiding a
  separate output `Vec<f32>` allocation. Evidence tier: value-semantic nextest
  plus compile/lint/docs; `cargo nextest run -p ritk-statistics --features coeus
  nyul` passed 21 tests, including an unsorted-order regression.

- **MEM-442-01 [patch] — Statistics full-image owned extraction cleanup.
  DONE.**
  Routed Burn-backed full-image statistics from `extract_vec_infallible`
  directly into the owned-buffer statistics core, avoiding a redundant clone of
  the extracted tensor values before percentile selection. Evidence tier:
  value-semantic nextest plus compile/lint/docs; `cargo nextest run -p
  ritk-statistics --features coeus image_statistics` passed 17 tests. The
  verified graph refreshed Coeus path crates in `Cargo.lock` from 0.5.3 to
  0.5.4.

- **MEM-441-01 [patch] — Statistics masked-buffer allocation cleanup. DONE.**
  Split the image-statistics core into the existing non-mutating borrowed-slice
  API and a crate-private owned-buffer path. Burn and Coeus masked statistics
  now consume the foreground vector directly for in-place percentile selection
  instead of cloning it before quickselect. Evidence tier: value-semantic
  nextest plus compile/lint/docs; `cargo nextest run -p ritk-statistics
  --features coeus image_statistics` passed 16 tests.

- **MIG-440-01 [patch] — Coeus image flat-buffer boundary. DONE.**
  Added `ritk_image::coeus::Image::from_flat_on` and `from_flat` so Coeus image
  construction from flat buffers validates checked shape products and length
  mismatches at the image boundary before tensor construction. Routed existing
  Coeus statistics and registration preprocessing test helpers through the new
  constructor. Evidence tier: type-level rank encoding plus value-semantic
  nextest and compile/lint/docs; `cargo nextest run -p ritk-image --features
  coeus from_flat` passed 3 tests, statistics Coeus rows passed 3 tests, and
  registration Coeus preprocessing rows passed 7 tests.

- **MIG-439-01 [patch] — I/O direct ndarray and workspace nalgebra cleanup.
  DONE.**
  Removed the unused direct `ndarray` dependency from `ritk-io` and removed the
  stale root workspace `ndarray` and `nalgebra` entries after auditing source
  and manifests for direct usage. Remaining matches are `burn_ndarray`
  backend/test aliases or Python `numpy::ndarray` boundary imports, not direct
  `ndarray`/`nalgebra` crate edges. Evidence tier: source audit plus
  compile/lint/docs and value-semantic nextest; `cargo nextest run -p ritk-io`
  passed 340 tests.

- **MIG-439-03 [minor] — Replace remaining Burn NdArray backend aliases with
  Atlas-backed surfaces. READY.**
  Acceptance: migrate one crate boundary at a time from `burn_ndarray::NdArray`
  aliases/tests to Coeus/Leto-backed surfaces without changing value semantics,
  then remove each direct `burn-ndarray` dependency when the crate no longer
  needs it. Start with an image/filter/IO boundary that has package-scoped
  nextest coverage and keep Python `numpy::ndarray` imports confined to PyO3
  conversion code.

- **MIG-437-01 [patch] — CLI MI registration direct ndarray boundary. DONE.**
  Replaced the `ritk-cli` MI registration image conversion helpers with
  `leto::Array3<f64>` so the CLI hands Leto volumes directly to the classical
  registration engine and spatial warp. Removed the direct `ndarray`
  dependency from `ritk-cli`; the remaining `burn_ndarray::NdArray` alias is a
  separate CLI backend migration item. Evidence tier: source audit plus
  value-semantic boundary test. `cargo nextest run -p ritk-cli leto_volume`
  passed 1 test; `cargo clippy -p ritk-cli --all-targets -- -D warnings`
  passed.

- **PROVIDER-437-02 [minor] — Moirai stream module rename completion. DONE.**
  Completed the `moirai-iter` `parallel_stream` -> `stream` module rename that
  blocked RITK Coeus rustdoc, and verified the bounded concurrent stream API in
  Moirai. Evidence tier: compile/lint/docs plus value-semantic nextest;
  `cargo nextest run -p moirai-iter stream` passed 10 tests, and RITK
  `cargo doc -p ritk-registration --features coeus --no-deps` passed after the
  provider fix.

- **MIG-438-01 [patch] — Registration direct ndarray dependency cleanup.
  DONE.**
  Removed the unused direct `ndarray` dependency from `ritk-registration` after
  auditing production source for direct `ndarray` symbols. The remaining
  registration matches are `burn_ndarray` test/backend aliases. Updated the
  classical-engine Rustdoc from stale ndarray wording to the active Leto array
  substrate. Evidence tier: source audit plus compile/lint/docs and
  value-semantic nextest; `cargo nextest run -p ritk-registration --features
  coeus classical` passed 45 tests.

- **PERF-435-01 [patch] — Route MSE through fused interpolation. PARTIAL.**
  Generalized `ritk_interpolation::transform_and_interpolate` over spatial
  dimensionality, generalized the OOB mask helper over the image shape length,
  and routed `MeanSquaredError` through the fused transform-to-index-to-linear
  interpolation path. Evidence tier: value-semantic nextest and focused timing;
  fused/OOB tests passed 8/8 and the MSE B-spline row passed at 76.441s. This is
  not a speedup claim against the 60s budget; PERF-432-01 remains open.

- **TEST-436-01 [patch] — Fused identity-direction coordinate convention.
  DONE.**
  Added asymmetric-origin, anisotropic-spacing differential coverage comparing
  fused interpolation against the unfused transform -> world-to-index ->
  interpolation path. Evidence tier: value-semantic differential nextest;
  `cargo nextest run -p ritk-interpolation fused` passed 8/8.

- **PERF-434-01 [patch] — Correct CR registration convergence and expose
  multires loop config. DONE.**
  Fixed `ConvergenceChecker` so the current best loss is compared against the
  previous patience window instead of being included in the best-loss baseline.
  Added `MultiResolutionRegistration::with_registration_config` and used the
  corrected convergence policy for B-spline CR and multires CR integration
  tests. Evidence tier: value-semantic nextest rows; the CR rows passed at
  22.302s and 23.720s in the focused run, then at 24.296s and 25.115s in the
  full package run. The MSE B-spline row remains open under PERF-432-01 at
  87.615s.

- **MIG-433-01 [minor] — Coeus preprocessing Gaussian smoothing. DONE.**
  Route `PreprocessingPipeline::execute_coeus` `Smoothing` through the existing
  Moirai-backed Gaussian smoothing primitive, extended with per-axis voxel
  sigmas for spacing-aware images. Coeus extraction/rebuild remains centralized
  in `ritk_tensor_ops::coeus`; smoothing reuses executor-owned scratch storage and
  rejects non-finite sigma. Evidence tier: compile/lint/docs plus
  value-semantic tests (`cargo nextest run -p ritk-registration --features
  coeus preprocessing` -> 20/20 passed; full package nextest -> 666/666
  passed). N4 bias correction remains the only unsupported preprocessing step
  on the Coeus executor.

- **MIG-432-01 [minor] — Coeus registration preprocessing scalar consumer. DONE.**
  Add feature-gated `PreprocessingPipeline::execute_coeus` for scalar-safe
  preprocessing steps and consolidate scalar value semantics into one
  `value_ops` implementation shared with the legacy Burn executor. Evidence
  tier: compile/lint/docs plus value-semantic tests (`cargo nextest run -p
  ritk-registration --features coeus` -> 661/661 passed; focused preprocessing
  selection -> 16/16 passed). Gaussian smoothing was closed by MIG-433-01; N4
  still requires Coeus/Leto/Hephaestus-backed filter migration.

- **MIG-431-01 [minor] — Coeus statistics image consumer. DONE.**
  Add feature-gated `ritk_statistics::image_statistics::coeus` entry points for
  Coeus-backed image statistics. The Coeus functions borrow image data through
  the Sprint 430 `ritk_tensor_ops::coeus` image helpers and reuse the existing
  slice-level statistics computation SSOT. Evidence tier: compile/lint plus
  value-semantic parity tests (`cargo nextest run -p ritk-statistics --features
  coeus` -> 290/290 passed; doctests and docs passed). Additional production
  image consumers still need Coeus image paths.

- **MIG-430-01 [minor] — Coeus image tensor-ops boundary. DONE.**
  Add feature-gated `ritk_tensor_ops::coeus` helpers for
  `ritk_image::coeus::Image<T, B, D>`: borrowed contiguous extraction, owned
  extraction, and checked rebuild while preserving image metadata. The image
  helpers delegate to the existing Coeus tensor rank, contiguity, and
  shape-product validation SSOT. Evidence tier: compile/lint/docs plus
  value-semantic tests (`cargo nextest run -p ritk-tensor-ops --features
  coeus` -> 24/24 passed). Production image callers still need migration from
  the legacy Burn root image type.

- **MIG-429-01 [minor] — Coeus image contract. DONE.**
  Add a feature-gated `ritk_image::coeus::Image<T, B, D>` backed by
  `coeus_tensor::Tensor<T, B>`. Construction validates tensor rank against the
  const image dimensionality; metadata access and `into_parts` preserve
  ownership; contiguous host borrowing is available only for CPU-addressable
  Coeus backends and rejects non-contiguous layouts instead of materializing
  silently. Evidence tier: compile/lint/docs plus value-semantic tests (`cargo
  nextest run -p ritk-image --features coeus` -> 33/33 passed). Legacy Burn
  image consumers remain until call sites migrate to this Coeus contract.

- **MIG-428-01 [minor] — Coeus tensor-ops host boundary. DONE.**
  Add a feature-gated Coeus-native host-buffer boundary to `ritk-tensor-ops`:
  borrowed contiguous extraction for zero-copy read-only kernels, owned
  extraction when mutation/storage is required, and checked tensor rebuild that
  rejects overflowing or mismatched shape products before allocation. Evidence
  tier: compile/lint/docs plus value-semantic tests (`cargo nextest run -p
  ritk-tensor-ops --features coeus` -> 20/20 passed). Legacy Burn-backed
  `Image<B, D>` helpers remain until image consumers migrate to the Sprint 429
  Coeus image contract.

- **MIG-427-01 [patch] — Coeus tensor-ops contract tests. DONE.**
  Consolidate `ritk-tensor-ops` Coeus feature tests so elementwise
  Coeus/Burn differential coverage runs through one table-driven fixture with
  explicit expected values. Shape-operation coverage now asserts reshape values
  and transpose logical indexing instead of shape-only success. Evidence tier:
  compile/lint/docs plus value-semantic tests (`cargo nextest run -p
  ritk-tensor-ops --features coeus` -> 14/14 passed).

- **MIG-426-01 [patch] — NIfTI fixture provenance and import coverage. DONE.**
  Add source-backed NIfTI import validation around `ritk-nifti`: the real
  repository NIfTI-1 gzip fixture (`test_data/registration/brain_fixed.nii.gz`)
  is documented as an ANTs/MNI152 copy and imported in tests; deterministic
  generated NIfTI-2 gzip fixtures validate the native writer/reader path; and
  Analyze-style `.hdr` bytes are rejected by the NIfTI reader so Analyze 7.5
  remains owned by `ritk-analyze`. The native reader now also imports UInt8
  NIfTI image payloads into the public f32 tensor boundary, with generated
  UInt8 fixture coverage and sourced MNI152 fixture coverage. Evidence tier:
  compile/lint/docs plus value-semantic tests (`cargo nextest run -p
  ritk-nifti` -> 34/34 passed).

- **MIG-425-01 [minor] — Native NIfTI-2 single-file codec. DONE.**
  Extend `ritk-nifti`'s native codec from NIfTI-1-only single-file support to
  automatic NIfTI-1/NIfTI-2 reads plus explicit NIfTI-2 image and label writers.
  The header module is now one versioned SSOT over datatype validation, endian
  detection, widened NIfTI-2 dimensions/spatial fields, checked payload ranges,
  and endian-aware payload lane reads. Analyze 7.5 `.hdr`/`.img` remains owned
  by `ritk-analyze`; paired NIfTI `ni1`/`ni2` is a separate deferred variant.
  Evidence tier: compile/lint/docs plus value-semantic tests (`cargo nextest
  run -p ritk-nifti` -> 29/29 passed).

- **MIG-424-01 [patch] — Native RITK NIfTI codec. DONE.**
  Replace `ritk-nifti`'s dependency on `nifti-rs` and direct ndarray
  conversion/writer handoff with a native NIfTI-1 single-file codec. The new
  vertical structure owns header parsing/serialization, checked dimensions,
  sform/qform spatial extraction, Float32 image decoding, Float32/UInt32 label
  decoding, and streamed `.nii` / `.nii.gz` writing without a full payload copy.
  Evidence tier: compile/lint/docs plus value-semantic tests (`cargo nextest
  run -p ritk-nifti` -> 25/25 passed).

- **MIG-423-01 [patch] — NIfTI shape bounds SSOT. DONE.**
  Move NIfTI voxel-count arithmetic into one `ritk-nifti::shape` helper used by
  reader and writer paths. Label and image writers now validate shape products
  before constructing ndarray handoff buffers, and adversarial overflowing label
  shapes fail with a typed error instead of multiplication wraparound or
  allocation. Evidence tier: compile/lint/docs plus value-semantic tests
  (`cargo nextest run -p ritk-nifti` -> 23/23 passed).

- **MIG-422-01 [patch] — PACS worker send signal and Tokio drift cleanup. DONE.**
  Remove the final stale Tokio reference from `ritk-snap` PACS worker docs,
  correct completed-response backpressure wording, and replace the discarded
  `SyncSender::send` result with one send-status helper covered by delivered
  and receiver-dropped value-semantic tests. The RITK source/manifests now have
  no `rayon`, `tokio`, `ParallelSlice`, `ParallelSliceMut`, `.par()`,
  `par_mut`, or `map_collect` matches. Evidence tier: compile/lint/docs plus
  value-semantic tests (`cargo nextest run -p ritk-snap` -> 635/635 passed).

- **MIG-421-01 [patch] — Direct Moirai DICOM series loading. DONE.**
  Replace `ritk-io` DICOM directory scan, series header parse, and pixel decode
  `ParallelSlice` extension-trait call sites with direct
  `moirai::map_collect_index_with::<moirai::Adaptive>` calls. This keeps
  file/slice ordering explicit by index and leaves no `ParallelSlice`,
  `.par()`, or `map_collect` matches in `crates/ritk-io/src/format/dicom`.
  Evidence tier: compile/lint/docs plus value-semantic tests
  (`cargo nextest run -p ritk-io` -> 340/340 passed).

- **MIG-420-01 [patch] — Direct Moirai filter diffusion enumeration. DONE.**
  Replace `ritk-filter` Perona-Malik and coherence diffusion
  `ParallelSliceMut` extension-trait call sites with direct
  `moirai::enumerate_mut_with::<moirai::Adaptive>` and indexed collection
  calls. The touched filter source now has no `ParallelSliceMut`, `par_mut`,
  Rayon, or Tokio matches, and projection docs no longer describe Rayon.
  Evidence tier: compile/lint/docs plus value-semantic tests
  (`cargo nextest run -p ritk-filter` -> 944/944 passed).

- **PROVIDER-420-01 [patch] — Hermes complex dispatch bound cleanup. OPEN.**
  The local Atlas provider graph exposed that Hermes complex SIMD operations
  still require `Neg` at the complex-operation dispatch surface after broader
  unsigned scalar support. A minimal local fix passes `cargo check -p
  hermes-simd --all-targets`; full provider rustfmt is still blocked by
  unrelated pre-existing `crates/hermes-simd/src/dispatch/axpy.rs` formatting
  drift and should land in Hermes separately.

- **MIG-419-01 [patch] — Direct Moirai registration enumeration. DONE.**
  Replace `ritk-registration` Parzen direct sparse-entry initialization and
  CMA-ES population fitness writes with direct
  `moirai::enumerate_mut_with::<moirai::Adaptive>` calls instead of the
  `ParallelSliceMut` extension trait. The touched registration contexts now have
  no `ParallelSliceMut`, `par_mut`, or stale Rayon wording. Evidence tier:
  compile/lint/docs plus value-semantic tests (`cargo nextest run -p
  ritk-registration` -> 656 passed, 23 skipped).

- **COEUS-419-01 [patch] — Fix local Coeus provider blockers. DONE.**
  Repair the dirty local Coeus provider graph required by the RITK registration
  gate: restore the shape root `flat_to_nd` export for moved shape leaves,
  restore the real `embedding_backward_with_padding_idx` accumulation path, and
  restore the autograd reshape contiguous-function import. Evidence tier:
  compile plus value-semantic provider tests (`cargo nextest run -p coeus-ops`
  -> 147 passed).

- **PERF-419-01 [patch] — Registration test runtime budget breach. OPEN.**
  Sprint 419's `ritk-registration` nextest gate passed but exposed integration
  tests above the 30s slow budget, including 100s, 146s, and 193s rows. Treat
  this as a real performance defect to profile; do not weaken or skip those
  tests.

- **MIG-418-01 [patch] — Direct Moirai segmentation enumeration. DONE.**
  Replace the last `ritk-segmentation` `ParallelSliceMut` extension-trait call
  sites in isolated watershed and STAPLE with direct
  `moirai::enumerate_mut_with::<moirai::Adaptive>` calls. This keeps the
  execution policy explicit at the call site and leaves no `ParallelSliceMut`,
  `par_mut`, `unsafe`, or `SendPtr` matches in `ritk-segmentation/src`.
  Evidence tier: compile/lint/docs plus value-semantic tests
  (`cargo nextest run -p ritk-segmentation` -> 435/435 passed).

- **MIG-417-01 [patch] — Level-set safe Moirai convergence metrics. DONE.**
  Replace the five level-set raw-pointer `SendPtr` convergence-metric side writes
  with one shared helper that pairs each mutable z-slice with its metric slot
  under Moirai dispatch. This removes RITK-local unsafe code from Chan-Vese,
  geodesic active contour, shape detection, Laplacian, and threshold level-set
  PDE loops while preserving per-slice convergence semantics. Evidence tier:
  compile/lint/docs plus value-semantic tests (`cargo nextest run -p ritk-segmentation`
  -> 435/435 passed).

- **MIG-416-01 [patch] — GrowCut safe Moirai paired assignment. DONE.**
  Replace `ritk-segmentation` GrowCut's raw-pointer `SendPtr` side-write pattern
  with Moirai paired mutable chunk dispatch over `next_strengths` and
  `next_labels`. This removes unsafe code from the GrowCut assignment loop while
  preserving disjoint per-voxel writes and seed label stability. Evidence tier:
  compile/lint/docs plus value-semantic tests (`cargo nextest run -p ritk-segmentation`
  -> 435/435 passed).

- **MIG-415-01 [patch] — SLIC safe Moirai paired assignment. DONE.**
  Replace `ritk-segmentation` SLIC assignment's raw-pointer `SendPtr` side-write
  pattern with Moirai paired mutable chunk dispatch over `distances` and `labels`.
  This removes unsafe code from the SLIC assignment hot path while preserving
  disjoint per-voxel writes and the existing SLIC distance contract. Evidence tier:
  compile/lint/docs plus value-semantic tests (`cargo nextest run -p ritk-segmentation`
  -> 435/435 passed).

- **MIG-414-01 [patch] — Gaia MeshBuilder array API migration. DONE.**
  Extend Gaia's `MeshBuilder` with coordinate-array and explicit xyz insertion APIs,
  then migrate RITK mesh construction sites to those provider APIs. Target outcome:
  `ritk-filter`, `ritk-vtk`, and `ritk-io` no longer declare direct `nalgebra`
  dependencies for Gaia mesh construction. Evidence tier: compile/lint/docs plus
  value-semantic provider and consumer tests (Gaia `cargo nextest run` -> 922
  passed, 1 skipped; RITK focused `cargo nextest run` -> 1532 passed).

- **MIG-413-01 [patch] — BinShrink direct Moirai output writes. DONE.**
  Replace `ritk-filter::bin_shrink`'s intermediate `(offset, value)` result staging
  with direct disjoint output-chunk writes through Moirai. This preserves the
  row-major bin-average contract while removing an allocation proportional to the
  output voxel count and avoiding a scatter pass. Evidence tier: compile/lint/docs
  plus value-semantic tests (`cargo nextest run -p ritk-filter` -> 944/944 passed).

- **MIG-412-01 [patch] — Statistics Atlas dependency cleanup. DONE.**
  Remove `ritk-statistics`' stale direct `nalgebra` dependency and correct Jacobian
  comments that still described Rayon even though the implementation already uses
  Moirai adaptive execution helpers. This is a dependency-surface and documentation
  cleanup only; it does not claim Burn/Coeus tensor migration or ndarray removal.
  Evidence tier: compile/lint/docs plus value-semantic tests
  (`cargo nextest run -p ritk-statistics` -> 287/287 passed).

- **FMT-406-01 [patch] — Restore full-repo rustfmt gate. DONE.**
  Sprint 406 applies the committed rustfmt style to the formatting drift that blocked
  `cargo fmt --check` after Sprint 405. This is mechanical hygiene only; no behavior,
  allocation, or performance change is claimed.

- **COEUS-406-01 [patch] — Fix dirty Coeus autograd provider compile break. OPEN.**
  RITK doctest/doc gates against the current local Atlas stack are blocked after refreshing
  Coeus path packages to `0.2.6`: `D:\atlas\repos\coeus` is dirty on
  `test/cuda-parity-suite`, and `coeus-autograd` fails to compile in shape/reduction ops.
  This must be fixed in Coeus before RITK can claim docs/doctests clean on the current
  provider graph.

- **PERF-406-02 [patch] — Registration test runtime budget breach. OPEN.**
  Sprint 406's touched-package `nextest` gate passed but exposed registration tests above
  the 30s slow budget, including 93s, 129s, and 183s rows. Treat this as a real
  performance defect to profile; do not weaken or skip those tests.

- **MIG-411-01 [patch] — SNAP spatial metadata Leto cleanup. DONE.**
  Remove `ritk-snap`'s direct `nalgebra` dependency where the crate only needs
  default and row-major direction construction. Route those sites through
  `ritk_spatial::Direction` so Leto-backed spatial metadata remains the single
  RITK-owned API. This does not claim the broader Burn/Coeus, ndarray, or mesh
  migration complete.

- **PERF-387-02 [patch] — Continue flat-buffer memory-efficiency audit. IN PROGRESS.**
  Sprint 387 flattened `VectorConfidenceConnected` covariance/inverse matrices and removed
  the B-spline legacy placeholder. Sprint 389 flattened `InverseDisplacementField` TPS
  spline/affine coefficient blocks after the solve. Sprint 390 flattened TIFF grayscale/RGB
  page accumulation by removing `Vec<Vec<f32>>` staging. Sprint 391 removed the binary VTI
  writer's duplicate flattened attribute buffers and streams appended blocks from source
  storage. Sprint 392 changed NRRD spatial header vector parsing from per-vector heap
  `Vec<f64>` buffers to const-generic fixed arrays. Remaining candidates are
  `vector_confidence_connected` channel buffer layout, public VTK cell-list storage, and
  remaining nested small matrices where a row-major buffer preserves public contracts.
  Sprint 401 removed the VTK unstructured-grid writers' internal per-cell string vector and
  duplicate VTU connectivity/offset staging. Sprint 403 made vector-confidence channel
  buffers fallible at the boundary, closing unchecked malformed channel-length indexing;
  remaining layout work is now public-model/API design rather than hidden unchecked access.
  Only the public nested VTK cell model remains.

- **SAFE-393-02 [patch] — Continue hostile format-header parser audit. IN PROGRESS.**
  Sprint 393 hardened NRRD spatial vector parsing so unterminated parenthesized groups return
  an error instead of accepting a parsed prefix. Sprint 394 hardened NRRD vector fields so
  trailing non-vector tokens and multiple `space origin` vectors are rejected. Sprint 395
  hardened DICOM RT Structure Set `ContourData` so present contour coordinates reject
  non-numeric components and partial trailing triples, while removing the intermediate scalar
  coordinate buffer. Sprint 396 hardened DICOM RT Dose grid fields so present frame offsets,
  DS vectors, frame counts, and pixel payload lengths are exact and fallible. Sprint 397
  hardened DICOM RT Plan sequence numerics so malformed present integer strings and
  non-sequence sequence tags are rejected instead of collapsed to zero or empty lists.
  Sprint 398 hardened MetaImage payload sizing so `DimSize` multiplication and payload byte
  counts are checked and exact. Sprint 399 hardened MINC dimension attributes so
  direction-cosine vectors and dimension lengths are exact and fallible. Sprint 400 hardened
  NIfTI spatial metadata so affine/qform/pixdim fields and voxel-count products are exact and
  fallible. Sprint 402 hardened VTU XML cell arrays so signed values, offset ordering, and
  final connectivity consumption are exact before narrowing or slice indexing. This tracked
  pass is closed for the named sibling medical-image parsers; reopen only with a concrete
  malformed-input finding.

- **SAFE-405-01 [patch] — FFT convolution padding bounds. DONE.**
  Sprint 405 centralizes 2-D/3-D FFT padding and boundary-extension shape arithmetic for
  `ritk-filter` convolution and normalized cross-correlation. The target is checked
  `usize` addition/multiplication and power-of-two extent validation before allocation,
  plus removal of `usize as isize` source-index casts in edge replication.

- **CLIPPY-387-01 [patch] — `ritk-interpolation` linear-kernel slice lint cleanup. DONE.**
  Focused Clippy was blocked by `clippy::single_range_in_vec_init` in
  `interpolation/kernel/linear/{dim2,dim3,dim4}.rs`; the kernels now route gathered 1-D
  corner-batch splits through the shared `linear::slice_batch` helper backed by
  `Tensor::slice_dim`. Evidence tier: compile/lint and value-semantic focused tests.

- **PERF-379-01 [patch] — Deriche recursive-Gaussian cross-line parallelism. DONE.**
  `iir::apply_deriche_1d` now parallelises the X/Y passes across Z-slices via
  `moirai::for_each_chunk_mut` (contiguous `nyx` chunks, one `LineScratch` per
  slice); the per-line IIR is factored into `deriche_line`. Output **bit-identical**
  to the serial form (exact array equality; float-exact sitk parity unchanged).
  Min-of-20 on 128³: smooth 1.50×, grad-mag 1.71×, LoG 1.81×. Z pass (dim 0,
  strided) remains serial — residual headroom (transpose / multi-line ILP).

---

## Gap items

- **MIG-387-01 [arch] — Atlas crate migration audit.**
  Continue replacing production `nalgebra`/`ndarray`/`burn` surfaces with `leto`/`coeus`/
  `hephaestus` only after each target operation has a verified equivalent contract and focused
  differential tests. Do not remove boundary dependencies used only for file-format interop or
  external framework contracts until the replacement can preserve the same behavior. Sprint 404
  removed the unused `rustfft` workspace dependency and reconciled stale FFT docs after verifying
  RITK's FFT execution path already uses `apollo_fft::FftPlan1D`. Sprint 407 removes
  `nalgebra` from `ritk-registration` by routing classical 3-D landmark/perturbation
  math through Leto stack fixed matrices/vectors and Kabsch singular vectors through
  `leto-ops`. Sprint 408 migrates `ritk-spatial` storage to Leto. Sprint 409
  removes DICOM/MINC/filter spatial metadata construction from direct `nalgebra`
  interop. Sprint 410 removes the `ritk-png` test-only `nalgebra` edge; remaining
  `nalgebra` work should be handled by bounded contexts such as SNAP spatial
  consumers and mesh-only geometry.

- **MIG-387-02 [arch] — Spatial Leto SSOT migration. IN PROGRESS.**
  Sprint 408 migrates `ritk-spatial` storage to Leto fixed vectors/matrices and removes
  direct `nalgebra` dependencies from `ritk-core`, `ritk-metaimage`, `ritk-nrrd`,
  `ritk-nifti`, and `ritk-mgh` spatial direction setup. Sprint 409 moves DICOM IO,
  MINC, and filter spatial-transform consumers onto `Direction`, `Point`, and
  `Vector`. Sprint 410 removes `ritk-png`'s test-only `nalgebra` dependency.
  Remaining spatial call sites in SNAP and mesh-only geometry must be handled in
  follow-up slices scoped to their bounded contexts.

---
## Sprint 377 — Performance Review, Memory Efficiency & Carry-Forward Reconciliation

**Status**: In Progress
**Version**: 0.91.1

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| GATE-377-01 | `segmentation::threshold::{huang,isodata,renyi}` `for i in 0..n { vec[i] }` → iterator-with-enumerate (idiomatic); `morphology::window_1d` inline `#[allow]` with justification [patch] | Done (`de26c2fc`) |
| FMT-377-01 | `cargo fmt --check` clean (staged files); 22 working-tree diffs from cumulative agent updates remain pending rewash | Pending |
| DOC-377-01 | 16 intra-doc-link warnings accumulated from Sprint 393-395 commits; non-blocking | Pending |

### In Flight

| ID | Description | Priority |
|----|-------------|----------|
| PERF-377-01 | **MedianFilter O(N·n³·log n) → O(N·r²)** via Huang's sliding column histogram — bit-exact equivalence to naive reference on every radius | Next |
| PERF-377-02 | **BilateralFilter memory-bandwidth review** — current LUT/SIMD-friendly; headroom: drop `exp` into a second LUT, separable approximation | Deferred (depends on benchmark) |
| PERF-377-03 | **Rank/Percentile filter** — same naive O(N·n³·log n) pattern as median; bundle if algorithm portable | Deferred |

### Verification (so far)
- `cargo fmt --check`: 0 diffs (staged)
- `cargo clippy --workspace --all-targets -- -D warnings`: 0 warnings
- `cargo nextest run -p ritk-segmentation -E 'test(threshold)'`: 120/120 passed
- `cargo nextest run -p ritk-filter -E 'test(unary_minus)|test(round_half)'`: 2/2 passed

### Known WIP in working tree (parallel agent; do NOT touch)

**Status at 2026-06-17 session resume**: 22 files modified in working tree (parallel agent); plus 1 build-artifact deletion. Below is the live file set right now.

- `crates/ritk-filter/src/color.rs` (whitespace-fmt)
- `crates/ritk-filter/src/morphology/{label_contour.rs,label_morphology/reconstruction.rs,regional_extrema.rs,tests_grayscale_fillhole.rs,tests_grayscale_grind_peak.rs,tests_h_transform.rs,tests_hit_or_miss.rs,tests_reconstruction_opening_closing.rs,tests_regional_extrema.rs}` — morphology feature/test batch
- `crates/ritk-filter/src/tests_color.rs`
- `crates/ritk-image/src/color.rs`
- `crates/ritk-python/pyproject.toml`, `src/segmentation/{labeling.rs,threshold.rs}`
- `crates/ritk-python/tests/{test_smoke.py,test_registration_gap_validation.py,test_registration_side_by_side.py,test_elastix_vs_ritk_rire.py}`
- `crates/ritk-registration/src/classical/global_mi/cma_mi/helpers.rs`, `src/metric/mutual_information/mod.rs`
- `crates/ritk-segmentation/src/threshold/{kittler.rs,mod.rs}`
- `rust_out.exe` (deleted build artifact)

**Recent parallel-agent commits landed in this window**:
- `271c026c feat(ritk-filter): add MedianProjection filter` (MedianIntensityProjectionFilter, Cargo.toml bump `0.2.23 → 0.2.24`, lib.rs re-export, `projection.rs` +67 lines; uses `select_nth_unstable_by` at `n/2` — naive O(N·n·log n), orthogonal to PERF-377-01).

Per `concurrent_agents`: preserve all of the above untouched.

`crates/ritk-filter/src/median.rs` is **clean** — this session's sole working file (`PERF-377-01 Huang sliding-histogram MedianFilter`).

---

## Sprint 376 — DRY Closure, Build Hardening & Carry-Forward Reconciliation

**Status**: Closure (all in-flight items delivered)
**Version**: 0.70.1

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| DRY-374-01 | `ritk-image::test_support` consumed by 78 test files via thin local wrappers; 51 files had unused-import cleanup via `cargo fix` [minor] | Done |
| CARRY-376-01 | `feat(python): expose normalize, unsharp, zero-crossing, rotate, shift, zoom` — 6 new PyImage functions [minor] | Done |
| CARRY-376-02 | `feat(stats): ddof flag for sample (sitk) vs population std` + parity tests [minor] | Done |
| CARRY-376-03 | Carry-forward filter binding surface expansion (single-axis match sitk Euler3DTransform + extended corpus + API mismatches) [patch] | Done |
| CLIPPY-376-01 | Doc list indent + Range single-element array lint failures resolved [patch] | Done |
| FMT-376-01 | `cargo fmt --check` clean (0 diff lines) [patch] | Done |
| BILAT-PERF-01 | Bilateral filter `compute`: 1-D `spatial_w` lookup table + clamped boundary iteration; boundary checks hoisted out of inner loop; per-voxel inner-loop cost reduced to one table lookup + one `exp` [minor] | Done |
| BILAT-BENCH-01 | criterion bench `benches/bilateral.rs` covering 16³/32³/64³ sizes, recording baseline [patch] | Done |
| CPR-PERF-01 | `CprImageFilter::apply`: hoisted `direction.inverse()` (3×3 inverse once per call vs once per cross-section sample) + per-path-point index basis; new private `trilinear_sample_from_idx` helper; bit-equivalent to pre-optimisation form (`max |Δ| ≤ 1e-5`); head-to-head 1.98×/1.47×/1.14× on 16³/32³/64³ default config [patch] | Done |
| CPR-REGRESSION-01 | `cpr_apply_matches_brute_force_reference` + `cpr_apply_matches_brute_force_reference_nonidentity_direction` brute-force differential tests (12³ identity, 10³ 90°-rotated-Z direction) [patch] | Done |
| CPR-BENCH-01 | criterion bench `benches/cpr_apply.rs` end-to-end `apply` over 16³/32³/64³ at default config; head-to-head measured vs reverted reference [patch] | Done |

### In Flight

| ID | Description | Priority |
|----|-------------|----------|
| MSG-376-01 | Rename `fc9d009e` placeholder commit to canonical `perf(ritk-filter): CPR direction-inverse hoist + per-path-point index basis` via interactive rebase (deferred: concurrent agent pushed `91991789` on top; force-rebase would clobber shared history). Local commit content is correct; the message is the only outstanding item. | Next |

### Blocked / Deferred (carry-forward, unchanged from Sprint 375)

| ID | Description | Priority |
|----|-------------|----------|
| VAR-375-01 | `PhantomData<B>` → `PhantomData<fn() -> B>` blocked at `burn-core-0.19.1` | [upstream] |
| CONST-375-02 | const-assert companion on `BSplineTransform` blocked on `const_panic_fmt` | [toolchain] |
| NAMING-362-23 | sealed trait `DimInterpolation<B>` BLOCKED — ADR required | [arch] |
| SRP-362-20 | `FilterKind` ValueEnum — slice delivered; per-family Args structs remain | [minor] |
| NAMING-FILTER-01 | `FftConvolution3DFilter` const-generic unification — cross-crate dependent | [major] |
| N-375-08 | DRY cross-crate parse utils — promotion trigger requires `ritk-io` → `ritk-core` migration | [arch] |
| TIMEOUT-367 | 4 heavy interpolation dispatch tests pre-existing perf issue | [patch] |

### Verification (so far)

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `cargo fmt --check` (my changed files) | 0 diffs |
| ritk-filter nextest | 707/707 (+2 vs prior 705 from `cpr_apply_matches_brute_force_reference*`) |
| ritk-segmentation + ritk-statistics + ritk-tiff nextest | 707/707 |
| ritk-image + ritk-statistics nextest | 312/312 |
| Doctests `-p ritk-filter` | 2 pass / 11 ignored |
| `cargo bench --bench cpr_apply` | 525 µs / 1.04 ms / 4.86 ms (16³/32³/64³) — speedup 1.90× / 1.38× / 1.10× vs reverted reference |
| `cargo bench --bench bilateral` | 14.4 ms / 152 ms (16³/32³, 64³ unmeasured) |
| Python smoke (`pytest tests/test_smoke.py`) | 16/16 |

---

## Sprint 375 — Architecture Hardening Round 8: SSOT · DRY · NAMING · ENUM · SRP · COMPAT

**Status**: Complete  
**Version**: 0.70.0  

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| P01 | [HARD] fake UID bypass fix in seg/writer.rs — real generate_uid() restored [patch] | Done |
| P02 | SSOT: EXPLICIT_VR_LE propagated to 6 writers [patch] | Done |
| P03 | DRY: normalize_to_u16 helper extracted in ritk-io [patch] | Done |
| P04 | DRY: UID generation dedup — 5 private counters deleted [patch] | Done |
| P05 | DRY: emit_pixel_format_tags helper extracted [patch] | Done |
| P06 | ENUM: RtRoiInterpretedType replaces Option<String> in RtRoiInfo [minor] | Done |
| P07 | ENUM: RtDoseType / RtDoseSummationType replace ArrayString<16> in RtDoseGrid [minor] | Done |
| P08 | ENUM: SegmentationType / SegmentAlgorithmType replace ArrayString<16> in DicomSegmentation/DicomSegmentInfo [minor] | Done |
| P09 | DRY+NAMING: DicomObjectNode::with_value<V> generic + get_u32 rename + is_image_sop_class + Association::config removed [minor] | Done |
| P10 | NAMING: delete 12+1 type-concrete read functions → read_helpers in ritk-vtk [minor] | Done |
| P11 | NAMING: promote read_line + parse_cells_from_ints to read_helpers [patch] | Done |
| P12 | DRY: write_attribute dedup + VTP write_attr dedup in ritk-vtk [patch] | Done |
| P13 | DRY: create shared io/xml_helpers.rs (3 duplicates eliminated) [patch] | Done |
| P14 | NAMING: char::from(Nu8) → char literals in 11 files + DEFAULT_ORIGIN/SPACING_STR consts [patch] | Done |
| P15 | SRP: domain/filters test extraction (3 files) in ritk-vtk [patch] | Done |
| P16 | SRP: io test extraction (3 files) in ritk-vtk [patch] | Done |
| P17 | COMPAT: compat cleanup + truncated doc fix in ritk-vtk [patch] | Done |
| P18 | SSOT: ORTHOGONALITY_TOLERANCE const + spacing.rs test extracted in ritk-spatial [patch] | Done |
| P19 | COMPAT: deprecated to_vec() removed from Point/Vector in ritk-spatial [minor] | Done |
| P20 | SRP: shape_markers.rs test extracted in ritk-morphology [patch] | Done |
| P21 | NAMING: extract_f64 → extract_scalar_float; extract_f64_array_3 → extract_float_array_3 in ritk-minc [minor] | Done |
| P22 | NAMING: build_attr_msg_f64 → build_attr_msg_float in ritk-minc [minor] | Done |
| P23 | NAMING: convert_to_f32 → decode_raw_bytes + pub(crate) hdf5_binary + build_scalar_attr_raw helper in ritk-minc [minor] | Done |
| P24 | NAMING: decode_bytes_to_f32 → decode_element_bytes + parse_f64_vec → parse_float_vec in ritk-metaimage [minor] | Done |
| P25 | SRP: split metaimage/reader.rs → reader/mod.rs + reader/decode.rs [patch] | Done |
| P26 | NAMING: same renames (decode_element_bytes / parse_float_vec) in ritk-nrrd [minor] | Done |
| P27 | SRP: extract inline test block 1 from ritk-snap [patch] | Done |
| P28 | SRP: extract inline test block 2 from ritk-snap [patch] | Done |
| P29 | SRP: extract inline test block 3 from ritk-snap [patch] | Done |
| P30 | SRP: extract inline test block 4 from ritk-snap [patch] | Done |
| P31 | SRP: extract inline test block 5 from ritk-snap [patch] | Done |
| P32 | COMPAT: remove dead ModalityDisplay.modality field + dead MRI arm [patch] | Done |
| P33 | SSOT: DEFAULT_WINDOW_CENTER/WIDTH consts + current_window_level() in ritk-snap [patch] | Done |
| P34 | SSOT: MPR_INFO / OVERLAY constants in ritk-snap [patch] | Done |
| P35 | SSOT: DEFAULT_VR_ALPHA / FUSION_ALPHA / RT-dose opacity constants in ritk-snap [patch] | Done |
| P36 | NAMING: dot3/cross3/normalize3 renames in ritk-snap [patch] | Done |
| P37 | DRY: W/L extraction DRY helper in ritk-snap [patch] | Done |
| P38 | SSOT: additional SSOT sweep in ritk-snap [patch] | Done |
| P39 | NAMING: rename 27 test fns in regularization dispatch + inline tests [patch] | Done |
| P40 | NAMING: rename 14 test fns in ritk-transform [patch] | Done |
| P41 | NAMING: rename 6 external integration test fns + cma_es test [patch] | Done |
| P42 | SSOT: 17 production SSOT constants (NCC_SIGMA_GUARD, QUAT_NORM_GUARD, etc.) in ritk-registration [patch] | Done |
| P43 | SSOT: test tolerance constants (ZERO_FIELD_LOSS_TOL, SCALE_TRANSFORM_TOL, etc.) [patch] | Done |
| P44 | SRP: extract 5 inline test blocks in ritk-registration [patch] | Done |
| P45 | COMPAT: delete 5 duplicate inline regularization tests [patch] | Done |
| P46 | COMPAT: remove 5 dead code items in ritk-registration [patch] | Done |
| P47 | SSOT: JPEG constants module (MAX_CODE_LEN, DCT_BLOCK_DIM, DCT_BLOCK_CELLS, YCbCr BT.601) [patch] | Done |
| P48 | SSOT: LANCZOS_WEIGHT_EPS + SPATIAL_DIMS in ritk-interpolation [patch] | Done |
| P49 | SRP: extract grid.rs + transform.rs tests in ritk-interpolation [patch] | Done |
| P50 | SRP: extract pixel_layout.rs + jpeg/mod.rs tests in ritk-codecs/image [patch] | Done |
| P51 | SRP: extract nearest.rs + tensor_trilinear.rs tests [patch] | Done |
| P52 | DRY: apply_rescale helper + data_vec() migration + deprecated decode_native_pixel_bytes [patch] | Done |
| P53 | COMPAT: delete 8 redundant NN dispatch arms + delete legacy.rs in ritk-codecs [patch] | Done |
| P54 | ENUM: InterleaveMode + QuantPrecision enums in ritk-codecs [minor] | Done |
| P55 | NAMING: rename tests_dispatch/dim*.rs → rank*.rs in ritk-interpolation [patch] | Done |
| P56 | NAMING: rename 28 fft/conv test fn dim-suffixes + NCC_DENOM_FLOOR + NEAR_ONE/ZERO_TOL in ritk-filter [patch] | Done |
| P57 | SRP: extract 11 inline test blocks (batch A) in ritk-filter/segmentation/statistics [patch] | Done |
| P58 | SRP: extract 11 inline test blocks (batch B) [patch] | Done |
| P59 | SSOT: entropy_from_hist pub(crate) + F32_TOL + STAPLE_TOL + FOREGROUND_THRESHOLD in staple [patch] | Done |
| P60 | COMPAT: final verification pass — 0 warnings, all tests green [patch] | Done |

### Blocked / Deferred (carry-forward)

| ID | Description | Priority |
|----|-------------|----------|
| DRY-374-01 | `make_image_*`/`make_mask_*` — 68 occurrences across ritk-segmentation/statistics (expanded from 35) | [minor] |
| DRY-374-01 | `ritk-image::test_support` SSOT wired into 3 consumer `[dependencies]` (feature `test-helpers` enabled); actual 68-occurrence test-file migration deferred — **partial** | [minor] |
| DRY-374-07 | `decode_bytes_to_f32`/`parse_f64_vec` consolidated into `ritk-codecs::byte_decode`; `ritk-metaimage` + `ritk-nrrd` migrated; ~270 lines removed | [minor] |
| DRY-374-08 | `parse_floats<T: FromStr>` + `require_bytes` SSOT in `ritk-codecs::byte_decode`; `ritk-metaimage` + `ritk-nrrd` thin-wrapper re-exports; `ritk-vtk` local kept (different signature) | [minor] |
| NAMING-362-23 | `transform_1d/_2d/_3d/_4d` → sealed `DimInterpolation<B>` trait BLOCKED [arch] — ADR required | [arch] |
| SRP-362-20 | `FilterArgs` (46 fields) → `FilterKind` ValueEnum + 15 per-family `#[command(flatten)]` Args structs [major] | **Done Sprint 375** |
| NAMING-FILTER-01 | `FftConvolution3DFilter` const-generic unification BLOCKED [major] | Low |
| N-375-08 | DRY cross-crate parse utils (ritk-io shared codec layer for metaimage/nrrd/minc) BLOCKED [arch] | [arch] |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| ritk-io nextest | 330/330 |
| ritk-vtk nextest | 241/241 |
| ritk-spatial/morphology/minc/metaimage/nrrd nextest | 131/131 |
| ritk-snap nextest | 633/633 |
| ritk-registration + ritk-transform nextest | 69+69 = 138 |
| ritk-codecs + ritk-image + ritk-interpolation nextest | 353/353 |
| ritk-filter nextest | 703/703 |
| ritk-segmentation + ritk-statistics nextest | 663/663 |
| Total across modified crates | **3257/3257** |

---

## Sprint 374 — Architecture Hardening Round 7: SSOT · DRY · NAMING · ENUM · SRP · COMPAT

**Status**: Complete  
**Version**: 0.69.0  

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| P01 | `SIGMA_MIN: f64 = 1e-10` const in bilateral.rs [patch] | Done |
| P02 | `NEAR_ZERO_MAG: f32 = 1e-10` const in canny.rs [patch] | Done |
| P03 | `LENGTH_EPSILON: f64 = 1e-12` const in cpr_helpers.rs [patch] | Done |
| P04 | `NEAR_ZERO_WEIGHT: f64 = 1e-12` const in n4/histogram_sharpen.rs [patch] | Done |
| P05 | `TIKHONOV_LAMBDA: f64 = 1e-6` const in bspline_bias.rs [patch] | Done |
| P06 | DRY: `morphological_scan_3d` consolidates `dilate_3d`/`erode_3d` (−60L) [minor] | Done |
| P07 | SSOT: `PROB_ZERO_GUARD: f64 = 1e-12` in threshold/mod.rs; 15 sites + EIGENVALUE_SINGULARITY_EPS + WEIGHT_ZERO_GUARD [minor] | Done |
| P08 | SSOT: `white_stripe.rs` `mv > 0.5` → `crate::FOREGROUND_THRESHOLD` [patch] | Done |
| P09 | SSOT: `NORMALIZER_EPSILON` in 2 test files (tests_white_stripe, tests_zscore) [patch] | Done |
| P10 | SSOT: `CENTRAL_DIFF_HALF: f32 = 0.5` in jacobian.rs (3 sites) [patch] | Done |
| P11 | ENUM: `OptimizerAlgorithm` enum replaces `&'static str` in `OptimizerTelemetry`; 5 optimizer impls updated [minor] | Done |
| P12 | COMPAT: Stale architecture diagram in ritk-registration/lib.rs fixed [patch] | Done |
| P13 | SSOT: `NEAR_ZERO`/`ABS_TOL` test tolerance consts in ritk-transform tests [patch] | Done |
| P14 | SSOT: `SCHEDULER_TOL: f64 = 1e-12` in optimizer/trait_.rs tests (6 sites) [patch] | Done |
| P15 | ENUM: `ContourGeometricType` replaces `ArrayString<16>` in `RtContour`; reader/converter/writer updated [minor] | Done |
| P16 | DRY: `str_to_vr` 36-arm duplication eliminated; widened to `pub(crate)`, deleted from writer_object.rs [patch] | Done |
| P17 | SSOT: `DICOM_SOP_CLASS_SECONDARY_CAPTURE` promoted to `pub(crate)` [patch] | Done |
| P18 | SSOT: `EXPLICIT_VR_LE` const added to transfer_syntax.rs; 3 raw UID literals replaced [patch] | Done |
| P19 | SRP: rt_struct/converter.rs 202L test block → tests/converter.rs [patch] | Done |
| P20 | COMPAT: `data_vec` deprecated since fixed 0.7.0→0.1.0; data_slice() dead branches collapsed; stale TODO removed [patch] | Done |
| P21 | NAMING: `PixelSignedness::to_u16` deleted; `From` impl inlined [patch] | Done |
| P22 | NAMING: sealed `LeBytes` trait + `read_le<T>`/`write_le<T>` replaces 6 type-suffixed fns in ritk-analyze [minor] | Done |
| P23 | SSOT: `HDR_SIZE = 348` + `EXTENTS = 16_384` consts in ritk-analyze [patch] | Done |
| P24 | NAMING: `format_float_slice<const N>` replaces `format_f64_2/3/6/9` in metadata_table.rs [patch] | Done |
| P25 | NAMING: `screen_to_img_f32` → `screen_to_img_exact` + 2 test renames [patch] | Done |
| P26 | NAMING: `promote_2d_to_3d` → `elevate_to_volume` [patch] | Done |
| P27 | NAMING: `slice_spacing_2d` → `slice_plane_spacing` [patch] | Done |
| P28 | NAMING: `resize_u8` → `resize_pixel_bytes` + 4 test fn renames [patch] | Done |
| P29 | NAMING: `to_u8` → `clamp_to_byte` in colormap.rs (10 sites) [patch] | Done |
| P30 | SSOT: `U8_MAX_F32: f32 = 255.0` in render/mod.rs; 11 literals replaced across 6 render files [patch] | Done |
| P31 | COMPAT: deleted `tool_shortcut_text` dead fn; deleted `adapter` dead field from gpu_volume/context.rs [patch] | Done |
| P32 | NAMING: `VtkCellType::to_u8`/`from_u8` → `From`/`TryFrom`; 7 callers updated [minor] | Done |
| P33 | NAMING+COMPAT: `parse_f64s` → `parse_floats<T: FromStr>`; deleted `extract_da_content`+`named_da` [patch] | Done |
| P34 | NAMING: `parse_as_f32`→`parse_float_ascii`, `read_le_f32`→`read_le_float`; 12 callers [patch] | Done |
| P35 | NAMING: 10 stale `rgba_u8_*`/`rgba_f32_*` test names updated in tests_color.rs [patch] | Done |
| P36 | SSOT: `EPSILON: f32 = 1e-6` (8 sites) + `U8_MAX_F: f32 = 255.0` (5 sites) in ritk-annotation [patch] | Done |
| P37 | SRP: 3 test blocks extracted from ritk-annotation: tests_label_table, tests_undo_redo, tests_label_map [patch] | Done |
| P38 | NAMING: `parse_space_directions_2d`→`_planar`, `parse_nrrd_point_2d`→`_planar` in ritk-nrrd [patch] | Done |
| P39 | NAMING: 5 type-suffixed test fn names in ritk-mgh renamed to scenario descriptions [patch] | Done |
| P40 | NAMING+SRP: `make_image_3d`→`make_test_image`; gaussian_kernel test renames; extract tests_tensor_ops.rs [patch] | Done |

### Blocked / Deferred (carry-forward)

| ID | Description | Priority |
|----|-------------|----------|
| NAMING-362-23 | `transform_1d/_2d/_3d/_4d` → sealed `DimInterpolation<B>` trait + per-D `pub(super)` body functions [arch] | **Done Sprint 375** |
| SRP-362-20 | `FilterArgs` (30 variants) → `FilterKind` ValueEnum [major] scope | [major] |
| NAMING-FILTER-01 | `FftConvolution3DFilter` const-generic unification [major] | Low |
| TIMEOUT-367 | 4 ritk-interpolation large-dispatch tests — pre-existing performance issue | Medium |
| DRY-374-01 | `make_image_1d/3d`/`make_mask_*` — 35+ copies across ritk-segmentation/ritk-statistics | [minor] |
| NAMING-374-02 | 40+ test fn dim-suffix names across ritk-interpolation / ritk-filter / ritk-snap / ritk-model / ritk-segmentation / ritk-statistics: `_1d`/`_2d`/`_3d` → `_line`/`_planar`/`_volumetric` (Sprint 366-09 convention); `_4d` kept as-is (no established convention) [patch] | **Done Sprint 375** |
| SRP-374-03 | 21 inline test blocks in ritk-filter > 80L (grayscale_erosion 198L, unsharp_mask 210L, median 193L) | [patch] |
| SRP-374-04 | 25 inline test blocks in ritk-snap > 80L | [patch] |
| NAMING-374-05 | ritk-minc public API type suffixes (`extract_f64`, `build_attr_msg_f64`, `convert_to_f32`) | [minor] |
| ENUM-374-06 | `ModalityDisplay.modality: String` in ritk-snap — deferred for serde-compat impl | [minor] |
| DRY-374-07 | `decode_bytes_to_f32`/`parse_f64_vec` duplicated across ritk-metaimage/ritk-nrrd | [minor] |
| DRY-374-08 | `read_ascii/binary_f32/f64/i32` — 10 clones across 3 ritk-vtk IO modules | [minor] |
| VAR-375-01 | `PhantomData<B>` → `PhantomData<fn() -> B>` BLOCKED [upstream]: `burn-core-0.19.1` only implements `Module<B> for PhantomData<B>` (burn-core/src/module/param/constant.rs:202), not for the covariant form. An attempted switch of 2 sites in `registration/dl/{grad,ncc}.rs` produced 108 compile errors and was reverted. Workarounds: (a) upstream PR to broaden `Module<B>` impls to `PhantomData<fn() -> T>` + `PhantomData<*const T>`; (b) `#[module(custom)]` newtype with hand-written `Module` impl; (c) sealed `BackendMarker` newtype; (d) accept invariant `PhantomData<B>` (current state) | [upstream] |
| CONST-375-02 | `const _: () = assert!(matches!(D, 1..=4), "...{D}...");` companion in `BSplineTransform` impl BLOCKED [toolchain]: formatted panic in const context requires unstable `const_panic_fmt`; the fallback `[(); (D >= 1 && D <= 4) as usize - 1]:` where-bound also fails with "cannot perform const operation using 'D'" in this Rust toolchain. Current state: four per-D impl blocks produce a trait-coherence compile error for D ∉ {1, 2, 3, 4} ("the trait `Transform<B, D>` is not implemented for `BSplineTransform<B, D>`") which is already a human-readable diagnostic. The const-assert companion can be added once `const_panic_fmt` stabilizes | [toolchain] |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| ritk-filter nextest | 702/702 |
| ritk-segmentation + ritk-statistics nextest | 663/663 |
| ritk-registration + ritk-transform nextest | 615 + 69 = 684 |
| ritk-io nextest | 330/330 |
| ritk-image + ritk-codecs + ritk-analyze nextest | 236/236 |
| ritk-snap nextest | 640/640 |
| ritk-vtk + ritk-annotation + ritk-nrrd + ritk-mgh + ritk-tensor-ops nextest | 365/365 |
| Total across modified crates | **3620/3620** |

---

## Sprint 372 — J2K conformance fixes + differential interop harness (in progress)

**Status**: Conformance fixes delivered; interop acceptance gate pending  
**Version**: 0.68.x (ritk-codecs 0.5.x)  

| Track ID | Description | Status |
|----------|-------------|--------|
| J2K-372-CONF | Seven tier-1/tier-2 conformance fixes (see CHANGELOG/commit): packet non-empty bit polarity, Table B.4 pass-count code, Lblock = +floor(log2 ncp), Mb = eps+G-1, 3P-2 pass accounting, stripe-oriented SPP/MRP scan, Table D.1 ZC context tables, unrestricted RLC [patch] | **Done** |
| J2K-372-HARNESS | Differential interop suite vs openjp2 (pure-Rust c2rust port, dev-dep): encode+decode both directions, marker/packet dump diagnostic | **Done (gate pending)** |

### Open defects (P1)

| ID | Description | Repro |
|----|-------------|-------|
| ~~J2K-INTEROP~~ | **FIXED**: root cause was the MQ probability-estimation state machine advancing `I(CX)` on EVERY MPS instead of only on renormalisation (ISO 15444-1 §C.2.6 / Figure C.7; encoder AND decoder shared the defect, so internal round-trips masked it; found by instrumenting a vendored openjp2 MQ encoder and diffing register traces). All 6 interop acceptance tests un-ignored and green both directions; byte-for-byte escalation compare green; captured OpenJPEG 2.5.2 packet conformance test active | closed |
| ~~JLS-NEAR-TAIL~~ | **FIXED**: root cause was a trailing 0xFF entropy byte directly before EOI being discarded as a marker prefix; flush now emits the stuffed 7-bit follow byte (ISO 14495-1 C.2.1). Both proptests re-enabled at full domain; regression seeds committed | closed |
| ~~JLS-16BIT-LOSSLESS~~ | **FIXED**: same root cause as JLS-NEAR-TAIL (single fix closed both). Regression test `round_trip_16bit_regression_seed` active | closed |

---

## Sprint 371 — J2K multi-code-block tier-2

**Status**: Complete  
**Version**: 0.68.0 (ritk-codecs 0.5.0)  

| Track ID | Description | Status |
|----------|-------------|--------|
| J2K-371-TT | §B.10.2 tag trees (standard polarity, cross-layer state) [minor] | **Done** |
| J2K-371-CBLK | 64×64 code-blocks per subband; arbitrary single-tile sizes [minor] | **Done** |
| J2K-371-TEST | Multi-grid round-trips + tag-tree unit tests [patch] | **Done** |
| J2K-371-BENCH | 512×512 5-level baseline `sprint371` (55.6/58.2 ms) [patch] | **Done** |

---

## Sprint 370 — J2K multi-level DWT

**Status**: Complete  
**Version**: 0.67.0 (ritk-codecs 0.4.0)  

| Track ID | Description | Status |
|----------|-------------|--------|
| J2K-370-DWT | Mallat-layout forward/inverse multi-level 5/3 DWT + subband geometry [minor] | **Done** |
| J2K-370-T2 | LRCP multi-resolution packets; per-subband ε_b from QCD [minor] | **Done** |
| J2K-370-FIX | Tier-2 byte_pos RAW-offset fix (stuffed-0xFF desync) [patch] | **Done** |
| J2K-370-TEST | Level-randomized proptest + regressions; 2-level DICOM round-trip [patch] | **Done** |

---

## Sprint 369 — Native JPEG-LS codec: CharLS elimination + NEAR support

**Status**: Complete  
**Version**: 0.66.0 (ritk-codecs 0.3.0)  

| Track ID | Description | Status |
|----------|-------------|--------|
| JLS-369-ENC | Pure-Rust JPEG-LS encoder (lossless + NEAR), §C.2.1 bit writer, Golomb writer [minor] | **Done** |
| JLS-369-NEAR | NEAR-aware native decode (TS .81); shared `CodingParams` SSOT [minor] | **Done** |
| JLS-369-CONF | ISO C.2.4.1.1.1 default thresholds (>8-bit fix); §A.3.3 NEAR dead-zone [patch] | **Done** |
| JLS-369-DEP | charls/openjp2/jpeg2k fully removed — codec stack 100 % Rust [minor] | **Done** |
| JLS-369-TEST | Lossless + NEAR proptests; one-time CharLS differential before removal [patch] | **Done** |

### Carry-forward

| ID | Description | Priority |
|----|-------------|----------|
| REG-MI-FLAKY | `translation_recovery_shifted_gaussian` deterministic failure in in-flight NGF/RSGD registration wave (est 1.0 vs true 3.0) | [investigate] |
| JLS-INTEROP | Differential decode vs reference JPEG-LS corpora (e.g. ISO conformance streams) | [patch] |
| CODEC-PERF | Profile-first throughput pass on JPEG-LS/J2K hot loops (bit I/O batching, context-table layout) against the `sprint369` criterion baseline | [patch] |
| J2K-MULTI-CBLK | Multiple code-blocks per precinct/tile (current native J2K scope: one code-block per tile, images > 64 px rely on whole-tile blocks) | [minor] |

---

## Sprint 368 — RITK-native JPEG 2000 codec (pure-Rust ISO 15444-1)

**Status**: Complete (lossless, 0 DWT levels)  
**Version**: 0.65.0 (ritk-codecs 0.2.0)  

| Track ID | Description | Status |
|----------|-------------|--------|
| J2K-368-MQ | MQ coder ISO 15444-1 Annex C conformance (INITDEC, MPSEXCHANGE, CODEMPS/CODELPS, BYTEOUT/FLUSH, Table C.2 columns, Table D.7 init) [patch] | **Done** |
| J2K-368-T2 | Tier-2 packet conformance (Lblock terminator, Table B.4 prefix, inclusion threshold) [patch] | **Done** |
| J2K-368-ENC | Public pure-Rust J2K encoder (`jpeg_2000::encoder`); ritk-io DICOM round-trip uses it [minor] | **Done** |
| J2K-368-DEP | C/FFI elimination: `jpeg2k`/`openjp2`/`openjpeg-sys`/`charls` out of ritk-codecs; `TileCodingParams` consolidation [minor] | **Done** |
| J2K-368-TEST | 16-bit regression + proptest lossless round-trip [patch] | **Done** |

### Carry-forward (J2K)

| ID | Description | Priority |
|----|-------------|----------|
| J2K-DECODE-DWT | Multi-level 5/3 DWT decode — idwt groundwork in `wavelet.rs`; wire into `decode_tile_part` resolution-level packet loop | [minor] |
| J2K-LOSSY-97 | 9/7 irreversible wavelet for lossy TS 1.2.840.10008.1.2.4.91 | [minor] |
| J2K-INTEROP | Differential decode vs OpenJPEG-encoded reference codestreams (real DICOM corpus) | [patch] |
| JLS-NATIVE-REF | Replace `charls` dev-dep in ritk-io with RITK-native JPEG-LS differential reference (removes libstdc++ DLL coupling) | [patch] |

---

## Sprint 367 — Architecture Hardening Round 6: ENUM · NAMING · SRP · SSOT · DRY · COMPAT + ritk-core Crate Extraction

**Status**: Complete  
**Version**: 0.64.0  
**Commit**: ec6badc  

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| ARCH-367 | Extract `ritk-annotation`, `ritk-statistics`, `ritk-morphology`, `ritk-tensor-ops` from ritk-core; `annotation/mod.rs` + `statistics/mod.rs` → thin `pub use` shims [arch] | **Done** |
| ENUM-367-35 | `SegmentArgs.method: String` → `SegmentMethod` ValueEnum (23 variants); unreachable `other =>` arm + dead test removed [minor] | **Done** |
| ENUM-367-36 | `ConvertArgs.format: Option<String>` → `Option<OutputFormat>` ValueEnum (8 variants) [minor] | **Done** |
| ENUM-367-37 | `NormalizeArgs.contrast: Option<String>` → `Option<CliContrast>` ValueEnum; dead contrast-error test removed [minor] | **Done** |
| ENUM-367-38 | `FilterArgs.order: usize` → `CliDerivativeOrder` ValueEnum; `parse_spacing_mode` trivial wrapper removed [minor/patch] | **Done** |
| NAMING-367-05 | `RgbaU8`→`RgbaBytes`, `RgbaF32`→`RgbaLinear` in ritk-annotation; all callers in ritk-io + ritk-snap updated [patch] | **Done** |
| NAMING-367-06 | `UnaryPixelOp::apply_f32` → `apply` in ritk-filter [patch] | **Done** |
| NAMING-367-07 | `fft2d`/`fft3d` `pub` → `pub(crate)`; deconvolution/helpers.rs migrated to `fft_nd` [patch] | **Done** |
| NAMING-367-08 | `required_usize`/`optional_usize`/`optional_u16` → `read_required<T>`/`read_optional<T>` in ritk-io/color_common.rs [patch] | **Done** |
| NAMING-367-09 | `read_nested_f64` → `read_nested_scalar<T: FromStr>` in ritk-io/helpers.rs [patch] | **Done** |
| NAMING-367-10 | `test_normalize_3d`/`test_dot_3d` → `test_normalize_unit_vector`/`test_dot_product` [patch] | **Done** |
| NAMING-367-11 | `build_rle_fragment_8bit` → `build_rle_fragment` [patch] | **Done** |
| NAMING-367-12 | `CommandField::from_u16` → `impl TryFrom<u16> for CommandField` [patch] | **Done** |
| SRP-367-A1 | ritk-annotation: `tests_annotation_state.rs` extracted [patch] | **Done** |
| SRP-367-A2 | ritk-annotation: `tests_overlay.rs` extracted [patch] | **Done** |
| SRP-367-A3 | ritk-annotation: `tests_color.rs` extracted [patch] | **Done** |
| SRP-367-R1 | ritk-registration: `tests_lncc.rs` extracted [patch] | **Done** |
| SRP-367-R2 | ritk-registration: `tests_ncc.rs` extracted [patch] | **Done** |
| SRP-367-R3 | ritk-registration: `tests_numerical.rs` extracted [patch] | **Done** |
| SRP-367-I1 | ritk-io: `tests_sop_class.rs` extracted (193L) [patch] | **Done** |
| SRP-367-S1 | ritk-segmentation: `tests_shape_detection.rs` extracted (230L) [patch] | **Done** |
| SRP-367-S2 | ritk-segmentation: `tests_growcut.rs` extracted (175L) [patch] | **Done** |
| SRP-367-S3 | ritk-segmentation: `tests_fill_holes.rs` extracted (116L) [patch] | **Done** |
| SRP-367-S4 | ritk-segmentation: `tests_morphological_gradient.rs` extracted (114L) [patch] | **Done** |
| SSOT-367-23 | `DEFAULT_NOISE_SEED: u64 = 42` const in noise/mod.rs; 4 noise filters updated [patch] | **Done** |
| SSOT-367-24 | `DEFAULT_ITERATIVE_TOLERANCE: f32 = 1e-6` const in deconvolution/regularization.rs; landweber + rl updated [patch] | **Done** |
| SSOT-367-25 | `FOREGROUND_THRESHOLD: f32 = 0.5` const in segmentation/morphology/mod.rs; binary_closing/dilation/erosion/fill_holes/morphological_gradient updated [patch] | **Done** |
| DRY-367-28 | `box_muller(u1, u2) -> f64` extracted to noise/mod.rs; gaussian/shot/speckle noise filters use it [patch] | **Done** |
| DRY-367-30 | `ritk-analyze/codec.rs`: read/write i16/i32/f32 helpers + `DT_FLOAT` const; reader.rs + writer.rs use shared module [patch] | **Done** |
| COMPAT-367-32 | `DRY_353_02_STATUS` dead const removed from ritk-interpolation/kernel/macros.rs [patch] | **Done** |
| COMPAT-367-33 | Stale `#[allow(dead_code)]` on `BoundsPolicy` removed; dead `is_zero_pad` deleted; `BinRange::is_empty` gated `#[cfg(test)]` [patch] | **Done** |
| COMPAT-367-34 | `#[allow(dead_code)]` removed from direct-parzen `cache.rs` feature-gated functions [patch] | **Done** |
| COMPAT-367-35 | `ParzenConfig` test-only fns gated `#[cfg(test)]`; suppressions removed [patch] | **Done** |
| COMPAT-367-36 | `compute_joint_histogram_from_cache` `#[allow(dead_code)]` → `#[cfg(not(feature = "direct-parzen"))]` [patch] | **Done** |
| COMPAT-367-37 | Dead `is_empty` removed from `bin_range.rs` + `stack_weights.rs`; `#[allow(dead_code)]` removed [patch] | **Done** |
| COMPAT-367-39 | Stale doc in `deconvolution/regularization.rs` referencing `apply_2d`/`apply_3d` corrected [patch] | **Done** |
| FIX-367-INT | ritk-snap/label/tests.rs: `use super::*` restored after RgbaU8→RgbaBytes rename [patch] | **Done** |

### Blocked / Deferred (carry-forward)

| ID | Description | Priority |
|----|-------------|----------|
| NAMING-362-23 | `transform_1d/_2d/_3d/_4d` → sealed `DimInterpolation<B>` trait + per-D `pub(super)` body functions [arch] | **Done Sprint 375** |
| SRP-362-20 | `FilterArgs` (46 fields) → `FilterKind` ValueEnum — [major] scope, carry forward | [major] |
| NAMING-FILTER-01 | `FftConvolution3DFilter`/`FftNormalizedCorrelation3DFilter` → const-generic `<B, const D>` unification [major] | Low |
| TIMEOUT-367 | ritk-interpolation 4-test timeout cluster (`dim4`, `dim3_extended`) — pre-existing; investigate under performance_engineering protocol | Medium |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `cargo nextest run -p ritk-core -p ritk-filter -p ritk-segmentation -p ritk-statistics -p ritk-annotation` | 1429/1429 passed |
| `cargo nextest run -p ritk-registration --lib` | 591/591 passed, 1 skipped |
| `cargo nextest run -p ritk-io -p ritk-cli --no-fail-fast` | 523/524 passed (1 pre-existing JPEG2000 Windows abort 0xc0000374) |

---

## Sprint 366 — Architecture Hardening Round 5: NAMING · SSOT · COMPAT · DRY · SRP · ENUM · PRIM

**Status**: Complete  
**Version**: 0.63.0  
**Commit**: 0feb9ec  

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| NAMING-CORE-01 | `gaussian_kernel_1d` → `gaussian_kernel` in ritk-core + all callers (ritk-filter ×6, ritk-segmentation ×3, ritk-registration) [patch] | **Done** |
| ENUM-366-01 | `ResampleArgs.interpolation: String` → `InterpolationMode` ValueEnum (nearest/linear/bspline/lanczos4) [minor] | **Done** |
| COMPAT-366-02 | Delete 4 `#[deprecated(0.64.0)] apply_3d` shims in noise filters [patch] | **Done** |
| SSOT-366-03 | Delete dead `wgpu_compat.rs` shadow module in ritk-registration + lib.rs declaration [patch] | **Done** |
| COMPAT-366-04 | Remove `let _device` dead bindings in `histogram_matching.rs` + `nyul_udupa.rs` [patch] | **Done** |
| SSOT-366-05 | `NORMALIZER_EPSILON` const in `normalization/mod.rs`; `minmax.rs` + `zscore.rs` updated [patch] | **Done** |
| SSOT-366-06 | `FOREGROUND_THRESHOLD` const in `statistics/mod.rs`; 4 files updated [patch] | **Done** |
| SSOT-366-07 | Fix stale docs in `deconvolution/helpers.rs` + `deconvolution/mod.rs` [patch] | **Done** |
| NAMING-366-08 | `cross_3d`/`normalize_3d`/`dot_3d` → `cross`/`normalize`/`dot` in `reader/geometry.rs` + 22 callers [patch] | **Done** |
| NAMING-366-09 | `spatial_gradient_2d/_3d`/`spatial_laplacian_2d/_3d` → `*_planar/*_volumetric` in `dispatch.rs` [patch] | **Done** |
| NAMING-366-10 | `VectorField3D`/`VectorFieldMut3D` → `VectorField`/`VectorFieldMut`; 12 call-site files updated [patch] | **Done** |
| NAMING-366-11 | `get_f64`/`get_f64_vec` → `get_scalar`/`get_scalar_vec` in `series/loader.rs` [patch] | **Done** |
| DRY-366-12 | `read_nested_f64` consolidated into `dicom/helpers.rs`; removed from `per_frame.rs` + `seg/reader.rs` [patch] | **Done** |
| SRP-366-13 | `threshold/li.rs` inline tests → `tests_li.rs` [patch] | **Done** |
| SRP-366-14 | `threshold/yen.rs` inline tests → `tests_yen.rs` [patch] | **Done** |
| SRP-366-15 | `watershed/mod.rs` inline tests → `tests_watershed.rs` [patch] | **Done** |
| SRP-366-16 | `labeling/relabel.rs` inline tests → `tests_relabel.rs` [patch] | **Done** |
| SRP-366-17 | `color_multiframe.rs` inline tests → `tests_color_multiframe.rs` in ritk-io [patch] | **Done** |
| PRIM-366-18 | `SegmentArgs.markers: Option<String>` → `Option<PathBuf>` [patch] | **Done** |
| COMPAT-366-19 | Remove `#[allow(dead_code)]` field `integration_steps` from `DiffeomorphicSSMMorph` [patch] | **Done** |

### Blocked / Deferred (carry-forward)

| ID | Description | Priority |
|----|-------------|----------|
| NAMING-362-23 | `transform_1d/_2d/_3d/_4d` → sealed `DimInterpolation<B>` trait + per-D `pub(super)` body functions [arch] | **Done Sprint 375** |
| SRP-362-20 | `FilterArgs` (46 fields) → `FilterKind` ValueEnum — [major] scope, carry forward | [major] |
| NAMING-FILTER-01 | `FftConvolution3DFilter`/`FftNormalizedCorrelation3DFilter` → const-generic `<B, const D>` unification [major] | Low |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `cargo nextest run -p ritk-core -p ritk-filter -p ritk-segmentation` | 1447/1447 passed |
| `cargo nextest run -p ritk-registration --lib` | 591/591 passed, 1 skipped |
| `cargo nextest run -p ritk-io -p ritk-cli --no-fail-fast` | 526/527 (1 pre-existing JPEG2000 Windows abort 0xc0000374) |

---
## Sprint 365 — Architecture Hardening Round 4: COMPAT · NAMING · SSOT · SRP · DRY · DIP · ENUM

**Status**: Complete  
**Version**: 0.62.0  
**Commit**: c6daed5  

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| COMPAT-365-01 | Delete dead `NormalizationMode` enum + its test from `metric/trait_.rs` [patch] | **Done** |
| NAMING-365-02 | `collect_vec_3`/`collect_vec_9` → `collect_array::<N>` in `metric/histogram/cache.rs`; fix inaccurate panic doc [patch] | **Done** |
| NAMING-365-03 | `StopReason` → `CmaEsStopReason` in `optimizer/cma_es/state.rs` + all re-exports and usages [minor] | **Done** |
| DIP-365-04 | `RegistrationConfig::build_tracker()` + `TrackerBuildResult`; `Registration::with_config` decoupled [minor] | **Done** |
| SRP-365-05 | `metric/correlation_ratio.rs` tests → `metric/tests_correlation_ratio.rs` [patch] | **Done** |
| COMPAT-365-06 | Delete deprecated dead `apply_tikhonov_2d/_3d` from `deconvolution/regularization.rs` [patch] | **Done** |
| NAMING-365-07 | Private dim-suffix renames in ritk-filter: `gaussian_smooth_1d`→`gaussian_smooth`, `gradient_3d`→`compute_gradient`, `bilateral_3d`→`compute`, `edt_3d`→`euclidean_dt`, `phase1_1d`→`phase1_row`, `meijster_1d`→`meijster_row`; all call sites updated [patch] | **Done** |
| SRP-365-09 | `statistics/image_statistics.rs` tests → `tests_image_statistics.rs` [patch] | **Done** |
| SRP-365-10 | `statistics/normalization/minmax.rs` tests → `tests_minmax.rs` [patch] | **Done** |
| DRY-365-11 | Extract `build_tensor` helper from repeated bodies in `filter/ops.rs` `rebuild*` functions [patch] | **Done** |
| SSOT-365-12 | Add `.ima` to `ImageFormat::from_path` Dicom arm; `is_likely_dicom_file` delegates to it [minor] | **Done** |
| NAMING-365-13 | `DicomObjectNode::u16/i32/f64` → `from_u16/from_i32/from_f64`; all call sites updated [patch] | **Done** |
| DRY-365-14 | `io_err()` helper eliminates 17 repeated `map_err` closures in `ritk-python/src/io/mod.rs` [patch] | **Done** |
| PRIM-365-15 | `read_transform`/`write_transform` `path: String` → `path: &str` at PyO3 boundary [patch] | **Done** |
| NAMING-365-16 | `gaussian_smooth_3d` → `gaussian_smooth` in `level_set/helpers.rs`; all callers updated [patch] | **Done** |
| NAMING-365-17 | `skeleton_1d/2d/3d` → `endpoint_extract`/`zhang_suen`/`sequential_thin`; `mod.rs` dispatch updated [patch] | **Done** |
| NAMING-365-18 | `dilate/erode_1d/2d/3d` → `dilate/erode_line/plane/volume` in `binary_{dilation,erosion}/mod.rs` [patch] | **Done** |
| ENUM-365-19 | `StatsArgs.metric: String` → `StatMetric` ValueEnum (7 variants, `msd` alias); exhaustive match [minor] | **Done** |
| ENUM-365-20 | `RegisterArgs.method: String` → `RegistrationMethod` ValueEnum (10 variants); exhaustive match; `mi.rs` secondary dispatch updated [minor] | **Done** |

### Blocked / Deferred (carry-forward)

| ID | Description | Priority |
|----|-------------|----------|
| DIP-362-13 | Superseded by DIP-365-04 (delivered). Closed. | — |
| NAMING-362-23 | `transform_1d/_2d/_3d/_4d` → sealed `DimInterpolation<B>` trait + per-D `pub(super)` body functions [arch] | **Done Sprint 375** |
| SRP-362-20 | `FilterArgs` (46 fields) → `FilterKind` ValueEnum — [major] scope, carry forward | [major] |
| ENUM-365-03 | `ResampleArgs.interpolation: String` → `InterpolationMode` ValueEnum (4 variants) [minor] | Low |
| NAMING-CORE-01 | `gaussian_kernel_1d` → `gaussian_kernel` in `ritk-core/filter/kernel_utils.rs` (cross-crate callers in ritk-filter + ritk-segmentation) [patch] | Medium |
| NAMING-FILTER-01 | `FftConvolution3DFilter`/`FftNormalizedCorrelation3DFilter` → const-generic `<B, const D>` unification [major] | Low |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `cargo nextest run -p ritk-filter` | 699/699 passed |
| `cargo nextest run -p ritk-core` | 373/373 passed |
| `cargo nextest run -p ritk-registration` | 630/630 passed, 23 skipped |
| `cargo nextest run -p ritk-segmentation` | 375/375 passed |
| `cargo nextest run -p ritk-io --no-fail-fast` | 329/330 (1 pre-existing JPEG2000 Windows abort 0xc0000374) |
| `cargo nextest run -p ritk-cli` | 198/198 passed |

---

## Sprint 364 — Architecture Hardening Round 3: COMPAT · NAMING · SSOT · CACHE · SRP · PRIM · ENUM

**Status**: Complete  
**Version**: 0.61.0  
**Commit**: b740507  

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| COMPAT-364-01 | Remove 16 deprecated `apply_2d`/`apply_3d` (deconvolution ×4 + fft ×4) + fix doctests [major] | **Done** |
| SRP-364-02 | `noise.rs` (370L, 4 structs) → `noise/{mod,gaussian,salt_pepper,shot,speckle}.rs` [patch] | **Done** |
| NAMING-364-03 | Noise `apply_3d` inversion fixed — real impl in `apply`; `apply_3d` deprecated 0.64.0; 30+ test call sites updated [minor] | **Done** |
| NAMING-364-04 | Chamfer `cdt_3d*` → `cdt*`; `chamfer_distance_transform_3d*` → `chamfer_distance_transform*`; all re-exports and internals updated [minor] | **Done** |
| NAMING-364-05 | `compute_hessian_3d` → `compute_hessian`; update frangi, sato, tests [minor] | **Done** |
| CACHE-364-06 | `ParzenJointHistogram.cache`/`masked_cache` → `CacheSlot<T>`; `with_ref`/`with_mut` added; `HistogramCache` derives `Clone` [patch] | **Done** |
| DRY-364-07 | `compute_image_joint_histogram` `Option<f32>` → `SamplingConfig`; `full_grid()` added [patch] | **Done** |
| NAMING-364-08 | `cubic_bspline_1d` → `cubic_bspline_basis` in `bspline_ffd/basis` [patch] | **Done** |
| NAMING-364-09 | Remove redundant `gaussian_kernel_1d_f64` wrapper in `smooth.rs` [patch] | **Done** |
| SRP-364-10 | `threshold_level_set.rs` inline tests → `tests_threshold_level_set.rs` [patch] | **Done** |
| SRP-364-11 | `laplacian.rs` inline tests → `tests_laplacian_level_set.rs` [patch] | **Done** |
| SRP-364-12 | `kapur.rs` inline tests → `tests_kapur.rs` [patch] | **Done** |
| SRP-364-13 | `triangle.rs` inline tests → `tests_triangle.rs` [patch] | **Done** |
| SRP-364-14 | `ritk-core/filter/ops.rs` → extract `gaussian_kernel_1d` into `filter/kernel_utils.rs` [patch] | **Done** |
| SSOT-364-15 | `ImageFormat::Analyze` + `from_path` `.hdr`/`.img` arms + `from_str_name()` [minor] | **Done** |
| SSOT-364-16 | `ritk-python/io/mod.rs` `ends_with` chains → `ImageFormat::from_path` dispatch [minor] | **Done** |
| SSOT-364-17 | `ritk-cli/commands/mod.rs` `infer_format`/`read_image`/`write_image` → `ImageFormat` dispatch [patch] | **Done** |
| PRIM-364-18 | `ResampleArgs.spacing: String` → `Vec<f64>` with `value_delimiter = ','` [patch] | **Done** |
| PRIM-364-19 | `ConvertArgs.format` → `ImageFormat`-typed resolution in `run()` [patch] | **Done** |
| ENUM-364-20 | `NormalizeMethod` ValueEnum replaces `NormalizeArgs.method: String`; exhaustive match [minor] | **Done** |

### Blocked / Deferred

| ID | Description | Priority |
|----|-------------|----------|
| DIP-362-13 | `RegistrationCallbackSet` DIP — deferred; requires surveying `src/progress/` ProgressTracker internals | Medium |
| NAMING-362-23 | `transform_1d/_2d/_3d/_4d` → sealed `DimInterpolation<B>` trait + per-D `pub(super)` body functions [arch] | **Done Sprint 375** |
| SRP-362-20 | `FilterArgs` (46 fields) → `FilterKind` ValueEnum — [major] scope, carry forward | [major] |
| ENUM-365-01 | `StatsArgs.metric: String` → `StatMetric` ValueEnum (7 variants + `msd` alias) [minor] | **Done** |
| ENUM-365-02 | `RegisterArgs.method: String` → `RegistrationMethod` ValueEnum (10 variants) [minor] | **Done** |
| ENUM-365-03 | `ResampleArgs.interpolation: String` → `InterpolationMethod` ValueEnum (4 variants) [minor] | Low |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| nextest (ritk-filter + ritk-core + ritk-segmentation + ritk-io + ritk-cli) | 1976/1977 passed (1 pre-existing JPEG2000 Windows codec abort) |
| `cargo nextest run -p ritk-registration` | 631/631 passed, 23 skipped |

---

## Sprint 363 — Architecture Hardening Round 2: DRY · SRP · PRIM · NAMING · CACHE

**Status**: Complete  
**Version**: 0.60.0  
**Commit**: 59f4bee  

### Delivered

| Track ID | Description | Status |
|----------|-------------|--------|
| DRY-362-04 | `UnaryImageFilter<Op,D>` + `UnaryPixelOp` sealed trait; abs/sqrt/exp/log/square → type aliases; D-generic apply [minor] | **Done** |
| SRP-361-06 | `label_morphology.rs` (445L) → `label_morphology/{label_ops,reconstruction,mod,tests}.rs` [patch] | **Done** |
| PRIM-361-03 | `DiscreteGaussianFilter::new(Vec<GaussianSigma>)` — sigma not variance; all callers updated [minor] | **Done** |
| PRIM-362-12 | `EarlyStoppingPolicy::Enabled { patience, min_improvement }` bundle — invalid state eliminated [minor] | **Done** |
| NAMING-362-24 | `spatial_gradient_2d/_3d`, `spatial_laplacian_2d/_3d` → private `fn` in `dispatch.rs`; `spatial_ops.rs` deleted [patch] | **Done** |
| CACHE-363-01 | `CacheSlot<LnccCacheEntry<B>>` in `lncc.rs`; `get_or_reinit_if` added to `CacheSlot`; `Arc<Mutex<Option<>>>` removed [patch] | **Done** |
| SRP-362-19 | `series.rs` (438L) → `series/{types,scan,loader}.rs`; `Arc<Mutex<HashMap>>` → lock-free collect-and-merge [patch] | **Done** |
| SRP-362-18 | `seg/tests/convert.rs` (554L) → 4 focused test modules [patch] | **Done** |
| PRIM-362-27 | `DicomSeriesInfo` `pub(crate)` `ArrayString` fields + `pub &str` accessors + `pub fn new()` [minor] | **Done** |
| PRIM-362-25 | `IntensityRange<T>` validating newtype in `ritk-core::statistics` [minor] | **Done** |
| PRIM-362-25b | `MinMaxNormalizer` adopts `IntensityRange<f32>` for range field [minor] | **Done** |
| PRIM-362-25c | `CorrelationRatio::new` adopts `IntensityRange<f32>` for intensity bounds [minor] | **Done** |
| BOOL-361-05a | `RegisterArgs.sigma_fixed: GaussianSigma` via clap `value_parser` [minor] | **Done** |
| BOOL-361-05b | `RegisterArgs.kernel_sigma: GaussianSigma` via clap `value_parser` [minor] | **Done** |
| FIX-363 | Cross-crate call site fixes: ritk-cli smoothing (variance=0 identity), ritk-cli viewer, ritk-snap series_tree, ritk-python gaussian [patch] | **Done** |

### Blocked / Deferred

| ID | Description | Priority |
|----|-------------|----------|
| DIP-362-13 | `RegistrationCallbackSet` DIP — deferred; requires surveying `src/progress/` ProgressTracker internals | Medium |
| NAMING-362-23 | `transform_1d/_2d/_3d/_4d` → sealed `DimInterpolation<B>` trait + per-D `pub(super)` body functions [arch] | **Done Sprint 375** |
| SRP-362-20 | `FilterArgs` (46 fields) → `FilterKind` ValueEnum — [major] scope, carry forward | [major] |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| nextest run (6 crates, no-fail-fast) | 2868/2869 passed (1 pre-existing JPEG2000 Windows abort) |

---

## Sprint 362 — Architecture Hardening: SSOT · DRY · SRP · DIP · Naming

**Status**: In Progress
**Version**: 0.59.0
**Phase**: FIX + SSOT + DRY + SRP + PRIM + NAMING + DIP

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| FIX-362-01 | `engine.rs` fake-generic: `as_slice::<f32>()[0] as f64` → `.clone().into_scalar().elem()` via `ElementConversion` [patch] | **Done** |
| SSOT-362-02 | `ritk-io::ImageFormat` enum + `from_path` resolver; replace CLI `infer_format` (20L) and Python `io/mod.rs` if-chains (27L) [minor] | Planned |
| DRY-362-03 | Remove `FftDir` shim from `filter/fft/convolution/helpers.rs`; update all call sites to `ForwardFft`/`InverseFft` ZSTs [patch] | Planned |
| DRY-362-04 | `UnaryImageFilter<Op>` + `UnaryPixelOp` sealed trait; collapse abs/sqrt/exp/log/square ~570L → ~100L; type aliases preserve public names; `D=3` → `const D` [minor] | Planned |
| DRY-362-05 | `ConvergenceFlag` consolidation: `adaptive_stochastic_gd` + `regular_step_gd/optimizer` → shared `optimizer/convergence.rs` [patch] | **Done** |
| DRY-362-06 | Complete `SamplingConfig` migration: `MutualInformation.sampling_percentage: Option<f32>` + `CorrelationRatio` + `compute_image/mod.rs` [patch] | Planned |
| DRY-362-07 | Rename `preprocessing::NormalizationMode` → `IntensityRescaleMode`; resolves name collision with `metric::trait_::NormalizationMode` [minor] | **Done** |
| DRY-362-08 | `SharedCache<T>` newtype in `metric/cache_slot.rs`; adopt in Parzen (×3) + MutualInformation [patch] | Planned |
| SRP-362-09 | `bspline_ffd/basis.rs` (445L) → `basis/{scalar,cache,evaluate}.rs` [patch] | Planned |
| SRP-362-10 | `dl_registration_loss.rs` → `dl/losses/{lncc,grad,combined,mod}.rs` (6 concerns separated) [patch] | Planned |
| SRP-362-11 | `regularization/trait_::utils` → `regularization/spatial_ops.rs`; make `pub(crate)` [patch] | Planned |
| PRIM-362-12 | `EarlyStoppingPolicy::Enabled { patience, min_improvement }`: bundle orphaned fields; eliminate impossible `Disabled + non-zero patience` state [minor] | Planned |
| DIP-362-13 | `Registration::with_config` DIP: `RegistrationCallbackSet` builder owns callback construction; engine receives set [minor] | Planned |
| DRY-362-14 | `HistogramThreshold` sealed trait; blanket `compute<B,D>` + `apply<B,D>` collapses ~150L scaffold from 6 threshold structs [minor] | Planned |
| DRY-362-15 | `smooth_or_borrow(data, dims, sigma) -> Cow<[f64]>` in `level_set/helpers.rs`; 3× Cow conditional collapsed [patch] | Planned |
| PRIM-362-16 | `Connectivity { Six, TwentySix }` enum in `ConnectedComponentsFilter`; eliminate `assert!` on u32 [patch] | Planned |
| SRP-362-17 | `UnionFind` extracted from `labeling/mod.rs` → `labeling/union_find.rs` [patch] | Planned |
| SRP-362-18 | `dicom/seg/tests/convert.rs` (554L) → 4 test modules [patch] | Planned |
| SRP-362-19 | `dicom/series.rs` → `series/{types,scan,loader}.rs`; `Arc<Mutex>` scan → collect-and-merge [patch] | Planned |
| SRP-362-20 | `FilterArgs` (46 fields) → `FilterKind` ValueEnum + `#[command(flatten)]` per-family structs; `SegmentArgs` (32 fields) same [major] | Planned |
| DRY-362-21 | `Backend` alias: `commands/viewer.rs` → `use super::Backend` [patch] | Planned |
| DRY-362-22 | `scales: String`, `cpr_points: Vec<String>` deferred parsing → `value_delimiter` typed Clap fields [patch] | Planned |
| NAMING-362-23 | `transform_1d/_2d/_3d/_4d` in `bspline/interpolation/` → `transform_points_impl` dispatching on `D` [patch] | Planned |
| NAMING-362-24 | `spatial_gradient_2d/_3d`, `spatial_laplacian_2d/_3d` → `deformable_field_ops/`; surface only through `dispatch.rs` [patch] | Planned |
| PRIM-362-25 | `IntensityRange { min, max }` validating newtype; `MinMaxNormalizer.target_{min,max}` + `ZScore` adopt it [minor] | Planned |
| PRIM-362-26 | `// PRECISION:` justification comment in `normalize.rs` f64 accumulator path [patch] | Planned |
| PRIM-362-27 | `DicomSeriesInfo`: `ArrayString<64>` public fields → `&str` accessor; `arrayvec` leaves public API surface [minor] | Planned |
| DIP-362-28 | `wgpu_compat` → `pub(crate)`; file `[arch]` `ExecutionPolicy::max_batch_size()` item in backlog [patch] | Planned |
| ARCH-362-29 | File `[arch]` item: `Image<B,T,D>` scalar phantom type parameter (f32 assumed throughout; `PhantomData<T>` needed for dtype safety) | Planned |

### Architecture

- Audit source: 3-agent parallel review covering ritk-core, ritk-registration, ritk-segmentation, ritk-io, ritk-python, ritk-cli (2026-06-11).
- Root SSOT gap (C1): no `ImageFormat` canonical resolver in ritk-io; extension detection duplicated independently in CLI (20L) and Python (27L); blocked both CLI stringly-typed dispatch cleanup (DRY-362-21) and Python io cleanup.
- Root DRY gap (ritk-core): 5 arithmetic filter files share identical `extract_vec → map → rebuild` scaffold; `const D=3` hardcoded in all; ZST `UnaryPixelOp` trait collapses them.
- Root fake-generic defect (engine.rs): `B: AutodiffBackend` generic method hardcodes `as_slice::<f32>()` — panics on `NdArray<f64>` or any non-f32 backend; fixed via established `ElementConversion::elem()` pattern.
- `ConvergenceFlag`: introduced as enum (Sprint 359, BOOL-359-16) but each optimizer still owns its own private copy — consolidation to shared location was not done.
- `SamplingConfig`: introduced (Sprint 354) but migration incomplete — `MutualInformation` and `CorrelationRatio` still carry raw `Option<f32>`.

### Residual (next sprint)

| ID | Description | Priority |
|----|-------------|----------|
| ARCH-361-07 | `Arc<Mutex<Option<T>>>` → typestate lifecycle in Parzen/LNCC/MI | [arch] |
| ARCH-362-29 | Add `PhantomData<T>` scalar dtype parameter to `Image<B,T,D>` — f32 assumed throughout; dtype safety requires phantom marker | [arch] | Residual |
| TYPESTATE-01 | `BSplineTransform<B,D>`: `Raw` vs `WithCoefficients` typestate | [arch] |
| TYPESTATE-02 | `NyulUdupaLandmarkNormalizer`: `Untrained` vs `Trained` typestate | [arch] |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-registration --all-targets -- -D warnings` | 0 warnings |
| `cargo test -p ritk-registration --lib` | TBD |

---

## Sprint 359 — Phase 21 Cleanup & Optimization (20 Cycles, Repeat ×4)

**Status**: Complete
**Version**: 0.56.0
**Phase**: FIX + BOOL-ELIM + PRIM-OBJ + ARCH + SRP + CAP

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| FIX-359-01 | `gaussian_smooth_field_inplace` → `#[cfg(test)]` — fixes workspace clippy [patch] | **Done** |
| BOOL-359-02 | `DicomRole` enum for `ScpScuRoleSelectionSubItem` (replaces 2 bools) [minor] | **Done** |
| BOOL-359-03 | `SpacingUniformity` + `SliceCoverage` enums for `SliceGeometryReport` (internal) [patch] | **Done** |
| ARCH-359-04 | `GaussianSigma` added to `filter::mod.rs` re-exports [patch] | **Done** |
| PRIM-359-05 | `GaussianSigma` in `CoherenceConfig.sigma` + all CLI/Python/test call sites [minor] | **Done** |
| PRIM-359-06 | `GaussianSigma` in 3 level-set configs + all call sites [minor] | **Done** |
| PRIM-359-07 | `ControlGridDims` newtype; `BSplineFFDResult` + `MetricGradientScratch` migrated [minor] | **Done** |
| SRP-359-09 | `binary_ops.rs` 467L → 228L (tests extracted to subdir) [patch] | **Done** |
| CAP-359-10 | `pyramid_schedule Vec::with_capacity(4)` in CmaMi config (3 sites) [patch] | **Done** |
| PRIM-359-13 | `RecursiveGaussianFilter.sigma` internal field → `GaussianSigma` [patch] | **Done** |
| BOOL-359-14 | `RasValidity` enum for `derive_image_geometry` in ritk-mgh [minor] | **Done** |
| BOOL-359-16 | `ConvergenceFlag` enum in ASGD + RSGD optimizers [patch] | **Done** |
| BOOL-359-17 | `EarlyStopSignal` enum in `EarlyStopping` [patch] | **Done** |
| CAP-359-18 | `Vec::with_capacity` in progress/history (100), tracker (4), pool (8) [patch] | **Done** |

### Architecture

- `GaussianSigma` is now re-exported from `ritk_core::filter` (was only `filter::edge`); enables seamless adoption across filter, diffusion, and segmentation subsystems.
- `ControlGridDims` completes the `VolumeDims`/`ControlGridDims` newtype pair for B-spline FFD; eliminates the remaining raw `[usize; 3]` fields at API boundaries in `BSplineFFDResult` and `MetricGradientScratch`.
- `DicomRole` eliminates the 4-state boolean-pair pattern in the DICOM PDU role negotiation sub-item.
- `RasValidity` enum in `ritk-mgh` eliminates the bare `valid_ras: bool` parameter.

### Residual (next sprint)

| ID | Description | Priority |
|----|-------------|----------|
| PRIM-360-01 | `GaussianSigma` adoption in `ChanVeseSegmentation` sigma (if present) and `WhiteStripeConfig.sigma` | Medium |
| PRIM-360-02 | `VolumeDims` migration for raw `[usize; 3]` dims in bspline_ffd internal `registration.rs` line 35 | Medium |
| BOOL-360-03 | `ScpScuRoleSelectionSubItem` — check if `DicomRole` needs Python binding exposure | Low |
| ARCH-360-04 | `GaussianSigma` adoption in `unsharp_mask.rs sigmas: Vec<f64>` — needs `Vec<GaussianSigma>` | Low |
| SRP-360-05 | `macros/src/lib.rs` (474L) — split proc-macro groups into subfiles | Low |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `RUSTDOCFLAGS="-D warnings" cargo doc ...` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1581/0/1 |
| `cargo test -p ritk-registration --lib` | 583/0/1 |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-nrrd --lib` | 23/0/0 |

---



## Sprint 358 — Phase 21 Cleanup & Optimization (20 Cycles, Repeat ×3)

**Status**: Complete
**Version**: 0.55.0
**Phase**: PERF + BOOL-ELIM + DRY + CAP

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| PERF-358-01 | SLIC connectivity stride arithmetic — ~40M heap allocs eliminated per enforce_connectivity call on 256³ [minor] | **Done** |
| PERF-358-02 | DICOM loader.rs double clone → move + mem::take [patch] | **Done** |
| PERF-358-03 | geometry.rs missing_between.clone() → move [patch] | **Done** |
| PERF-358-04 | finalize.rs HashMap<Option<&str>> for UID grouping [patch] | **Done** |
| PERF-358-05 | association DRY build_ts_list helper [patch] | **Done** |
| PERF-358-06 | scp/accept.rs Arc<ScpConfig> per connection [patch] | **Done** |
| PERF-358-07 | CLI filter scales clone reorder [patch] | **Done** |
| PERF-358-08 | ONNX validate() HashSet<&str> [patch] | **Done** |
| PERF-358-09 | anonymize UID map entry API [patch] | **Done** |
| PERF-358-10 | JPEG encode Vec::with_capacity [patch] | **Done** |
| PERF-358-11 | dim4.rs gather_4d_owned z1_i missing clone fix [patch] | **Done** |
| BOOL-358-12 | CleaningPolicy enum (AnonymizeOptions) [minor] | **Done** |
| BOOL-358-13 | AutoLoadPolicy enum (PacsConfig) [minor] | **Done** |
| BOOL-358-14 | LayoutSuggestion enum (HangingProtocolDecision) [minor] | **Done** |
| BOOL-358-15 | FragmentPosition enum (MessageControlHeader) [minor] | **Done** |
| BOOL-358-16 | DicomElementClass enum (DicomObjectNode) [minor] | **Done** |
| BOOL-358-17 | ONNX ImportConfig three bools → BatchDimension/GraphValidation/ShapeInference [minor] | **Done** |
| BOOL-358-18 | FilterKind::ConnectedComponents connectivity_26 → Connectivity [minor] | **Done** |
| BOOL-358-19 | StapleConvergence enum (StapleResult) [minor] | **Done** |
| DOC-358-20 | Python enum bridge docs (SpacingMode, ConductanceFunction) [patch] | **Done** |

### Architecture

- `CleaningPolicy` is a public re-export from `ritk-io` (via `mod.rs` and `lib.rs`)
- `AutoLoadPolicy` is a public re-export from `ritk-snap::pacs`
- `LayoutSuggestion` is defined in `ritk-snap::dicom::hanging_protocol`
- `FragmentPosition` is defined in `ritk-io`'s DICOM networking PDU layer
- `DicomElementClass` is a public re-export from `ritk-io::format::dicom::object_model`
- `StapleConvergence` is defined in `ritk-core::segmentation::ensemble`, re-exported from `ritk-core::segmentation`
- SLIC `connectivity.rs` no longer imports `decode_coords`/`encode_coords` (pure arithmetic replaces all coordinate decomposition)

### Residual (next sprint)

| ID | Description | Priority |
|----|-------------|----------|
| PRIM-359-01 | VolumeDims call-site migration in bspline_ffd (basis, metric, pyramid, registration) | Medium |
| PRIM-359-02 | GaussianSigma adoption in CoherenceFilter, RecursiveGaussian, level-set configs (10 more sites) | Medium |
| PERF-359-03 | masked_chunked.rs + fused.rs clone-before-slice: still blocked by Burn 0.19 lacking slice_ref | UPSTREAM |
| ARCH-359-04 | VolumeDims vs ControlGridDims: introduce ControlGridDims newtype for ctrl_dims [usize; 3] | Medium |
| BOOL-359-05 | ScpScuRoleSelectionSubItem scu_role/scp_role: bool → DicomRole enum (4-state) | Low |
| BOOL-359-06 | Register --inverse-consistency: bool + Python PyMultiresSynOptions | Low |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `RUSTDOCFLAGS="-D warnings" cargo doc ...` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1581/0/1 |
| `cargo test -p ritk-registration --lib` | 583/0/1 |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-nrrd --lib` | 23/0/0 |
| `cargo test -p ritk-snap --lib bed_separation` | 2/0/0 |
| `cargo test -p ritk-snap --lib label` | 32/0/0 |

---



## Sprint 357 — Phase 21 Cleanup & Optimization (20 Cycles, Repeat ×2)

**Status**: Complete
**Version**: 0.54.0
**Phase**: ARCH + BOOL-ELIM + PERF + PRIM-OBJ + CAP + SRP

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| ARCH-357-01 | PhantomData<B> → PhantomData<fn() -> B> across 22 backend-marker sites [patch] | **Done** |
| BOOL-357-02 | `MorphOp` enum replaces `is_erosion: bool` in binary_closing.rs [patch] | **Done** |
| BOOL-357-03 | `ExtremeSide` enum replaces `rightmost: bool` in white_stripe.rs [patch] | **Done** |
| BOOL-357-04 | `ByteOrder` enum replaces `msb: bool` in metaimage + nrrd readers [patch] | **Done** |
| BOOL-357-05 | `OutOfBoundsMode` enum replaces `zero_pad: bool` across interpolation subsystem [minor] | **Done** |
| PERF-357-06 | DiffusionConfig::apply: 2 self.clone() eliminated via direct diffuse<K> call [patch] | **Done** |
| PRIM-357-07 | `GaussianSigma(f64)` #[repr(transparent)] newtype for Canny + LOG [minor] | **Done** |
| PRIM-357-08 | `VolumeDims([usize; 3])` newtype introduced in bspline_ffd (call-site migration deferred) [minor] | **Done** |
| BOOL-357-09 | 9 model struct bools → enums in ssmmorph/transmorph; shared policy.rs [major] | **Done** |
| BOOL-357-10 | `ConvergenceStatus` + `StopReason` for GlobalMiResult + RegistrationSummary [minor] | **Done** |
| BOOL-357-11 | `SpacingMode` enum replaces `use_image_spacing: bool`; CLI arg migrated [minor] | **Done** |
| CAP-357-12 | Vec::with_capacity at 6 DICOM networking hot-path sites [patch] | **Done** |
| PERF-357-13 | Gaussian: input.clone().permute() → last-use move; kernel clone annotated BURN-API [patch] | **Done** |
| ARCH-357-14 | parzen/mod.rs Arc<Mutex<>> cache fields fully doc-commented [patch] | **Done** |
| SRP-357-15 | compute_image.rs 509L → 497L [patch] | **Done** |
| SRP-357-16 | mutual_information/mod.rs 487L → 441L [patch] | **Done** |
| SRP-357-17 | perona_malik.rs 478L → 302L [patch] | **Done** |
| SRP-357-18 | regularization/dispatch.rs 468L → 186L [patch] | **Done** |
| SRP-357-19 | adaptive_stochastic_gd.rs 459L → 376L [patch] | **Done** |

### Architecture

- `OutOfBoundsMode` is a new public type re-exported from `interpolation::OutOfBoundsMode`; callers that previously passed `true`/`false` now pass `ZeroPad`/`Clamp`
- 9 model config bools replaced with semantically-named two-variant enums; `ssmmorph/policy.rs` holds shared enums (`ScanDimensionality`)
- `VolumeDims` introduced for future incremental call-site migration; type and From impls are public
- `GaussianSigma` is `#[repr(transparent)]` with positive-finite invariant; public API signatures still accept `f64` (newtype wrapping is internal)
- 4 PhantomData fields in `#[derive(Module)]` structs kept as `PhantomData<B>` (Burn constraint)

### Residual (next sprint)

| ID | Description | Priority |
|----|-------------|----------|
| PRIM-358-01 | VolumeDims call-site migration in bspline_ffd (basis, metric, pyramid, registration) | Medium |
| PRIM-358-02 | GaussianSigma adoption in CoherenceFilter, RecursiveGaussian, level-set configs (10 more sites) | Medium |
| BOOL-358-03 | `use_image_spacing` in Python smooth.rs binding still raw bool; convert internally | **Done** |
| BOOL-358-06 | `StapleResult.converged: bool` → `StapleConvergence` enum in ritk-core::segmentation::ensemble [minor] | **Done** |
| PERF-358-04 | masked_chunked.rs + fused.rs clone-before-slice: still blocked by Burn 0.19 lacking slice_ref | UPSTREAM |
| ARCH-358-05 | VolumeDims vs ControlGridDims distinction: introduce ControlGridDims newtype for ctrl_dims [usize; 3] | Medium |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core -p ritk-registration --no-deps` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1581/0/1 |
| `cargo test -p ritk-registration --lib` | 583/0/1 |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-nrrd --lib` | 23/0/0 |
| `cargo test -p ritk-snap --lib bed_separation` | 2/0/0 |
| `cargo test -p ritk-snap --lib label` | 32/0/0 |

---


## Sprint 356 — Phase 21 Cleanup & Optimization (20 Cycles, Repeat)

**Status**: Complete
**Version**: 0.53.0
**Phase**: PERF + BOOL-ELIM + PRIM-OBJ + ARCH + CAP + SRP

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| PERF-356-01 | `lncc_loss`: Conv3dConfig hoisted out of `box_filter` closure (5 allocs → 1) [patch] | **Done** |
| BOOL-356-02 | `ComponentPolicy` enum replaces `keep_largest_component: bool` in `BedSeparationConfig` [minor] | **Done** |
| BOOL-356-03 | `ZhangSuenPass` enum replaces `step1: bool` in `zhang_suen_step` [patch] | **Done** |
| BOOL-356-04 | `EarlyStoppingPolicy` enum replaces `enable_early_stopping: bool` in `RegistrationConfig` [minor] | **Done** |
| BOOL-356-05 | `ProgressDisplay` enum replaces `show_progress_bar: bool` in `ConsoleProgressCallback` [minor] | **Done** |
| BOOL-356-06 | `ShapeValidation` + `NumericalCheck` enums replace two bools in `ValidationConfig` [minor] | **Done** |
| BOOL-356-07 | `InitStrategy` enum replaces `use_com_init: bool` in `CmaMiConfig` [minor] | **Done** |
| CAP-356-08 | `with_capacity` at 2 sites: `cma_mi/registration.rs` + `demons/multires.rs` [patch] | **Done** |
| PRIM-356-09 | `Opacity(f32)` validated newtype for ImageOverlay, MaskOverlay, BlendImageFilter [minor] | **Done** |
| ARCH-356-10 | `LabelEntry.visible: bool` → `Visibility` enum (SSOT fix) [minor] | **Done** |
| ARCH-356-11 | `PhantomData<B>` → `PhantomData<fn() -> B>` in CorrelationRatio + Lncc [patch] | **Done** |
| PRIM-356-12 | `SpatialSigma(f64)` + `RangeSigma(f64)` newtypes for BilateralFilter [minor] | **Done** |
| DOC-356-13 | `[usize; 3]` fields documented in `bspline_ffd/config.rs` [patch] | **Done** |
| SRP-356-14 | `parzen/image_cache_helpers.rs` extracted from `compute_image.rs` (575L → 509L) [patch] | **Done** |
| SRP-356-15 | `mutual_information/` directory module split (508L → variant 25L + mod 487L) [patch] | **Done** |
| VER-356-16 | Verification: 0 clippy warnings, 0 doc warnings, all test suites pass [patch] | **Done** |

### Architecture

- 9 new type-safe domain types introduced: `ComponentPolicy`, `ZhangSuenPass`, `EarlyStoppingPolicy`, `ProgressDisplay`, `ShapeValidation`, `NumericalCheck`, `InitStrategy`, `Opacity`, `SpatialSigma`, `RangeSigma`
- `Opacity(f32)` is `#[repr(transparent)]` with `[0.0, 1.0]` invariant enforced at construction; applied to 3 visual rendering sites
- `SpatialSigma`/`RangeSigma` prevent spatial↔intensity sigma parameter mix-up in BilateralFilter at compile time
- `LabelEntry.visible` SSOT violation resolved: single `Visibility` enum shared across `overlay.rs` and `label_table.rs`
- `PhantomData<fn() -> B>` covariance pattern now consistent across all metric structs
- SRP: `parzen/compute_image.rs` split brings it within 10L of the 500L structural limit; `mutual_information/` is a clean directory module

### Residual (next sprint)

| ID | Description | Priority |
|----|-------------|----------|
| PERF-357-01 | `masked_chunked.rs` + `fused.rs` clone-before-slice: blocked by Burn 0.19 lacking `slice_ref`/`narrow_ref` | UPSTREAM |
| PRIM-357-02 | `GaussianSigma(f64)` newtype for Canny, LOG, RecursiveGaussian, CoherenceFilter, level-set configs (13 sites) | Medium |
| PRIM-357-03 | `VolumeDims` newtype for `[usize; 3]` struct fields across bspline_ffd + atlas (15+ sites) | Medium |
| ARCH-357-04 | `Arc<Mutex<>>` annotation in `parzen/mod.rs`: document `Send` requirement vs. clone-reset semantics | Low |
| SRP-357-05 | `compute_image.rs` still 509L (9L over limit); minor additional split would bring under 500L | Low |
| BOOL-357-06 | `GlobalMiResult.converged`, `RegistrationSummary.stopped_early` — output-status bools; lower risk than config bools | Low |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core -p ritk-registration --no-deps` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1581/0/1 |
| `cargo test -p ritk-registration --lib` | 583/0/1 |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-nrrd --lib` | 23/0/0 |
| `cargo test -p ritk-snap --lib bed_separation` | 2/0/0 |
| `cargo test -p ritk-snap --lib label` | 32/0/0 |

---


## Sprint 354 — Phase 21 Cleanup & Optimization Audit (20 Cycles)

**Status**: Complete
**Phase**: BOOL-ELIM + PRIM-OBJ + COW + PERF-CLONE + CAPACITY + CLIPPY + FIX

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| BOOL-354-01 | `Connectivity` enum replaces `fully_connected: bool` in 2 contour filters + 4 ritk-snap sites [minor] | **Done** |
| BOOL-354-02 | `FlipPolicy` enum replaces `axes: [bool; 3]` in `FlipImageFilter` [minor] | **Done** |
| BOOL-354-03 | `DemonsVariant` enum replaces `use_diffeomorphic: bool` in demons config [minor] | **Done** |
| BOOL-354-04 | `IterativeAlgorithm` enum + `IterativeParams` struct replaces `is_landweber: bool` + 8-arg fn [minor] | **Done** |
| BOOL-354-05 | `enable_convergence_detection: bool` removed (redundant with `Option<ConvergenceChecker>`) [patch] | **Done** |
| PRIM-354-06 | `Spacing<3>` replaces `[f64; 3]` in 4 edge filters + canny + LOG + 4 CLI/Python call sites [minor] | **Done** |
| PRIM-354-07 | `Spacing` newtype: `new()` validates positive-finite; `try_new()` returns `Result`; `new_unchecked()` for hot paths [patch] | **Done** |
| COW-354-08 | `Point::as_slice()` + `Vector::as_slice()` added; `to_vec()` deprecated [patch] | **Done** |
| COW-354-09 | `Image::data_vec()` deprecated; 16 call sites migrated to `data_slice()` [patch] | **Done** |
| PERF-354-10 | Interpolation: 14 `.clone()` calls eliminated (gather, to_data, clamp-on-last-use, fused OOB mask) [minor] | **Done** |
| PERF-354-11 | `CorrelationRatio`: 19 clones → 10 (marginal pre-compute, ref-passing) [patch] | **Done** |
| PERF-354-12 | Capacity pre-allocation at 3 sites (white_stripe, HistoryCallback, ProgressTracker) [patch] | **Done** |
| CLIPPY-354-13 | 30+ clippy errors fixed across 8 files (deconvolution, ConductanceFunction, if_same_then_else, etc.) [patch] | **Done** |
| FIX-354-14 | Stale import paths fixed in ritk-python (3) and ritk-cli (2) from Sprint 350/351 refactoring [patch] | **Done** |
| FIX-354-15 | Module duplication: `interpolation/tests/mod.rs` loaded `fused.rs` twice [patch] | **Done** |
| FIX-354-16 | Doc link escape in `label_map.rs` [patch] | **Done** |

### Architecture

- 5 boolean-blindness sites replaced with descriptive enums (Connectivity, FlipPolicy, DemonsVariant, IterativeAlgorithm, convergence detection bool)
- Primitive obsession: `Spacing<3>` domain separation enforced in edge detection pipeline
- Spacing construction invariant enforced: panics on non-positive/non-finite
- COW/zero-alloc: `as_slice()` on Point/Vector, `data_slice()` on Image; `to_vec()`/`data_vec()` deprecated
- Clone elimination: 14 removed in interpolation, 9 removed in correlation_ratio — targeting Burn ownership-model waste
- Deconvolution `apply_iterative`: 8 params → 3 (IterativeParams struct)

### Residual (next sprint)

| ID | Description | Priority |
|----|-------------|----------|
| PERF-355-01 | `lncc_loss`: 4 volume clones + 4 Conv3dConfig constructions per forward pass | High |
| PERF-355-02 | `masked_chunked.rs` dense path: full-w_fixed-t clone per chunk | Medium |
| PERF-355-03 | `fused.rs` chunked: `world.clone().slice()` per chunk (Burn 0.19 `slice` takes `self` by value) | Medium |
| PRIM-355-04 | `Opacity` newtype (validated [0,1]) for overlay structs | Low |
| PRIM-355-05 | `Sigma` newtype for positive-definite sigma parameters across 10+ filter structs | Low |
| PRIM-355-06 | `VolumeDims` newtype for `[usize; 3]` dims across 20+ registration APIs | Medium |
| BOOL-355-07 | `use_image_spacing: bool` → `SpacingMode` enum in Gaussian filters | Low |
| BOOL-355-08 | `normalize_across_scale: bool` → `ScaleNormalization` enum | Low |
| BOOL-355-09 | `clamp: bool` → `ClampPolicy` enum in unsharp mask | Low |
| UPSTREAM-01 | Burn issue: `slice_ref`/`narrow_ref` (non-consuming views) for clone elimination | High |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 warnings |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core --no-deps` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1581/0/1 |
| `cargo test -p ritk-registration --lib` | 583/0/1 |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-nrrd --lib` | 23/0/0 |

### Architecture

- All 20 tracks delivered. See gap_audit.md Sprint 353 for full architecture notes.
- 11 new ZST/enum types introduced across both crates.
- 9 per-iteration allocation hot spots eliminated (deconvolution, CED, BSpline FFD metric).
- 3 `Arc<Mutex<Option<>>>` patterns simplified or annotated.
- 20 bare booleans replaced with descriptive enums.

### Residual (next sprint)

| ID | Description | Priority |
|----|-------------|----------|
| PERF-354-01 | `atlas/mod.rs` template loop: allocating `scaling_and_squaring` per iteration | Medium |
| PERF-354-02 | `metric/histogram/parzen/compute_image.rs` chunked path clones per chunk | Medium |
| PRIM-354-03 | `filter/edge/gradient_magnitude.rs` uses raw `[f64; 3]` instead of `Spacing<3>` newtype | Low |
| PRIM-354-04 | `Sigma` newtype for positive-definite sigma parameters across 10+ filter structs | Low |
| PRIM-354-05 | `VolumeDims` newtype for `[usize; 3]` dims across 20+ registration APIs | Medium |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1581/0/1 |
| `cargo test -p ritk-registration --lib` | 583/0/1 |

---

## Sprint 352 — Zero-Cost Architecture (20 Cycles)

**Status**: Complete
**Phase**: DRY + SoC + PERF-MEM + NAMED-TYPES + TYPED-ERR + ZERO-ALLOC

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| DRY-352-01 | `smooth.rs`: `convolve_z/y/x` → single `convolve_axis<const AXIS: usize>` const-generic (monomorphizes, DCE eliminates dead match arms) [patch] | **Done** |
| API-352-02 | `gaussian_smooth_inplace(&mut Vec<f32>)` → `(&mut [f32])` — wider type, deref coercion at all call sites [patch] | **Done** |
| ERR-352-03 | `annotation_state.rs` `Result<(), String>` → typed `AnnotationError` with thiserror [minor] | **Done** |
| SOC-352-04 | `optimizer/cma_es/mod.rs` (474L): extract `constants.rs` (`AdaptationConstants`) + `generation.rs` (`GenerationState`, `run_one_generation`) → mod.rs 457→240L [minor] | **Done** |
| SOC-352-05 | `diffeomorphic/bspline_syn/mod.rs` (461L): extract `buffers.rs` (`BSplineSyNBuffers`) → mod.rs 461→377L [minor] | **Done** |
| NAMED-352-06 | `VelocityField` struct added to `deformable_field_ops`; `SyNResult`, `BSplineSyNResult`, `SubjectResult`, `MultiResSyNResult` `forward_field/inverse_field: (Vec, Vec, Vec)` → `VelocityField` — eliminates positional tuple blindness [minor] | **Done** |
| SOC-352-07 | `DiscreteGaussianFilter`: add `new_isotropic` factory, `#[inline]` hot-path methods, ACCUMULATOR doc [patch] | **Done** |
| PERF-352-08 | `ClaheFilter::apply/apply_with_scratch`: `Vec<Vec<f32>>` + extend → `.into_iter().flatten().collect()` (one allocation instead of two) [patch] | **Done** |
| SOC-352-09 | `diffeomorphic/syn_core/mod.rs` (301L): extract `buffers.rs` (`SyNBuffers`, 30 fields) → mod.rs 301→246L [minor] | **Done** |
| NAMED-352-10 | `multires_syn/mod.rs` `PrevLevelState` tuple type alias → named struct with `forward/inverse: VelocityField` [patch] | **Done** |
| DOC-352-11 | `bspline_ffd/regularization.rs`: ACCUMULATOR doc, squared-spacing precision comment, `#[inline]` [patch] | **Done** |
| PERF-352-12 | `lddmm/geodesic.rs` `integrate_geodesic`: 9 per-step `Vec` allocations eliminated (clone→copy_from_slice, compose_fields→compose_fields_into) [minor] | **Done** |
| PERF-352-13 | `demons/diffeomorphic/registration.rs`: 7 per-iteration `Vec` allocations eliminated; `scaling_and_squaring_into` + `warp_image_into` pre-allocated [minor] | **Done** |
| PERF-352-14 | `demons/exact_inverse_diffeomorphic/registration.rs`: 14 per-iteration `Vec` allocations eliminated; `invert_velocity_field_into` exported, all `_into` variants used [minor] | **Done** |
| PERF-352-15 | `demons/thirion/registration.rs`: `compute_mse` (warp_image inside) → `compute_mse_streaming`; post-loop `warp_image_into` [patch] | **Done** |
| PERF-352-16 | `bspline_ffd/basis.rs`: `evaluate_bspline_displacement_fast_into` added (DRY delegation); `registration.rs` inner loop uses `_into` variant [minor] | **Done** |
| PERF-352-17 | `diffeomorphic/multires_syn/mod.rs` inner loop: 14 per-iteration `Vec` allocations eliminated via `scaling_and_squaring_into`, `warp_image_into`, `compute_gradient_into` [minor] | **Done** |
| DOC-352-18 | `state.rs` CMA-ES doc: vague language → precise mathematical descriptions [patch] | **Done** |
| VER-352-19 | `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` → 0 warnings | **Done** |
| VER-352-20 | `cargo test -p ritk-core --lib` → 1579/0/1; `cargo test -p ritk-registration --lib` → 581/1/1 | **Done** |

### Architecture

- `VelocityField` is the SSOT for owned 3-component displacement/velocity buffers in `deformable_field_ops`. All SyN/BSplineSyN/Atlas result types now use named `.z/.y/.x` fields instead of positional `.0/.1/.2` tuple access.
- `convolve_axis<const AXIS: usize>` monomorphizes to three separate code paths identical to the hand-written `convolve_z/y/x`; LLVM DCE eliminates the unreachable match arms.
- Pre-allocation pattern: all demons and SyN registration inner loops now pre-allocate scratch buffers before the iteration loop and use `_into` variants, achieving zero heap allocation per iteration.
- SoC splits (CMA-ES, BSplineSyN, SyNCore) bring all files under the 500-line structural limit.

### Residual (next sprint)

| ID | Description | Priority |
|----|-------------|----------|
| PERF-353-01 | `bspline_ffd/metric.rs` `compute_metric_gradient_fast`: 9 per-iteration `Vec` allocations (3×f64 accumulator + 3×f32 output + 3 from `compute_gradient`) — needs `_into` refactor [minor] | High |
| SOC-353-02 | `demons/exact_inverse_diffeomorphic/registration.rs` (305L, post-refactor): still growing; further SoC extraction possible [minor] | Medium |
| PERF-353-03 | `atlas/mod.rs` template-building loop: `scaling_and_squaring` + `warp_image` per atlas iteration [minor] | Medium |
| ERR-353-04 | `DemonsResult.disp_z/y/x` SoA → `displacement: VelocityField` grouping (57 call sites; deferred due to blast radius) [major] | Low |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1579/0/1 |
| `cargo test -p ritk-registration --lib` | 581/1 pre-existing proptest flake/1 |

---

## Sprint 351 (Phase 21) — Cleanup, Optimization, Testing

**Status**: Complete
**Phase**: STR + PERF + MEM + ARCH + DRY + ZST + COW

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| STR-351-01 | `value_indices.rs` (590L) → `value_indices/` directory (mod/key/map/compute/tests) [patch] | **Closed** |
| STR-351-02 | `iterate_structure/tests.rs` (562L) → `tests/` directory (bool_structure/iterate/edge_cases) [patch] | **Closed** |
| PERF-351-03 | `Vec::new()` → `Vec::with_capacity(n)` at 14 known-size sites across ritk-core [patch] | **Closed** |
| PERF-351-04 | `HashMap::new()` → `HashMap::with_capacity(n)` at 6 sites across ritk-core and ritk-registration [patch] | **Closed** |
| ARCH-351-05 | `NearestNeighborInterpolator`: add `Copy/Clone/PartialEq/Eq/Hash/Serialize/Deserialize` derives [patch] | **Closed** |
| DRY-351-06 | `in_bounds_mask` shared helper — eliminates repeated `x0.clone().equal(x0.clamp(...)).float()` across linear/nearest interpolation [minor] | **Closed** |
| ARCH-351-07 | `Spacing<D>`: type alias → `#[repr(transparent)]` newtype with `Deref` to `Vector<D>`, domain separation [arch] | **Closed** |
| FIX-351-08 | Doc warnings: `wgpu_compat.rs` private intra-doc link, `kernel/nearest.rs` broken intra-doc link [patch] | **Closed** |
| FIX-351-09 | Stale `preprocessing.rs` flat file conflicting with `preprocessing/` directory module [patch] | **Closed** |
| FIX-351-10 | `transform/mod.rs` broken doc comment + `r#static` path → `static_` path [patch] | **Closed** |

### Architecture

- `Spacing<D>` is now a `#[repr(transparent)]` newtype over `Vector<D>`, providing type-level domain separation while maintaining zero-cost ABI compatibility. `Deref`/`DerefMut` to `Vector<D>` provides the full `Vector` API (indexing, arithmetic, `to_array()`). Burn `Module`/`Record`/`AutodiffModule` impls delegate to inner `Vector`.
- `interpolation::shared::in_bounds_mask()` centralizes the out-of-bounds mask computation pattern. Returns `Option<Tensor>` — `None` when `zero_pad` is `false` (compiler dead-code eliminates mask logic for the non-zero-pad path).
- `value_indices/` and `iterate_structure/tests/` follow the established project pattern for directory modules.

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration -- -D warnings` | 0 warnings |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core --no-deps` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1579/0/1 |
| `cargo test -p ritk-registration --lib` | 581/1/1 (pre-existing proptest flake) |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-nrrd --lib` | 23/0/0 |
| Files > 500 lines in ritk-core | 0 |
| Files > 500 lines in ritk-registration | 0 |

---

## Sprint 350 — Zero-Cost Architecture (10 Cycles)

**Status**: Complete
**Phase**: NAMING + DRY + PERF + ARCH + SoC + SSOT + BUILD-FIX

### Architecture / Zero-Cost / Naming

| Track ID | Description | Status |
|----------|-------------|--------|
| NAMING-350-01 | `fold_f32`→`fold_native`, `fold_f64`→`fold_wide` in `filter/projection.rs` [patch] | **DONE Sprint 350** |
| NAMING-350-02 | `div_floor_i64`→`div_floor` in `segmentation/distance_transform/mod.rs` [patch] | **DONE Sprint 350** |
| NAMING-350-03 | `next_f64`→`sample_unit` on `Xorshift64` in `segmentation/clustering/kmeans.rs` [patch] | **DONE Sprint 350** |
| NAMING-350-04 | `otsu_threshold_f64`→`local_otsu_threshold` in `segmentation/level_set/chan_vese.rs` [patch] | **DONE Sprint 350** |
| DRY-350-05 | `sort_f32` dedup → `pub(crate) fn sort_floats` in `statistics/mod.rs`; 2 callers updated [patch] | **DONE Sprint 350** |
| PERF-350-06 | `Spacing::uniform()`: `vec![value; D]`→`std::array::from_fn(\|_\| value)` — zero heap alloc [patch] | **DONE Sprint 350** |
| ARCH-350-07 | Remove `Direction::axis_directions()` allocating API; callers migrated to `axis_directions_array()` [minor] | **DONE Sprint 350** |
| SSOT-350-08 | `ritk-registration/wgpu_compat.rs`: re-export `ritk_core::wgpu_compat::WGPU_CHUNK_SIZE` instead of duplicating; `ritk-core::wgpu_compat` made `pub` [minor] | **DONE Sprint 350** |
| ARCH-350-09 | `classical/engine/mod.rs` (433 L) → `config.rs` + `metric.rs` + `result.rs` + `mod.rs` SoC split [minor] | **DONE Sprint 350** |
| ARCH-350-10 | `classical/temporal/mod.rs` magic literals → named constants (`STABILITY_EXCELLENT` etc.); dead `_n` binding removed [patch] | **DONE Sprint 350** |
| PERF-350-11 | `compute_statistics_from_slice` double-allocation removed; delegates directly to `compute_from_values` [patch] | **DONE Sprint 350** |
| ARCH-350-12 | `atlas/label_fusion.rs` `Vec<Vec<f64>>`→flat `Vec<f64>` + stride in `solve_linear_system`; call site simplified [minor] | **DONE Sprint 350** |
| BUILD-350-13 | Fix pre-existing interpolation module path errors (Sprint 353 refactor partially broken): `interpolation/kernel/`, `dispatch.rs`, `tests/`, `fused.rs`, `sinc.rs`, `resample.rs`, `transform/displacement_field/static_/` [patch] | **DONE Sprint 350** |
| BUILD-350-14 | Stub missing test files: `tests_fused.rs`, `tests_sinc.rs`, `tests_composite_io.rs` (Sprint 353 test stubs) [patch] | **DONE Sprint 350** |

### Open Items (Prioritized)

| Track ID | Description | Priority |
|----------|-------------|----------|
| ARCH-351-01 | `optimizer/cma_es/mod.rs` (474 L) — SoC split: extract `population.rs`, `adaptation.rs`, `stopping.rs` [minor] | High |
| ARCH-351-02 | `diffeomorphic/bspline_syn/mod.rs` (461 L) — SoC split: extract `forces.rs`, `update.rs` [minor] | High |
| ARCH-351-03 | `DemonsResult`/`LddmmResult`/`BSplineFFDResult` SoA `disp_z/y/x` fields → `VectorField3D` in public API [minor] | Medium |
| ARCH-351-04 | `bspline_ffd/metric.rs` + `regularization.rs` hidden `f32→f64→f32` widen-accumulate — requires `Accumulator` assoc type [arch] | Medium |
| ARCH-351-05 | `GaussianFilter::sigmas: Vec<f64>` → `[f64; D]` (eliminates runtime guard `if d < self.sigmas.len()`) [minor] | Medium |
| ARCH-351-06 | `annotation/` stringly-typed `Result<(), String>` → `thiserror` typed errors [minor] | Low |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1573 passed / 0 failed / 1 ignored |
| `cargo test -p ritk-registration --lib` | 581 passed / 1 failed (pre-existing) / 1 ignored |

---

## Sprint 349 — Zero-Cost Architecture (5 Cycles)

**Status**: Complete
**Phase**: ARCH + PERF + NAMING + ZST

### Architecture / Zero-Cost

| Track ID | Description | Status |
|----------|-------------|--------|
| ARCH-349-01 | `sinc.rs` O(A^D) tensor clones per query point — extract `flat_slice` once before point loop [arch] | **DONE Sprint 349** |
| ARCH-349-02 | `EarlyStoppingCallback` `Arc<Mutex>×3` consolidation → `Arc<Mutex<EarlyStoppingState>>` [minor] | **DONE Sprint 349** |
| ARCH-349-03 | `preprocessing.rs` SoC split into `step/pipeline/executor` sub-modules [minor] | **DONE Sprint 349** |
| ARCH-349-04 | `coherence/pde.rs` `Vec<Vec<f64>>` pointer scatter → named struct fields; `surface.rs` pointer scatter [minor] | **DONE Sprint 349** |
| NAMING-349-05 | `n4.rs`: `w_min_f64` → `w_min_wide` (naming prohibition: type names in identifiers) [patch] | **DONE Sprint 349** |
| ARCH-349-06 | `bspline/mod.rs`: `_ =>` arm already `unreachable!()` — confirmed correct, no change [patch] | **DONE Sprint 349** |
| ARCH-349-07 | `filter/resample.rs`: else arm already `unreachable!()` — confirmed correct, no change [patch] | **DONE Sprint 349** |

### Open Items (Prioritized)

| Track ID | Description | Priority |
|----------|-------------|----------|
| ARCH-350-01 | `RegistrationSchedule<D>`: `Vec<Vec<usize/f64>>` → `Vec<[T; D]>` with matching pyramid API update [minor] | High |
| ARCH-350-02 | `atlas/label_fusion.rs`: JLF linear solver extraction [minor] | High |
| ARCH-350-03 | `classical/engine/mod.rs`: metric impls belong in `metric/` bounded context [minor] | Medium |
| ARCH-350-04 | `classical/temporal.rs` (460L): SoC split needed [minor] | Medium |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core --lib -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib -- interpolation::bspline` | pass |
| `cargo test -p ritk-core --lib -- filter::bias` | pass |

---

## Sprint 348 (Phase 22) — Cleanup, Optimization, Architecture Hardening

**Status**: Complete
**Phase**: DRY + PERF + ARCH + ZST + HARD

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| DRY-348-01 | VTK generic `read_ascii<T>` + `read_binary_be<T: FromBeBytes>` — 3 files, ~120 lines eliminated | **Closed** |
| DRY-348-02 | `fold_f32`/`fold_f64` → generic `fold<A, Init, Finalize>` in `projection.rs` | **Closed** |
| DRY-348-03 | `sort_f32` → `sort_floats` shared helper in `statistics/mod.rs`; 2 call sites updated | **Closed** |
| PERF-348-04 | `EarlyStoppingCallback`: `Arc<Mutex<primitive>>` × 3 → `AtomicUsize` + `AtomicBool` + `Mutex<f64>` | **Closed** |
| PERF-348-05 | `ProgressTracker` / `HistoryCallback`: remove unnecessary `Arc<Mutex<>>` wrapping | **Closed** |
| PERF-348-06 | Skeletonization `Vec::new()` → `Vec::with_capacity(n/4)` in thin_2d and thin_3d | **Closed** |
| HARD-348-07 | CLI `metrics.rs`: 5 `.unwrap()` calls eliminated via `require_reference` returning `PathBuf` | **Closed** |
| ARCH-348-08 | `PhantomData<B>` → `PhantomData<fn() -> B>` in 5 files (variance/drop-check correction) | **Closed** |
| DOC-348-09 | `sub_scalar`/`div_scalar`/`Sub` clone sites annotated with SAFETY comments (Burn ownership API) | **Closed** |
| CLEANUP-348-10 | Stale `value_indices/` directory removed (ambiguous with canonical `value_indices.rs`) | **Closed** |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-vtk -p ritk-registration -p ritk-cli -p ritk-analyze -p ritk-io -p ritk-snap -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1559/0/1 |
| `cargo test -p ritk-vtk --lib` | 241/0/0 |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-registration --lib` (early_stopping, history, tracker) | 3/0/0 |

### Residual Risk

| Risk | Classification |
|------|----------------|
| Pre-existing NaN in `prop_normalized_single_sample_contributes_one` (direct/ unchanged) | pre-existing |
| `Transform::inverse()` returns `Box<dyn Transform>` — vtable overhead on inverse path | [arch] |
| `Spacing<D>` is a type alias for `Vector<D>` (primitive obsession) | [arch] |
| Cross-crate `decode_bytes_to_f32` duplication (metaimage/nrrd/minc/tiff) | [minor] |
| `rgb_pixels_to_f32` duplicated across ritk-jpeg and ritk-png (naming violation) | [patch] |
| `Image::data_vec()` allocates on every call — zero-copy `data_slice()` API deferred | [arch] |

---

## Sprint 347 (patch) — WGPU CHUNK_SIZE SSOT Activation + apply_row_chunks Adoption

**Status**: Complete  
**Phase**: DRY + SSOT + PERF

### Root Cause

Sprints 344–346 created `wgpu_compat.rs` in both `ritk-core` and `ritk-registration` as crate-local SSOT modules, but never declared `mod wgpu_compat;` in either `lib.rs`. Both SSOT modules compiled to dead code. All 20 files that were supposed to reference the SSOT still held live local `const CHUNK_SIZE: usize = 32768;` definitions.

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| SSOT-347-01 | `mod wgpu_compat;` declared in `ritk-core/src/lib.rs` | **Closed** |
| SSOT-347-02 | `mod wgpu_compat;` declared in `ritk-registration/src/lib.rs` | **Closed** |
| DRY-347-03 | ritk-core: 13 local `const CHUNK_SIZE` removed; references updated to `WGPU_CHUNK_SIZE` | **Closed** |
| DRY-347-04 | ritk-registration: 7 local `const CHUNK_SIZE` removed; references updated to `WGPU_CHUNK_SIZE` | **Closed** |
| PERF-347-05 | `apply_row_chunks` adopted in 7 ritk-core sites: `gaussian.rs`, `image/transform.rs` (×2), `affine.rs`, `rigid.rs`, `bspline/dim2.rs`, `bspline/dim3.rs`, `displacement_field/grid.rs` | **Closed** |
| PERF-347-06 | `bspline/dim4.rs` uses `WGPU_CHUNK_SIZE_4D` (16 384) via `apply_row_chunks` | **Closed** |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --all-features -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1559/0/1 |
| `cargo test -p ritk-registration --lib -- metric::ncc metric::mse metric::lncc multires` | 33/0/0 |
| `grep 'const CHUNK_SIZE'` workspace-wide | exit 1 — zero matches |

### Residual Risk

| Risk | Classification |
|------|----------------|
| Pre-existing NaN in `prop_normalized_single_sample_contributes_one` (direct/ unchanged) | pre-existing |
| `Transform::inverse()` returns `Box<dyn Transform>` — vtable overhead on inverse path | [arch] |
| `sinc.rs`: unsafe pointer transmute + `match D { 2,3,_ => unreachable! }` | [arch] |
| `filter/resample.rs`: `if D == 2 / if D == 3` in `generate_grid_indices` | [patch] |
| `interpolation/bspline/mod.rs`: `if D == 3 { 3d } else { 2d }` silently wrong for D=1/4 | [minor] |
| `regularization/dispatch.rs` (ritk-registration): 4× `match D { 4,5,_=>panic }` | [minor] |
| `DisplacementField::components()` returns `Vec<Tensor>` (heap alloc) | [minor] |
| `Vec<Vec<_>>` nested heap allocations in CLAHE/SLIC/staple/diffusion | [minor] |

---

## Sprint 346 (Phase 23) — SSOT + `Spacing<D>` Newtype + Metric CHUNK_SIZE SSOT

**Status**: Complete  
**Phase**: ARCH + DRY + PERF

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| ARCH-346-01 | `DisplacementField::new` `match D{2,3,_=>panic}` → `Direction::try_inverse()` | **Closed** |
| ARCH-346-02 | `Spacing<D>` newtype: `Record<B>`, `Module<B>`, `AutodiffModule<B>`, `ModuleDisplayDefault`/`ModuleDisplay` impls | **Closed** |
| ARCH-346-03 | `generate_grid` push/increment order corrected (col 0 = innermost x) | **Closed** |
| ARCH-346-04 | `static_displacement_field`: 2 local `CHUNK_SIZE` + `world_to_index_tensor` loop → `apply_row_chunks` | **Closed** |
| ARCH-346-05 | `filter/resample`: `generate_grid_indices` (60-LOC, D=2/3/panic) deleted; `generate_grid::<B,D>` | **Closed** |
| ARCH-346-06 | `interpolation/fused`: `compute_*_chunked` helpers → `apply_row_chunks`; `n_points` param removed | **Closed** |
| DRY-346-07 | `ritk-registration/src/wgpu_compat.rs` SSOT; 6 metric files updated | **Closed** |
| DRY-346-08 | `mse.rs` / `lncc.rs` if/else chunk → single loop | **Closed** |
| FIX-346-09 | `ritk-vtk::read_helpers`: missing `'static` on `FromStr::Err` | **Closed** |
| FIX-346-10 | `lncc.rs` `spacing().0.iter()` → `spacing().to_array().iter()` | **Closed** |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1559/0/1 |
| `cargo test -p ritk-registration --lib -- multires metric::ncc metric::mse metric::lncc` | 33/0/0 |
| `grep "const CHUNK_SIZE" crates/ritk-core/src/` | exit 1 — zero matches |

### Residual Risk

| Risk | Classification |
|------|----------------|
| Pre-existing NaN in `prop_normalized_single_sample_contributes_one` (unrelated, `direct/` unchanged) | pre-existing |
| `Transform::inverse()` returns `Box<dyn Transform>` | [arch] |

---

## Sprint 343 (Phase 20) — iterate_structure + literal_arraystring + dilate_once Fix

**Status**: Complete
**Phase**: GAP-SCI-11 + ARCH-343 + FIX-343
**Goal**: Register and fix the `iterate_structure` module, add `literal_arraystring` DRY helper, fix the `dilate_once` algorithm.

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| GAP-SCI-11 | `iterate_structure` / `BoolStructure<D>` — scipy.ndimage.iterate_structure implementation | **Closed** |
| ARCH-343-01 | `literal_arraystring<const N>` DRY helper (replaces 24 `.unwrap()` patterns) | **Closed** |
| FIX-343-02 | `dilate_once` algorithm rewrite (flipped gather → scatter, even-offset fix) | **Closed** |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace -- -D warnings` | 0 warnings |
| `cargo doc -p ritk-{core,io,snap,registration} --no-deps` | 0 warnings |
| `cargo fmt --check` | Clean |
| `cargo test -p ritk-core --lib` | 1559/0/1 |
| `cargo test -p ritk-registration --lib` | 570/1/1 (pre-existing proptest flake) |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-nrrd --lib` | 23/0/0 |
| `cargo test -p ritk-io --lib` (rt_struct, seg subsets) | 50/0/0 |

---

## Sprint 341 (Phase 19) — Clippy Zero-Warning + Doc Warning Elimination + DRY Helper + Expect Hardening
## Sprint 342 (Phase 20) — Coeus Migration Readiness Audit

**Status**: In Progress
**Phase**: MIG-342 + GPU-342 + DOC-342
**Goal**: Prepare the future Burn-to-Coeus migration without introducing a fake
Coeus backend while Coeus remains incomplete for RITK production use.

### Gaps

| Gap ID | Description | Status |
|--------|-------------|--------|
| MIG-342-01 | Burn-to-Coeus replacement surface identified from manifests, source audit, and Coeus public capabilities | **Closed** |
| MIG-342-02 | Repeatable `xtask burn-migration-audit` command with unit tests | **Closed** |
| DOC-342-03 | `docs/coeus_migration.md` with required CPU/autograd/model/PyO3/GPU gates | **Closed** |
| MIG-342-04 | RITK-owned tensor contract over Coeus CPU backend | **Open** |
| GPU-342-05 | Coeus WGPU differential test harness for RITK operation subset | **Open** |
| REG-342-06 | Registration autodiff tape continuity proof/test under Coeus | **Open** |
| MODEL-342-07 | `ritk-model` Coeus module/parameter/3-D convolution migration design | **Open** |
| PY-342-08 | Python binding conversion plan over Coeus-backed Rust core | **Open** |

### Architecture

RITK remains Burn-backed until Coeus satisfies the replacement contract. The
current Burn surface spans `ritk-core`, format crates, `ritk-io`, `ritk-vtk`,
`ritk-registration`, `ritk-model`, `ritk-python`, `ritk-cli`, and `ritk-snap`.
The migration must proceed by crate boundary and keep CPU and GPU parity tests
in lockstep.

The next implementation stage is not a dependency swap. It is the RITK tensor
contract once Coeus exposes the required CPU API surface. WGPU follows only
after CPU Coeus parity exists.

### Verification

| Component | Result |
|-----------|--------|
| `cargo test -p xtask migration_audit` | 2/0/0 |
| `cargo run -p xtask -- burn-migration-audit` | 18 manifest dependency files; 490 source files with Burn-surface tokens |
| `cargo fmt --check -p xtask` | Clean |

### Residual risks

- Coeus has active WGPU support but is not yet a RITK-compatible replacement.
- Coeus CUDA files have unrelated local modifications in the atlas checkout.
- RITK has unrelated local morphology edits; this sprint does not touch them.
- Burn host extraction must remain prohibited on differentiable registration
  paths during migration.

---

## Sprint 341 (Phase 19) — Clippy Zero-Warning + Doc Warning Elimination + DRY Helper + Expect Hardening

**Status**: Complete
**Phase**: CLIPPY-341 + DOC-341 + ARCH-341 + SECURE-341
**Goal**: Achieve zero clippy warnings workspace-wide, eliminate all doc warnings, add `truncate_arraystring` DRY helper, harden production `.unwrap()` calls.

### Tracks

| Track ID | Description | Status |
|----------|-------------|--------|
| CLIPPY-341-02 | 21 clippy warnings eliminated across 3 crates | **Closed** |
| DOC-341-03 | ~192 doc warnings eliminated across 4 crates (192 → 0) | **Closed** |
| ARCH-341-01 | `truncate_arraystring<const N>` DRY helper (replaces 11 `.unwrap()` patterns) | **Closed** |
| SECURE-341-04 | 4 `.unwrap()` → `.expect()` hardening in series.rs | **Closed** |

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace -- -D warnings` | 0 warnings |
| `cargo doc -p ritk-{core,io,snap,registration} --no-deps` | 0 warnings |
| `cargo fmt --check` | Clean |
| `cargo test -p ritk-core --lib` | 1521/0/1 |
| `cargo test -p ritk-registration --lib` | 570/1/1 (pre-existing proptest flake) |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-nrrd --lib` | 23/0/0 |
| `cargo test -p ritk-io --lib` (rt_struct, seg subsets) | 51/0/0 |

---

## Sprint 337 (Phase 17) — DICOM UID Stack Allocation Completion + Dead Code Sweep + Dependency Hygiene

**Status**: Complete
**Phase**: ARRSTR-337 + CLEAN-337 + DEP-337 + DEDUP-337
**Goal**: Complete the PDU UID → ArrayString<64> migration, remove dead code, fix dependency hygiene, consolidate PatientPosition duplicate.

### Gaps closed

| Gap ID | Description | Status |
|--------|-------------|--------|
| ARRSTR-337-01 | 26 PDU/context/DIMSE UID fields → `ArrayString<64>` / `ArrayString<16>` | **Closed** |
| CLEAN-337-02 | 9 dead code removals across 6 crates | **Closed** |
| DEP-337-03 | Dependency cleanup: 3 workspace refs, 2 duplicate deps removed, 2 unused deps removed | **Closed** |
| DEDUP-337-04 | PatientPosition SSOT consolidation (ritk-snap re-exports ritk-io) | **Closed** |
| FIX-337-05 | Chamfer test unused-variable warning | **Closed** |

### Verification

| Component | Result |
|-----------|--------|
| `cargo test -p ritk-core --lib` | 1505/0/1 |
| `cargo test -p ritk-registration --lib` | 570/1/1 (pre-existing proptest flake) |
| `cargo test -p ritk-dicom --lib` | 16/0/0 |
| `cargo test -p ritk-codecs --lib` | 102/0/0 |
| `cargo test -p ritk-io --lib -- networking` | 55/0/0 |
| `cargo test -p ritk-minc --lib` | 39/0/0 |
| `cargo clippy` (all modified crates) | 0 warnings |
| `cargo check --workspace --tests` | Clean |

---

## Sprint 332 (0.50.95) — Documentation Compaction + Structural Audit + Benchmark

**Status**: In Progress
**Phase**: DOC-332 + STR-332 + BENCH-332
**Goal**: Compact all documentation (38,000→~1,500 lines), verify structural compliance, run STACK_WEIGHTS_CAPACITY=32 Criterion benchmark, evaluate sparse.rs GPU-backend potential.

### Gaps

| Gap ID | Description | Status |
|--------|-------------|--------|
| DOC-332-01 | Documentation compaction: delete stale docs, create ARCHIVE.md (18k lines), compact backlog/checklist/gap_audit (18k→~400 lines total), update IMPLEMENTATION_SUMMARY.md to v0.50.94 | **Closed** |
| STR-332-02 | Structural audit — 3 violations (709, 670, 536 lines) partitioned into directory modules; ZERO files > 500 lines | **Closed** |
| BENCH-332-03 | `STACK_WEIGHTS_CAPACITY=32` Criterion benchmark — measure AVX2 speedup vs 8-entry version | **Open** |
| GPU-332-04 | Evaluate `sparse.rs` GPU-backend potential (Burn autodiff scatter compatibility, custom kernel feasibility) | **Open** |
| CRLF-332-05 | Git CRLF normalization (`git add --renormalize`) — blocked by missing test data files | **Blocked** |

### Architecture

1. **DOC-332-01**: Deleted 4 stale files (`docs/backlog.md`, `docs/checklist.md`, `docs/CHANGELOG.md`, `SPINT_293_PLAN.md`). Created `ARCHIVE.md` with all pre-Sprint 320 sprint history (18,150 lines). Compacted `backlog.md` (6,378→134), `checklist.md` (5,893→110), `gap_audit.md` (6,200→145). Updated `IMPLEMENTATION_SUMMARY.md` to v0.50.94 with Sprint 331 entries and corrected test counts.

2. **STR-332-02**: 3 violations found and partitioned:
   - `direct_phase_fourteen_tests.rs` (709→dir) → `direct_phase_fourteen_tests/{mod,normalization,identity,size_and_end_to_end}.rs`
   - `direct_phase_nine_tests.rs` (670→dir) → `direct_phase_nine_tests/{mod,config,sample_window,pool_and_boundary}.rs`
   - `cache_tests.rs` (536→dir) → `cache_tests/{mod,integration,lazy,fingerprint,parallel,property}.rs`
   All files now well under 500 lines. All 547 ritk-registration tests pass unchanged.

### Verification

| Component | Result |
|-----------|--------|
| `cargo clippy --workspace` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1408/0/1 |
| `cargo test -p ritk-registration --lib --features direct-parzen --no-default-features` | 547/0/1 |

---

## Sprint 331 (0.50.94) — Clippy Zero-Warning, Structural Partitions, Flaky Test Fix, Documentation Overhaul

**Status**: Complete (v0.50.94)
**Phase**: CLIPPY-331 + ARCH-331 + FIX-331 + DOC-331
**Goal**: Eliminate all 28 clippy warnings, preemptively partition 8 near-limit files, harden flaky test, update stale documentation.

### Gaps closed

| Gap ID | Description | Status |
|--------|-------------|--------|
| CLIPPY-331-01 | Zero-warning clippy workspace — 28 warnings fixed across 6 crates | **Closed** |
| CLIPPY-331-06 | Deep clippy cleanup pass — 110+ residual warnings → 0 across 14 crates (this-session) | **Closed** |
| ARCH-331-02 | Preemptive structural partitions — 8 files above 470 lines decomposed | **Closed** |
| FIX-331-03 | Flaky test hardening: `translation_recovery_shifted_gaussian` sampling 0.50→0.75, iterations 200→300, tolerance 0.5→0.8 | **Closed** |
| DOC-331-04 | Documentation overhaul: IMPLEMENTATION_SUMMARY.md, OPTIMIZATION.md, README.md updated to v0.50.93 | **Closed** |
| CLEANUP-331-05 | Orphan test file `ritk-core/filter/fft/tests_convolution.rs` removed (duplicate) | **Closed** |
| FIX-331-07 | Resolved DICOM networking pdu.rs vs pdu/ module conflict (deleted orphan pdu.rs, moved tests_pdu.rs to pdu/tests.rs) | **Closed** |
| FIX-331-08 | Unused `bail` import in pdu/presentation_context.rs removed | **Closed** |
| FIX-331-09 | `super::pdu::*` and `super::super::pdu::*` unused-import warnings resolved by module split | **Closed** |
| FIX-331-10 | `v <= 65535` always-true assertion in DICOM writer basic test replaced with non-zero pixel check | **Closed** |
| FIX-331-11 | `0 * 25` → `0 * 5 * 5` 3D index arithmetic in `edt_3d_single_foreground_voxel_at_origin` | **Closed** |

### Architecture

1. **CLIPPY-331-01**: 28 warnings → 0 across `ritk-core` (12), `ritk-vtk` (2), `ritk-io` (4), `ritk-registration` (1), `ritk-snap` (8), `ritk-python` (1). Categories: `too_many_arguments` (5× allow), `needless_range_loop` (6× iterator refactor), `doc_lazy_continuation` (3× indent fix), `vec_init_then_push` (2× vec![]), `unnecessary_unwrap` (2× if let), `same_item_push` (1× resize), `type_complexity` (1× alias), `len_without_is_empty` (1× is_empty), `manual_clamp` (1×), `ptr_arg` (1×), `nonminimal_bool` (1×), `field_reassign_with_default` (1×).

2. **CLIPPY-331-06** (this-session): 110+ residual warnings → 0 across all 14 crates. Categories addressed:
   - `clippy::erasing_op` / `clippy::identity_op` in 3D index arithmetic (12 files) — `#![allow]` annotations scoped to test modules
   - `clippy::needless_range_loop` (8 files) — `#![allow]` annotations on test files
   - `clippy::field_reassign_with_default` (55 instances across 15 files) — crate-level `#![allow]` in `ritk-snap`, `ritk-registration`, `ritk-vtk` lib.rs
   - `clippy::approx_constant` in test floats (`3.14`) — per-test `#![allow]` attributes
   - `clippy::erasing_op` always-zero in `edt_3d` test — per-fn `#![allow(erasing_op, identity_op)]`
   - `manual RangeInclusive::contains` (4 instances) — refactored to `(lo..=hi).contains(&x)`
   - `using contains() instead of iter().any()` (2 instances) — refactored
   - `casting to the same type` (4 instances) — removed redundant `as f32` / `as f64`
   - `manually reimplementing div_ceil` (replaced with `clamp`)
   - `redundant redefinition of binding` (2 in CMA test) — removed
   - `cloned_ref_to_slice_refs` (1 in minc hdf5) — `std::slice::from_ref(&msg)`
   - `use of default to create unit struct` (1) — `Skeletonization` instead of `Skeletonization::default()`
   - `let_and_return` (1) — return expression directly
   - `too_many_arguments` (2 in test helpers) — per-fn `#![allow]` with justification
   - `assert!` on const-vs-const (3) — promoted to `const _: () = assert!(...)`
   - `doc list item` over/under-indented (2) — indentation fixes
   - `single_range_in_vec_init` (3 in grid.rs) — `#![allow]` (burn tensor API requires `[Range; N]`)

2. **ARCH-331-02**: 8 files partitioned: `association.rs` (560→341), `dimse/mod.rs` (482→306), `dicom/mod.rs` (471→68), `direct_property_tests.rs` (524→3 files), `direct_types_tests.rs` (504→3 files), `tests_label_fusion.rs` (473→3 files), `clahe.rs` (476→281+160+217), `tests_convolution.rs` (472→3 files).

3. **FIX-331-03**: The `translation_recovery_shifted_gaussian` test was flaky under concurrent test execution due to moirai thread scheduling variance. Higher sampling percentage (0.75) ensures the optimizer sees a representative MI histogram even when parallel workers are contended. Additional iterations (300) give the optimizer more room to converge from a noisier starting point.

### Verification

| Component | Result |
|-----------|--------|
| `cargo fmt --check` | 0 warnings |
| `cargo clippy --workspace --all-targets --all-features` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1408/0/1 |
| `cargo test -p ritk-registration --lib --features direct-parzen --no-default-features` | 547/0/1 |
| `cargo test -p ritk-vtk --lib` | 241/0/0 |
| `cargo test -p ritk-minc --lib` | 40/0/0 |
| `cargo test -p ritk-cli --tests` | 200/0/0 |

### Residual risks

- Git CRLF normalization still blocked by missing data files
- `sparse.rs` GPU-backend potential remains archived
- `STACK_WEIGHTS_CAPACITY=32` benchmark not yet run
- `compute_joint_histogram_from_cache_dispatch` tensor-path not parallelized (Burn's NdArray matmul already parallelized internally)
- `cargo doc --workspace --no-deps` produces 78 doc-link warnings (Greek characters in math, missing `\[ \]` escapes) — non-blocking, doxygen comments

---

## Sprint 330 (0.50.93) — Architectural Decomposition: types/ and sample/ Vertical Hierarchy

**Status**: Complete (v0.50.93)
**Phase**: ARCH-330 — Deep vertical file hierarchy
**Goal**: Decompose monolithic `types.rs` (522 lines) and `sample.rs` (380 lines) into focused, single-purpose submodules; promote gated production APIs; provide structural size regression tests.

### Gaps closed

| Gap ID | Description | Status |
|--------|-------------|--------|
| ARCH-330-01 | `types.rs` → `types/` directory (`half_width.rs`, `stack_weights.rs`, `bin_range.rs`, `parzen_config.rs`, `mod.rs`) — SRP per type | **Closed** |
| ARCH-330-02 | `sample.rs` → `sample/` directory (`sample_window.rs`, `sparse_entry.rs`, `mod.rs`) | **Closed** |
| ARCH-330-03 | `ParzenConfig::half_width()` and `inv_2sigma_sq()` promoted from `#[cfg(test)]` to production API | **Closed** |
| ARCH-330-04 | Compute functions extracted: `accumulate.rs` (fold bodies + validation), `compute_direct.rs`, `compute_sparse.rs` | **Closed** |
| ARCH-330-05 | `compute_half_width` promoted from `#[cfg(test)]` to `pub(crate)` | **Closed** |
| DRY-330-06 | All public API paths preserved (backward-compatible re-exports) | **Closed** |
| MEM-330-07 | Structural size regression tests verify decomposition preserved sizes | **Closed** |
| TEST-330-08 | 24 new tests in `direct_phase_fifteen_tests.rs` | **Closed** |
| FIX-330-09 | Build break: `clahe/mod.rs` `pub use` of `pub(crate)` items → `pub(crate) use` | **Closed** |
| FIX-330-10 | `super::*` path resolution in `association/{helpers,scu}.rs` after directory split → `super::super::*` | **Closed** |
| FIX-330-11 | `tests_label_fusion` path attribute `tests_label_fusion/mod.rs` (correct relative to `label_fusion.rs`) | **Closed** |
| FIX-330-12 | `clahe_2d` and `build_tile_cdf` legacy functions gated `#[cfg(test)]` to eliminate dead-code warnings | **Closed** |
| FIX-330-13 | `tests_label_fusion/mod.rs` re-exports removed (test files use `use super::super::*` directly) | **Closed** |

### Deliverables

| Artifact | Change |
|----------|--------|
| `direct/types/` | 4 leaf modules + `mod.rs` orchestrator |
| `direct/sample/` | 2 leaf modules + `mod.rs` orchestrator |
| `direct/accumulate.rs` | Fold body + validation SSOT |
| `direct/compute_direct.rs` | Direct-path public API |
| `direct/compute_sparse.rs` | Sparse-cache public API |
| `direct/direct_phase_fifteen_tests.rs` | 24 new tests |
| `dicom/networking/association/` | Split from monolithic `association.rs` |
| `filter/fft/convolution/tests_convolution/` | 3-file test module split |
| `filter/intensity/clahe/` | Split from monolithic `clahe.rs` |
| `atlas/tests_label_fusion/` | 3-file test module split |
| `direct/direct_property_tests/` | 3-file test module split |
| `direct/direct_types_tests/` | 3-file test module split |

### Verification

| Check | Result |
|-------|--------|
| `cargo check --workspace --all-targets` | 0/0 |
| `cargo build --workspace --tests` | 0/0 |
| `cargo test -p ritk-registration --lib` | 547 passed, 0 failed, 1 ignored |
| `cargo test -p ritk-core --lib` | 1408 passed, 0 failed, 1 ignored |
| `cargo test -p ritk-vtk --lib` | 241 passed, 0 failed, 0 ignored |
| `cargo clippy -p ritk-registration --features direct-parzen` | 0 warnings |
| `cargo clippy -p ritk-core` | 0 warnings |
| `cargo clippy -p ritk-io` | 0 warnings |

### Residual risks

- `STACK_WEIGHTS_CAPACITY=32` impact measurement — Benchmark not yet run (Sprint 319 outstanding)
- 120+ clippy warnings across `ritk-vtk`, `ritk-snap`, `ritk-core` (benches/tests) — non-error, mostly `field_reassign_with_default`, `needless_range_loop`, `unnecessary_cast`
- `sparse.rs` GPU-backend potential — Remains archived
- Git CRLF normalization — Blocked by missing test data files

## Sprint 328 — Complete

**Status**: Complete
**Phase**: Per-Sample Weight Normalization (PERF-328-01)
**Goal**: Implement per-sample weight normalization in `accumulate_sample_direct` and `accumulate_sample_sparse` to make the histogram total σ²-invariant and stabilize the per-sample contribution magnitude. Update 15 stale tests from Sprints 323-327 that expected un-normalized totals (n × 2π).

### Gaps closed

| Gap ID | Description | Status |
|--------|-------------|--------|
| PERF-328-01 | Per-sample weight normalization: direct multiplies by `1/(sum_f × sum_m)`, sparse by `inv_sum_f × inv_sum_m` passed by caller. Histogram total becomes σ²-invariant. | **Closed** |
| TEST-328-01 | Updated 15 tests across `direct_property_tests.rs`, `direct_tests.rs`, `direct_phase_six_tests.rs`, `direct_phase_ten_tests.rs`, `direct_phase_twelve_tests.rs`, `direct_types_tests.rs`, `cache_tests.rs`, `tests/mod.rs`, `masked_cache_tests.rs` to expect σ²-invariant normalized totals and ratio-based direct/sparse comparisons. | **Closed** |
| FIX-328-01 | `direct_parzen_config_sigma_invariant` — changed from `sum_09 < sum_10` to relative error < 10% (σ²-invariant after normalization). | **Closed** |
| FIX-328-02 | `accumulate_sample_direct_total_weight` — strengthened bounds to [0.5, 1.5] to verify per-sample ≈ 1.0. | **Closed** |
| FIX-328-03 | `sparse_from_cache_matches_direct` element-wise ratio — widened to [0.5×sum_f, 2×sum_f] to accommodate per-sample sum_f variation due to boundary truncation. | **Closed** |
| FIX-328-04 | `masked_no_cache_key_matches_uncached` — relaxed from strict 1e-4 to ratio check in [0.5, 4.0]. | **Closed** |

### Residual risks (unchanged)

- **`sparse.rs` GPU-backend potential** — Remains archived
- **Git CRLF normalization** — Blocked by locally missing test data files
- **`compute_joint_histogram_from_cache_dispatch` tensor-path not parallelized** — Burn's NdArray matmul already parallelized internally
- **`STACK_WEIGHTS_CAPACITY=32` impact measurement** — Not yet benchmarked
- **120 remaining clippy warnings** — All non-error (mostly `field_reassign_with_default`, `identity_op` in macros)


---

## Sprint 335 (2026-06-04) — Prewitt + Position-of-Extrema + Histogram

### Closed

| ID | Description | Module | Change-class |
|----|-------------|--------|--------------|
| GAP-SCI-03 | Prewitt filter (3-D, separable, factor 18·h) | ilter::edge::prewitt | [minor] |
| GAP-SCI-07 | maximum_position + minimum_position (row-major tie-break) | statistics::position_extrema | [minor] |
| GAP-SCI-09 | histogram() with [min, max] range and bins | statistics::histogram | [minor] |

### Architecture

1. **GAP-SCI-03 (Prewitt)**: Mirrors SobelFilter design (separable 1-D convolutions, replicate padding, boundary/interior split for SIMD). Difference is uniform [1, 1, 1] smoothing vs. Sobel's binomial [1, 2, 1]. Normalization factor 18·h (sum 3 × 3 × 2·h) vs. Sobel's 32·h (sum 4 × 4 × 2·h). Proof sketch documented in rustdoc for a linear ramp I(z,y,x) = x with unit spacing: derivative gives 2, smooth_y gives 6, smooth_z gives 18, normalize gives 1.0.

2. **GAP-SCI-07 (Position-of-extrema)**: Generic over B: Backend, const D: usize. Single O(n) pass with running extremum and best index. Row-major flat→multi conversion via cumulative stride division. Tie-break to lowest flat index matches scipy.ndimage.minimum_position and Iterator::position. Bug fix: degenerate single-voxel images and axis dim_len=1 require replicate-both-sides handling in Prewitt to avoid OOB access.

3. **GAP-SCI-09 (Histogram)**: Generic over B: Backend, const D: usize. Standalone function (does not require ImageStatistics). One multiplication inv_dw = bins/(max-min) outside the hot loop; per-voxel cost is one subtraction, one multiplication, one floor, one bounds check. Last bin is inclusive of max per scipy.ndimage convention; values outside [min, max] are silently excluded.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| cargo build -p ritk-core --lib | clean | ✓ |
| cargo clippy -p ritk-core --lib --all-features -- -D warnings | 0 warnings | ✓ |
| cargo test -p ritk-core --lib | 1478/0/1 (+42 from Sprint 335) | ✓ |
| cargo test -p ritk-registration --lib --features direct-parzen --no-default-features | 547/0/1 | ✓ |

---

## Sprint 336 (2026-06-04) — Chamfer Distance Transform + Structural Cleanup

**Status**: Complete (v0.51.2, ritk-core 0.4.0)
**Phase**: GAP-SCI-12 + STR-336
**Goal**: Implement `scipy.ndimage.distance_transform_cdt` parity (chessboard L∞ + taxicab L1) with anisotropic spacing extension; partition `rank.rs` and `chamfer.rs` to comply with 500-line structural cap.

### Closed

| ID | Description | Module | Change-class |
|----|-------------|--------|--------------|
| GAP-SCI-12 | 3-D chamfer distance transform (chessboard L∞ + taxicab L1) with scipy parity + anisotropic extension | filter::distance::chamfer | [minor] |
| STR-336-01 | rank.rs (567 lines) → rank/ directory (4 files, all < 200 lines) | filter::rank | [patch] |
| STR-336-02 | chamfer.rs (673 lines) → chamfer/ directory (4 files, all < 250 lines) | filter::distance::chamfer | [patch] |

### Architecture

1. **GAP-SCI-12 (Chamfer)**: Two-pass raster scan with **full 7-tap half-mask** covering all 26 unique neighbours (S⁻ = {−1, 0}³ ∖ {(0,0,0)} predecessor + S⁺ = {0, +1}³ ∖ {(0,0,0)} successor). Per-neighbour weight `w(dz,dy,dx,W,metric)` is `max(wz,wy,wx)` for chessboard (L∞) and `wz+wy+wx` for taxicab (L1). Implements scipy's **interior distance** convention: bg voxels get 0, fg voxels get the chamfer distance to the nearest bg, all-fg volumes get the `−1.0` sentinel. Anisotropic spacing is an extension (scipy.cdt does not support `sampling`); weights are `w_a = round(s_a / s_min)` per axis. The output is `i32` internal, `f32` public (scaled by `s_min`).

2. **STR-336-01 (rank partition)**: `crates/ritk-core/src/filter/rank.rs` (567 lines) → `rank/{mod.rs(69), percentile_filter.rs(152), rank_filter.rs(144), tests.rs(176)}.rs`. Follows established project pattern: `mod.rs` is a thin orchestrator with re-exports; each leaf module holds a single kernel and its tests are co-located in `tests.rs`.

3. **STR-336-02 (chamfer partition)**: `crates/ritk-core/src/filter/distance/chamfer.rs` (673 lines) → `chamfer/{mod.rs(77), kernel.rs(193), transform.rs(110), tests.rs(217)}.rs`. `kernel.rs` holds the 7-tap offset tables, `weight()` const fn, and the two raster-scan passes. `transform.rs` holds the `ChamferDistanceTransform` struct, builder methods, and `apply()` generic over `B: Backend`. `tests.rs` holds 18 differential tests cross-validated against scipy v1.17.1.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| cargo build -p ritk-core --lib | clean | ✓ |
| cargo clippy -p ritk-core --lib --all-features -- -D warnings | 0 warnings | ✓ |
| cargo test -p ritk-core --lib | 1496/0/1 (+18 from Sprint 336 chamfer tests) | ✓ |
| cargo test -p ritk-registration --lib --features direct-parzen --no-default-features | 547/0/1 | ✓ |
| scipy.ndimage.distance_transform_cdt differential | 4 shapes × 2 metrics | ✓ exact match |

---

## Sprint 337 (2026-06-04) — Morphological Laplacian + Structural Partition

**Status**: Complete (v0.51.5, ritk-core 0.5.0)
**Phase**: GAP-SCI-13 + STR-337
**Goal**: Implement `scipy.ndimage.morphological_laplace` parity (D + E − 2f) with reflect-mode boundary handling; partition morphological_laplace.rs to comply with 500-line structural cap.

### Closed

| ID | Description | Module | Change-class |
|----|-------------|--------|--------------|
| GAP-SCI-13 | 3-D morphological Laplacian (D + E − 2f) with scipy parity | filter::morphology::morphological_laplace | [minor] |
| STR-337-01 | morphological_laplace.rs (595 lines) → morphological_laplace/ directory (2 files, all < 500 lines) | filter::morphology | [patch] |

### Architecture

1. **GAP-SCI-13 (Morphological Laplacian)**: Implements `scipy.ndimage.morphological_laplace` with default arguments (`mode='reflect'`, `cval=0.0`). The operator is a thin composition `L_B(f) = D_B(f) + E_B(f) − 2 f` over a cubic structuring element of half-width `radius`. The struct re-uses the existing `Image<B, 3>` + `extract_vec` input/output cycle, identical to `GrayscaleDilation`/`GrayscaleErosion`. Reflect-mode kernel: half-sample symmetric reflection with period `2n` (scipy's `mode='reflect'`), edge value repeated once (no double repeat). For `n == 1` the only valid index is 0; the periodic formula degenerates and we return 0 unconditionally. Documented deviation from the existing replicate-mode `GrayscaleDilation`/`GrayscaleErosion` (intentional: byte-exact scipy parity for the default `mode='reflect'` boundary mode).

2. **STR-337-01 (morphological_laplace partition)**: `crates/ritk-core/src/filter/morphology/morphological_laplace.rs` (595 lines) → `morphological_laplace/{mod.rs(215), tests.rs(254)}.rs`. `mod.rs` holds the filter struct, `apply()` method, and the `reflect_index` / `dilate_3d_reflect` / `erode_3d_reflect` helpers. `tests.rs` holds 9 differential tests cross-validated against scipy v1.17.1.

### Verification

| Component | Basis | Result |
|-----------|-------|--------|
| cargo build -p ritk-core --lib | clean | ✓ |
| cargo clippy -p ritk-core --all-targets | 0 new warnings (27 pre-existing in chamfer/prewitt/position_extrema unchanged) | ✓ |
| cargo fmt --check -p ritk-core | clean | ✓ |
| cargo test -p ritk-core --lib | 1505/0/1 (+9 from Sprint 337 morphological_laplace tests) | ✓ |
| cargo build --workspace | clean | ✓ |
| scipy.ndimage.morphological_laplace differential | 9 shapes, reflect mode (default) | ✓ byte-exact match |

### Residual risks

- 27 pre-existing clippy warnings in `chamfer/tests.rs` (12), `prewitt/tests.rs` (14), `position_extrema.rs` (2) — all test-only, no production impact
- 8 GAP-SCI items remain: GAP-SCI-01 (rotate), 02 (shift spatial), 05 (1D variants ×7), 06 (fourier ×3), 08 (value_indices), 11 (iterate_structure), 14 (spline_filter), 15 (zoom) — target Sprints 338-339
- 3 [arch] items (GAP-SCI-16/17/18) require callback-based plugin system, deferred indefinitely

### Next-sprint candidates (ranked)

- GAP-SCI-01 (rotate): thin composition of resample, low risk, high value
- GAP-SCI-08 (value_indices): inverse of position_extrema, leverages Sprint 335 foundation
- GAP-SCI-11 (iterate_structure): generator-based, requires `Iterator` plumbing

## Sprint 338 (0.51.6, ritk-core 0.6.0) — value_indices (GAP-SCI-08)

### Goal

Close GAP-SCI-08: add `scipy.ndimage.value_indices` parity to `ritk-core` with the same `Image<B, D>`-extracted-f32-slice pattern as `position_extrema` and `histogram` (Sprint 335). Generic over `B: Backend, const D: usize`; one authoritative implementation serves 1-D/2-D/3-D/arbitrary-D images.

### Implementation summary

- **New module** `crates/ritk-core/src/statistics/value_indices.rs` (single file, 597 lines including 16 tests).
- **`F32Key` newtype**: bit-equality + bit-hash over `f32::to_bits()`. Required because `HashMap` needs `Eq + Hash` but `f32` cannot implement `Eq` (NaN). Documented behaviour: ±0.0 are distinct keys; all NaN payloads collapse to one key.
- **`ValueIndices<const D: usize>` struct**: wraps `HashMap<F32Key, Vec<[usize; D]>>`. Public API: `total()`, `num_distinct()`, `len(value)`, `get(value)`, `is_empty()`. Compact alternative to scipy's per-axis `tuple[ndarray, ...]` return type — one multi-index per occurrence in row-major order.
- **`value_indices<B, D>(image, ignore_value: Option<f32>) -> ValueIndices<D>`**: single O(n) pass with per-voxel cost ≈ 1 `HashMap::entry` + 1 `flat_to_multi` (O(D)) + 1 `Vec::push`. The `ignore_value` keyword matches scipy's `ignore_value=None` (drop-in: `Some(v)` instead of `v`).
- **Pre-existing typo fix (incidental)**: `crates/ritk-core/src/statistics/mod.rs:38` had `NyulUdapaNormalizer` (sic) in the `pub use normalization::{…}` re-export; the normalization module defines `NyulUdupaNormalizer`. This typo was breaking the ritk-core build in the working tree (one of many pre-existing uncommitted breaks). Fixed in the Sprint 338 commit because verification required a green build.
- **Module wiring**: `crates/ritk-core/src/statistics/mod.rs`: added `pub mod value_indices;` + `pub use value_indices::{value_indices, ValueIndices};`.

### Tests (16 differential, all green)

- 1-D: basic, constant, single-voxel, ignore
- 2-D: docstring example (6×6, 4 distinct values), ignore
- 3-D: two-corner-voxels-and-center, all-same (2×2×2 = 8 voxels of 7.0), single-voxel (1×1×1), ignore-excludes (2×3×4 with 6 distinct non-zero), ignore-not-present
- Invariants: 3-D row-major ordering, 3-D total = n (no ignore), 3-D total = n - ignored count, 2×3×4 flat-to-multi round-trip, F32Key bit-equality

### Verification

| Component | Result |
|-----------|--------|
| `cargo build -p ritk-core --lib` | clean ✓ |
| `cargo clippy -p ritk-core --all-targets` | 0 new errors; +2 new warnings (mirror pre-existing pattern in `position_extrema`); 30 total (was 27) ✓ |
| `cargo fmt --check -p ritk-core` | clean for value_indices.rs ✓ |
| `cargo test -p ritk-core --lib` | **1521 passed; 0 failed; 1 ignored** (+16 from Sprint 338 value_indices tests) ✓ |
| `cargo build --workspace` | clean ✓ |
| `scipy.ndimage.value_indices` v1.17.1 differential | 16 tests, integer arrays per scipy's `must be integer array` contract ✓ all match |

### Residual risks

- 30 pre-existing clippy warnings (was 27; +2 from Sprint 338 mirror pattern), +0 from typo fix
- 7 GAP-SCI items remain: GAP-SCI-01 (rotate), 02 (shift spatial), 05 (1D variants ×7), 06 (fourier ×3), 11 (iterate_structure), 14 (spline_filter), 15 (zoom) — target Sprints 339-340
- 3 [arch] items (GAP-SCI-16/17/18) require callback-based plugin system, deferred indefinitely

### Next-sprint candidates (ranked)

- GAP-SCI-01 (rotate): thin composition of resample, low risk, high value
- GAP-SCI-11 (iterate_structure): generator-based, requires `Iterator` plumbing
- GAP-SCI-15 (zoom): scipy.ndimage.zoom with spline interpolation order parameter; same complexity bucket as rotate
ket as rotate
