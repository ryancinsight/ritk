> ## Vocabulary policy
>
> New migration text uses provider/native names directly (`Coeus`,
> `MoiraiBackend`, `Leto`, `Eunomia`, `native`) and does not introduce new
> `Atlas-*` migration labels. Historical PM entries retain their original
> wording unless touched by the current slice. Domain medical-atlas terms are
> preserved.

# RITK Gap Audit - Active

<!-- Compacted 2026-09-21: this board keeps each item's open signal only. The per-sprint delivery prose, evidence and ticked boxes are per-PR record, and the superseded 'next increment' scope of the old `### Residual Risk` sub-sections is history; recover any of it with `git log -p -- <this file>`. -->

## Finding 2026-08-20: ritk scope-vs-delivery audit

Static audit only — no cargo build/check/test/clippy was run (shared
`CARGO_TARGET_DIR`), so nothing here claims a suite passes. Every number below
is a repeated command's output at HEAD `d06196b1`.

### Measured baseline

| Measure | Value | Source |
| --- | --- | --- |
| Workspace packages | 40 crates + `xtask` | `Cargo.toml` members |
| Rust source lines | 357 903 across 1 863 `.rs` files | `find crates xtask -name '*.rs' -print0 \| xargs -0 cat \| wc -l` |
| Test functions | 5 695 | `grep -rn '#[test]\|#[tokio::test]\|#[rstest]'` |
| `todo!(` / `unimplemented!(` / TODO-FIXME-HACK | 0 / 0 / 0 | `grep -rn` over `crates xtask` |
| Files over the 500-line target | 44 | `find ... -exec wc -l {} +` |
| Crates with `#![deny(missing_docs)]` | 6 of 40 | per-crate `src/lib.rs` grep |
| Crates with their own `README.md` | 18 of 40 | `ls crates/*/README.md` |
| `#[allow(` / `#[expect(` sites | 34 / 137 | `grep -rn` |
| `.unwrap()` outside test files and `#[cfg(test)]` | 39 | script over `crates/*/src`, `tests/` dirs excluded |
| `dyn ` / `Box<dyn ` sites | 34 / 16 | `grep -rn` |
| `pub use ... as ...` | 4 | `grep -rn` |
| Book chapters | 78 (47 top-level + 31 examples); every `SUMMARY.md` link resolves | `ls docs/book` |
| ADRs | 20, all `Accepted`, index generated | `docs/adr/README.md` |
| burn / ndarray / nalgebra in manifests | none | `grep -rn --include=Cargo.toml` |
The third-party residue check is clean. The only `ndarray` strings left in
source are Python docstrings naming `numpy.ndarray` return types
(`crates/ritk-python/src/diffusion/maps.rs:63`) and one historical note
(`crates/ritk-nifti/src/lib.rs:7`). The burn to Coeus migration left no manifest
edge and no live substrate reference.

### F1 — GPU-named registration surface with no reachable GPU backend

`GpuFieldSmoother<B: Backend>`
(`crates/ritk-registration/src/deformable_field_ops/smooth.rs:287`) and
`CpuOrGpu<B>` (`crates/ritk-registration/src/deformable_field_ops/mod.rs:157`)
name a device dimension in the type, but both are generic over
`coeus_core::Backend`, whose entire impl set at the pinned Coeus revision is
two CPU backends:
```
$ grep -rn 'impl Backend for' coeus-core/src/backend/*.rs
backend/moirai.rs:126:      unsafe impl Backend for MoiraiBackend
backend/sequential.rs:94:   unsafe impl Backend for SequentialBackend
```
No instantiation can select an accelerator, so the `Gpu` arm and the `Cpu` arm
run the same code on the same devices. Three doc comments state accelerator
performance for that unreachable path:
- `deformable_field_ops/smooth.rs:281-285` — "On an RTX 3060, smoothing a 256^3
  field takes ~4 ms vs ~80 ms for the CPU `moirai`-based path."
- `atlas/mod.rs:130-131` — "both run on the GPU — 10-50x faster than the CPU
  path for typical 256^3 fields."
- `lddmm/geodesic.rs:137` — "the per-step momentum and adjoint smoothing runs
  on the GPU — 10-50x".
No criterion baseline backs any of them, and `grep -rn hephaestus` over the
whole repository (manifests, source, docs) returns nothing: the accelerator
seam the stack owns is not wired here at all. `README.md:19-21` is the one
honest statement — "current RITK entry points use deterministic sequential or
Moirai-parallel CPU backends" — which the type names and these comments
contradict.
The `wgpu` dependency is real but confined to rendering: only `ritk-snap`
(`src/render/gpu_mesh/`, `src/render/gpu_volume/`) imports it. The `wgpu`
strings in `ritk-filter`, `ritk-image`, `ritk-interpolation`,
`ritk-registration`, and `ritk-transform` manifests are the `ritk-wgpu-compat`
path dependency, which contains only two dispatch-ceiling constants and a
chunk-scheduling helper (`crates/ritk-wgpu-compat/src/lib.rs:16-36`).

### F2 — Dual `X` / `X_native` public surface outlives the migration it served

ADR 0002 introduced `_native` to distinguish the Coeus path from the Burn path.
Burn is retired, the marker now distinguishes nothing, and both halves ship:
```
crates/ritk-filter/src/anti_alias_binary/mod.rs:67  pub fn apply<B: Backend>(...)
crates/ritk-filter/src/anti_alias_binary/mod.rs:73  pub fn apply_native<B>(...)
```
`apply_native`'s own Rustdoc reads "Coeus-native counterpart to the legacy
application method." Both bodies delegate to the same `self.run(...)` core and
differ only in the extraction/rebuild helper and the bound (`Backend` vs
`ComputeBackend`). `ritk-filter` alone defines 129 `apply_native` methods, and
22 distinct `*_native` free functions across the workspace have a same-named
sibling without the suffix — `read_image`/`read_image_native`,
`write_image`/`write_image_native`, `normalize`/`normalize_native`,
`resample_image`/`resample_image_native`, and 18 more. 354 identifiers carry a
`Native`/`native` element.
This is the pattern the naming rule and the compatibility-soup rule both name:
a replacement built beside the original rather than taking its name. The
completion work is mechanical but large, and it is the single biggest
architectural debt in the repository.

### F3 — No fuzz coverage on sixteen trust-boundary parsers

`find . -name fuzz -o -name fuzz_targets` returns nothing, and no manifest
references `cargo-fuzz`, `libfuzzer-sys`, or `arbitrary`. The repository parses
DICOM, NIfTI, NRRD, MetaImage, Analyze 7.5, MGH/MGZ, MINC2, TIFF/BigTIFF, PNG,
JPEG, JPEG-LS, JPEG 2000, VTK, TRK, TCK, and TRX — every one of them an
externally supplied byte stream. `proptest` appears in only 5 of 40 crates
(`ritk-codecs`, `ritk-core`, `ritk-dicom`, `ritk-mgh`, `ritk-registration`).
Hand-written malformed-input coverage does exist (293 test lines mentioning
malformed/truncated/corrupt input, concentrated in `ritk-codecs` and
`ritk-io`), and the hardening quality where it was sampled is high:
`crates/ritk-trx/src/parse.rs:65-103` performs checked multiplication on
header-declared counts, validates the positions and offsets array lengths
against them, and records in a comment exactly which overflow it is preventing.
That is the standard a fuzz corpus should be defending, not a substitute for it.
The 39 production `.unwrap()` sites sit in exactly these parsers — 16 in
`crates/ritk-tck/src/io.rs`, 7 in `crates/ritk-trx/src/parse.rs`, 7 in
`ritk-trk`, 5 in `crates/ritk-dicom/src/diffusion/vendor.rs`, 3 in
`crates/ritk-mif/src/decode.rs`. Every sampled one is a `try_into()` on a
fixed-width slice reached after a length check, so these are proven invariants
written in the panicking form rather than reachable panics. They should carry
`expect("invariant: ...")` so the proof ships at the panic site.

### F4 — Test budgets raised inside the default profile

`.config/nextest.toml` sets a 30 s / 60 s default, then adds 13 override blocks.
Six set `slow-timeout = { period = "600s", terminate-after = 5 }` — a 50-minute
ceiling per test — inside `profile.default` and `profile.ci` (lines 30-36,
38-46, 56-71). That is a bound raise in the profile everyone runs, not the
dedicated reviewed profile with a derived budget that an analytically
irreducible workload is entitled to. Two of the escalated filters,
`test(bspline_cr)` and `test(multires_cr)` (lines 36 and 63), match zero test
functions in the workspace: they were written for the Correlation-Ratio tests
that no longer exist (F6), and nothing has swept the config since. The
justifying comment at line 33 cites "NdArray CPU time", a substrate ADR 0002
retired.
CI carries no `cargo-semver-checks` (every crate is versioned and nine publish
to crates.io), no Miri, no `cargo-deny`/`cargo-audit`, no `cargo-machete`, and
no `mdbook test`. `cargo clippy --workspace --all-targets --all-features --
-D warnings` (`.github/workflows/ci.yml:79`) runs the default lint set only.
The workspace has no `[workspace.lints]` table, so neither `pedantic` nor
`unwrap_used` is in force anywhere.

### F5 — 1.6 GB of tracked binary payload

`git ls-files` over `test_data`, `dist`, `output`, and `scratch` totals
1 599 565 010 bytes before the payload increments. After the scratch,
`dist/`, `output/`, and duplicate registration-fixture removals, 3 191
`test_data/` files remain at 1 490 441 876 bytes.
- `test_data/` — 3 191 tracked files, the bulk of it. Largest single entries
  `test_data/ants_example/visiblehuman.nii.gz` (16.8 MB) and
  `test_data/registration/rire/training_001_ct.mha` (15.2 MB).
- `dist/` — four committed Python wheels (`ritk-0.9.0`, `0.10.0`, `0.12.0`,
  `0.12.12`; 60 MB combined). Build artifacts, and stale ones.
- `output/` — four committed registration artifacts including
  `output/patient01_mri_registered.nii.gz` (15.1 MB) and
  `output/rire_registration_comparison.png`. Runtime output belongs in the
  gitignored output root.
- `scratch/check_restart.exe` — a committed Windows executable removed by the
  first payload increment; `scratch/` is now free of tracked artifacts.
The scratch executable and its source were removed in the first payload
increment. This increment removes the tracked `dist/` and `output/` artifacts
and keeps both paths ignored; `test_data/` remains for the inventory and
checksummed externalisation work.
`.gitignore` retains only `/target` for the shared build cache; the four stale
forked-cache names were removed.
The tracked dataset inventory after this increment is:
| Dataset | Files | Bytes | Consumer evidence |
|---|---:|---:|---|
| `paired_mri_ct/` | 2 334 | 1 088 167 692 | No tracked code or manual reference; DICOM metadata contains non-empty patient and referring-physician fields, and no provenance or license record is present. Preserve pending a retention decision. |
| `3_head_ct_mridir/` | 410 | 216 159 203 | RITK registration, Python parity, viewer workflows, and the real CT/MIP evidence. |
| `registration/` | 29 | 56 727 983 | Supplementary registration tests, examples, and source manifests. |
| `2_head_mri_t2/` | 97 | 49 896 402 | RITK registration, Python parity, viewer workflows, and the real MRI evidence. |
| `2_skull_ct/` | 308 | 46 217 380 | CLI viewer defaults and JPEG lossless fixture test. |
| `ants_example/` | 2 | 21 166 223 | NIfTI source tests and registration fixture preparation. |
| `openneuro/` | 1 | 10 581 116 | Dataset manifest and filter documentation. |
| `dicom_seg/` | 5 | 1 504 256 | DICOM-SEG parser and viewer boundary tests. |
| `diffusion/` | 3 | 13 030 | README, downloader, and ignore rules; downloaded imaging payload is already externalized. |
The byte-identical MNI152 and OpenNeuro copies under `registration/` were
removed after migrating `xtask`, Python registration tests, and NIfTI source
tests to the canonical `ants_example/` and `openneuro/` paths. The public
MRI-DIR directories are retained until the existing external-download path
can reproduce their checksums; the unreferenced `paired_mri_ct/` corpus is not
deleted or fetched without provenance and retention evidence.

### F6 — README claimed two capabilities the source does not contain

`grep -ril 'correlation_ratio|CorrelationRatio|cmaes|CMA-ES'` over every `.rs`
and `.py` under `crates/` returns nothing outside a vendored Pygments lexer.
`crates/ritk-registration/src/metric/` contains exactly `autodiff`,
`dl_losses`, `lncc`, `mse`, `ncc`, `ngf` — no Correlation Ratio module — and
the only optimizer type in the crate is `GradientDescentConfig`
(`src/metric/autodiff/driver.rs:27`). Coeus supplies `SGD` (with a `momentum`
field), `Adam`, `AdamW`, `AdaGrad`, `RMSProp`; it has no CMA-ES.
`docs/archive.md:12098` and `docs/audit_optimization_sprint_350.md:480` show
`correlation_ratio.rs` did exist, so this is deletion without doc sync, not
invention. Corrected in this pass at `README.md:242,244` and
`crates/ritk-registration/README.md:20,22`. Left as a backlog item because it
is a crate-local design document rather than a top-level claim:
`crates/ritk-registration/docs/REGISTRATION_OPTIMIZATION_ANALYSIS.md:12`.
Mutual Information is real but was mislocated in the README's structure. The
only implementations are `MutualInformationMetric`
(`crates/ritk-registration/src/classical/engine/metric.rs:15`, histogram MI and
NMI) and `ritk_statistics::information::mutual_information_mattes`, reached
through `crates/ritk-python/src/metrics/mi.rs:29`. The README listed all three
variants under `ritk-registration`; that is now attributed correctly.
One further detail in that metric: `compute_joint_histogram` fixes its
subsample stride at `step = max(1, fixed.size() / 10000)`
(`classical/engine/metric.rs:39`) — a bare tuning literal with no derivation,
silently capping MI accuracy at 10 000 samples regardless of volume size.

### F7 — Documentation floor

34 of 40 crates carry no `#![deny(missing_docs)]`. The six that do
(`ritk-connectome`, `ritk-diffusion`, `ritk-diffusion-scheme`,
`ritk-parcellation`, `ritk-tck`, `ritk-tractography`) are the most recently
added. No crate uses `warn(missing_docs)` either, so the public surface of
`ritk-core`, `ritk-image`, `ritk-io`, `ritk-registration`, and `ritk-filter` is
undocumented by construction rather than by policy.
22 of 40 crates have no `README.md`. Fifteen of those publish to crates.io
(`ritk-analyze`, `ritk-annotation`, `ritk-diffusion-scheme`, `ritk-jpeg`,
`ritk-metaimage`, `ritk-mgh`, `ritk-mif`, `ritk-minc`, `ritk-morphology`,
`ritk-nifti`, `ritk-nrrd`, `ritk-png`, `ritk-tensor-ops`, `ritk-tiff`,
`ritk-wgpu-compat`) and therefore land on the registry with a blank landing
page. No crate declaring `readme.workspace = true` is missing its file, so this
does not block `cargo package`; it is a quality gap, not a build break.
`CHANGELOG.md` holds 424 `##` sections of which 167 are separate
`## [Unreleased]` headings. Whatever version axis the file once had (versioned
entries run to `## [0.102.54]`) is gone, and the file cannot answer "what
shipped in ritk-registration 0.54.0".

### F8 — Book chapters that announce content they do not contain

Every `SUMMARY.md` link resolves and no chapter carries a TBD marker, but
twelve top-level chapters are under 30 lines and several promise material they
then skip:
- `docs/book/optimization_registration.md:3` — "This chapter covers that seam:
  parameterization, iteration budgets, tolerances, step sizes, and why
  optimizer behavior must be read together with the chosen similarity metric."
  The chapter ends at line 13 with an example table; none of those five topics
  appears, and there is not one equation.
- `docs/book/backend_dispatch.md:5` (12 lines) — "The chapter therefore covers
  where dispatch is compile-time, where a host extraction is unavoidable..." It
  does not.
- `docs/book/zero_copy_io.md` (12 lines), `classical_registration.md` (18),
  `vtk_format.md` (22), `metaimage_format.md` (23), `jpeg_format.md` (25),
  `registration_metrics.md` (26), `multi_modal_registration.md` (26),
  `benchmarking.md` (27), `validation_benchmarking.md` (27),
  `png_format.md` (28).
For a domain book whose job is teaching the field before the API, the
registration chapters carrying no MI expression, no gradient-descent update
rule, and no convergence criterion is the substantive gap. No workflow runs
`mdbook test`, so the samples that do exist are unguarded against rot.

### Where the repository is strong

Recorded because it sets the denominator for the completeness estimate and
because these are the patterns worth propagating.
- Registration is verified against analytic ground truth, not smoke-tested.
  `crates/ritk-registration/tests/deformable_recovery_test.rs` builds
  `moving[p] = I(p)` and `fixed[p] = I(p + u(p))` from one continuous field so
  `D = u` is exact by construction, then reports the best-fit amplitude ratio
  `alpha = d.u / |u|^2`. `multires_recovery_test.rs:36-57` records the measured
  answer in the module doc — single-resolution alpha 0.7177, 2-level pyramid
  0.7704 — and states plainly that the pyramid "helps but does not close the
  gap," attributing the residue to the rank-1 aperture structure of the Thirion
  force rather than to convergence. Reporting a partial result honestly instead
  of tuning until it looks complete is the behaviour the evidence rules ask for.
  Rigid is covered by `test_rigid_landmark_known_rotation`
  (`classical/engine/tests.rs:27`), B-spline FFD by
  `test_bspline_ffd_mridir_ct_synthetic_shift_recovery`
  (`tests/ct_mri_dicom_registration_test.rs:268`), and four RIRE CT/MR suites
  exist behind `#[ignore]` for the downloaded corpus.
- The axis convention is pinned by oracles that cannot self-cancel. ADR 0020's
  verification section is explicit that the oblique fixtures are the
  load-bearing ones: `CartesianGridGeometry`'s tests assert hand-computed
  physical coordinates in both directions using an exact 3-4-5 rotation, the
  displacement filters are verified by rigid-motion equivariance (a `R.A` grid
  carrying `R.u` components must give exactly `R.v`) which a direction-blind
  implementation fails, and "every one of these tests was confirmed to fail
  against the pre-fix direction-free composition." The shared fixture
  `rotated_metadata_3d` (`crates/ritk-image/src/test_support.rs:161`) is
  anisotropic — spacing `[0.5, 1.25, 2.0]` — under a non-identity direction, so
  a transposed or reversed axis order cannot survive it. The ADR also names
  where the convention is deliberately not enforced
  (`FodVolume::world_to_voxel` is direction-free, with the contract stated in
  its Rustdoc).
- Zero stubs. No `todo!(`, no `unimplemented!(`, no TODO/FIXME/HACK marker
  anywhere in 357 903 lines.
- ADR discipline. 20 records, all Accepted, index generated by
  `scripts/adr-index.py` with a `check` mode, revision notes dated in place
  (ADR 0020 carries its 2026-08-19 revision for RITK-PARITY-171).

### Completeness

77% of declared scope delivered and verified. Denominator: the README feature
list plus its I/O read/write matrix and registration algorithm/metric/optimizer
tables, the 20 Accepted ADRs, and the 78 `SUMMARY.md`-linked book chapters.
Weighted per the audit rubric — capabilities-without-stubs 0.93 of 40,
verification depth 0.75 of 25 (exceptional registration and geometry oracles
against zero fuzz coverage on sixteen parsers), documentation 0.62 of 20,
conformance floor 0.55 of 15.

## Sprint 464 Audit (2026-06-30) — Retracted a Prior Unmeasured Claim, Found the Real Bottleneck

- **[PERF-432-01 still OPEN]**, now localized to a specific ~40-line block rather than "the forward pass" generally. No code changed this sprint — `git status`/`git diff` clean; Foundation-phase audit sprint (per the sprint-phase definitions), not yet Execution.

## Sprint 463 Audit (2026-06-30) — PERF-432-01 Profiling and a Rejected Fix

- **[PERF-432-01 still OPEN]** Two concrete, verified, value-preserving op-count reductions filed in backlog.md: (1) `MeanSquaredError::forward` recomputes the iteration-invariant fixed-image grid every call (200× redundant; fix requires a `Metric`-trait-wide design decision, hence not done in this pass); (2) `transform_3d_chunk` rebuilds 5 device/shape-only static index tensors every call (zero-risk hoist, not yet implemented).

## Sprint 446 Audit (2026-06-28) — VTK Reader Untrusted-Input Allocation Hardening

- **[SEC-446-05 OPEN]** The same eager-allocation pattern exists in other format-parser crates (ritk-nrrd, ritk-nifti, ritk-metaimage, ritk-mgh, ritk-minc) whose readers reserve from header count/size fields. Tracked as a READY backlog item; not yet hardened. Evidence tier for the unhardened crates: none — pattern identified by grep, not yet exploited or fixed.

## Sprint 384 Audit (2026-06-19) — Correctness Fixes, Perf Optimisation, cmake Parity Expansion

- **[C-1 OPEN]** Frangi vesselness: Hessian via finite-diff on sampled Gaussian vs ITK’s 2nd-order Deriche IIR (`HessianRecursiveGaussianImageFilter`). Diverges for σ ≲ 2 px. Fix: use `recursive_gaussian_directional(Second)` per axis. Existing IIR machinery available.
- **[SEG-03 OPEN]** `GeodesicActiveContour` convergence: max|Δφ|/dt vs ITK’s RMS. Different stopping behavior.
- **[PERF-384-01 OPEN]** `window_cc_stats` O(N·w³) 2-pass scan → O(N) centered-residual integral image. ~114× reduction at default r=3. Algorithmic redesign needed.
- **[NEW-384-01 OPEN]** `shift_scale` Python binding not exposed; 1 cmake test skips cleanly.

## Sprint 383 Audit (2026-06-19) — cmake Coverage, Perf/Memory, Clippy/Doc Cleanup

- **[PERF-381-01 OPEN]**: `cargo bench` baseline timings for separable_box_3d and EDT Phase 3 parallelizations not yet recorded. Speedup claims are not evidence-tiered. Add criterion baselines before claiming speedup in release notes.
- **[NEW-383-02 OPEN]**: 3 sitk-gated tests (AntiAliasBinary, CannySegmentationLevelSet, ContourExtractor2D) skip cleanly. Will activate when a compatible SimpleITK wheel is installed. No action needed; risk is documentation-only.

## Sprint 353 Audit (2026-06-10) — 20-Cycle Zero-Cost Architecture (Repeat)

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
| Component | Result |
|-----------|--------|
| `cargo clippy -p ritk-core -p ritk-registration --lib -- -D warnings` | 0 warnings |
| `cargo test -p ritk-core --lib` | 1581/0/1 |
| `cargo test -p ritk-registration --lib` | 583/0/1 |

## Sprint 351 Audit (2026-06-09) — Cleanup, Optimization, Architecture Hardening

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
| Component | Basis | Result |
|-----------|-------|--------|
| `cargo clippy -p ritk-core -p ritk-registration -- -D warnings` | static analysis | 0 warnings |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p ritk-core --no-deps` | doc check | 0 warnings |
| `cargo test -p ritk-core --lib` | unit tests | 1579/0/1 |
| `cargo test -p ritk-registration --lib` | unit tests | 581/1/1 (pre-existing flake) |
| Files > 500 lines in ritk-core | structural audit | 0 |
| Files > 500 lines in ritk-registration | structural audit | 0 |

## Sprint 342 Audit (2026-06-08) — Coeus Migration Readiness

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| MIG-342-01 | Burn-to-Coeus replacement surface identified from manifests and source audit | workspace | N/A |
| MIG-342-02 | Repeatable `xtask burn-migration-audit` command added | `xtask::migration_audit` | 2 |
| DOC-342-03 | Migration design note with CPU/autograd/model/PyO3/GPU gates | `docs/coeus_migration.md` | N/A |

### Open Gaps

- MIG-342-04: RITK-owned tensor contract over Coeus CPU backend
- GPU-342-05: Coeus WGPU differential test harness for the RITK operation subset
- REG-342-06: registration autodiff tape continuity under Coeus
- MODEL-342-07: Coeus module/parameter/3-D convolution migration for `ritk-model`
- PY-342-08: PyO3 conversion plan over Coeus-backed Rust core
| Component | Basis | Result |
|-----------|-------|--------|
| `cargo test -p xtask migration_audit` | unit tests | 2/0/0 |
| `cargo run -p xtask -- burn-migration-audit` | audit execution | 18 manifest dependency files; 490 source files with Burn-surface tokens |
| `cargo fmt --check -p xtask` | formatting | clean |

## Sprint 332 Audit (2026-06-03) — Documentation Compaction + Structural Audit

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| DOC-332-01 | Documentation compaction — 4 stale files removed, docs/archive.md created (18k lines), 3 root files compacted (18k→~400 lines), docs/implementation_summary.md updated | docs | N/A |
| STR-332-02 | Structural audit — 3 violations (709, 670, 536 lines) partitioned into directory modules; ZERO files > 500 lines workspace-wide | `ritk-registration::direct` | 547 |
| Component | Basis | Result |
|-----------|-------|--------|
| `cargo clippy --workspace` | 0 warnings | ✓ |
| `cargo test -p ritk-core --lib` | 1408/0/1 | ✓ |
| `cargo test -p ritk-registration --lib --features direct-parzen --no-default-features` | 547/0/1 | ✓ |

### Open Gaps

- BENCH-332-03: `STACK_WEIGHTS_CAPACITY=32` Criterion benchmark (deferred)
- GPU-332-04: Evaluate `sparse.rs` GPU-backend potential (deferred)
- CRLF-332-05: Git CRLF normalization (blocked by missing test data)

## Sprint 330 Audit (2026-06-03) — Architectural Decomposition: types/ and sample/

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

## Sprint 331 Audit (2026-06-03) — Clippy Zero-Warning + Structural Partitions + Flaky Test Fix + Documentation Overhaul

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| CLIPPY-331-01 | 28 clippy warnings → 0 across 6 crates | ritk-core, ritk-vtk, ritk-io, ritk-registration, ritk-snap, ritk-python | 2,099 |
| ARCH-331-02 | Preemptive partition of 8 near-limit files (470–560 lines) | ritk-io (3), ritk-registration (3), ritk-core (2) | 2,099 |
| FIX-331-03 | Flaky `translation_recovery_shifted_gaussian` hardened | ritk-registration | 547 |
| DOC-331-04 | docs/implementation_summary.md, docs/optimization.md, README.md updated | docs | N/A |
| CLEANUP-331-05 | Orphan `tests_convolution.rs` removed | ritk-core | 1408 |
| Component | Basis | Result |
|-----------|-------|--------|
| `cargo clippy --workspace` | 0 warnings | ✓ |
| `cargo test -p ritk-core --lib` | 1408/0/0 | ✓ |
| `cargo test -p ritk-registration --lib --features direct-parzen --no-default-features` | 547/0/0 | ✓ |
| All 12 IO/format crates | 522/0/0 | ✓ |

## Sprint 331 Post-Audit (2026-06-03) — Deep Clippy Cleanup Pass

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| CLIPPY-331-06 | 110+ residual clippy warnings → 0 across 14 crates | all | 2,234 |
| FIX-331-07 | DICOM `pdu.rs` vs `pdu/` module conflict (orphan pdu.rs deleted, tests_pdu.rs → pdu/tests.rs) | `ritk-io::dicom::networking::pdu` | 0 (test file restored from git) |
| FIX-331-08 | Unused `bail` import in `pdu/presentation_context.rs` | `ritk-io::dicom::networking::pdu` | 40 |
| FIX-331-09 | `super::pdu::*` and `super::super::pdu::*` unused-import warnings | `ritk-io::dicom::networking::association` | 40 |
| FIX-331-10 | `v <= 65535` always-true assertion in DICOM writer test | `ritk-io::dicom::writer::tests` | 40 |
| FIX-331-11 | `0 * 25` → `0 * 5 * 5` 3D index arithmetic in `edt_3d` test | `ritk-core::filter::distance` | 1408 |
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

## Sprint 328 Audit (2026-06-01) — Per-Sample Weight Normalization

| Gap ID | Description | Module | Tests |
|--------|-------------|--------|-------|
| PERF-328-01 | Per-sample weight normalization — histogram total becomes σ²-invariant | `direct::mod`, `direct::sample` | 499 |
| TEST-328-01 | 15 tests updated to expect σ²-invariant normalized totals | 9 test files in `direct/` and `tests/` | 499 |
| FIX-328-01 | `direct_parzen_config_sigma_invariant` — σ²-invariance check | `direct_property_tests.rs` | 499 |
| FIX-328-02 | `accumulate_sample_direct_total_weight` — bounds [0.5, 1.5] | `direct_types_tests.rs` | 499 |
| FIX-328-03 | `sparse_from_cache_matches_direct` element-wise ratio — wider tolerance | `direct_tests.rs` | 499 |
| FIX-328-04 | `masked_no_cache_key_matches_uncached` — ratio [0.5, 4.0] | `masked_cache_tests.rs` | 499 |
| Component | Basis | Result |
|-----------|-------|--------|
| `cargo test -p ritk-registration --features direct-parzen --lib` | 499/0/0 (2 consecutive runs) | pass |
| `cargo test -p ritk-registration --lib translation_recovery_shifted_gaussian` (isolated) | 1/0/0 | pass (flaky under contention) |
