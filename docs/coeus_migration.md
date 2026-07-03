# RITK Burn to Coeus Migration

## Status

RITK still uses Burn as its tensor, autodiff, and model backend. Coeus is the
target replacement, but it is not yet a drop-in dependency for RITK because the
RITK surface spans image containers, I/O construction, registration metrics,
autodiff transforms, neural models, CLI boundaries, Python bindings, and GPU
execution.

Evidence tier: manifest audit and source audit. No migration proof is claimed.

**Strategy & sequencing: see [ADR 0002](adr/0002-core-image-burn-to-coeus-migration.md).**
Key findings from the 2026-06-30 audit that ADR 0002 acts on:
- The other stated migration targets are already clean in RITK — **no `rayon`,
  `tokio`, `nalgebra`, `ndarray`, or `rustfft` direct dependencies remain.**
  Burn is the entire remaining substrate surface.
- The parallel Coeus capability built so far (registration autodiff MIG-471…482,
  `ritk_image::coeus::Image`, per-format `read_*_coeus`) is the correct
  **capability-add** phase but has **not reduced the Burn token surface** — the
  Burn paths are all still load-bearing.
- Burn removal is **bottom-blocked** on `ritk_core`'s `Image<B: Backend>` and
  `ritk-io`'s `ImageReader`/`ImageWriter` traits; leaf crates cannot drop Burn
  in isolation. Capability is added bottom-up; **removal proceeds top-down**
  from the `ritk-io` consumers (`ritk-cli`/`ritk-python`) toward the core
  `Image` last (ADR 0002 §Cutover ordering).

## Current Burn Surface

Run the repeatable audit from the RITK workspace root:

```sh
cargo run -p xtask -- burn-migration-audit
```

The audit reports:

- crate manifests that depend on `burn` or `burn-ndarray`;
- Rust source files containing Burn-surface tokens such as `burn::`,
  `Tensor<`, `TensorData`, `Shape::new`, `Autodiff`, `GradientsParams`,
  `Conv3d`, and `Param<`;
- a crate-level summary of manifest and source references;
- the Coeus capabilities required before dependency replacement.

The audit is intentionally lexical. It is a synchronization gate for planning,
not a type-level compatibility proof.

Baseline result on 2026-06-08:

- 18 manifest dependency files include `burn` or `burn-ndarray`;
- 490 Rust source files contain Burn-surface tokens;
- largest surfaces by token count: `ritk-core`, `ritk-registration`,
  `ritk-model`, `ritk-io`, and `ritk-cli`.

## Required Coeus Capabilities

The migration requires Coeus to provide these RITK-facing capabilities before
Burn can be removed:

| Capability | Required for |
| --- | --- |
| Tensor construction from shape plus `f32` slices | image readers, tests, CLI input assembly |
| Host extraction with explicit synchronization | image writers, CLI output, Python conversions |
| Rank-generic reshape, slice, transpose, broadcast, and permute | image layout transforms, registration points, format adapters |
| Elementwise arithmetic and reductions | filtering, statistics, metrics |
| Matmul and batched matrix operations | transforms, affine and registration code |
| Interpolation and sampling primitives | resampling, registration metrics, deformation fields |
| Reverse-mode autodiff | MI/MSE/NCC/CR metric optimization and transform parameters |
| Parameter/module API | `ritk-model` affine and TransMorph-style model code |
| 3-D convolution and pooling forward/backward | neural registration models |
| WGPU backend parity | GPU registration and model execution |
| Sparse/scatter or fused histogram GPU kernel | Parzen histogram direct/sparse paths |
| CPU reference backend | deterministic differential tests |
| PyO3 conversion boundary | Python remains a thin binding layer over Rust logic |

## GPU Migration Gates

Coeus WGPU work exists, including tensor transfers, elementwise operations,
matmul, convolution, pooling, activation, optimizer, and backward kernels. RITK
must not switch GPU registration to Coeus until these RITK-specific gates pass:

1. CPU Coeus backend matches current Burn NdArray results for representative
   RITK tensor operations.
2. WGPU Coeus backend matches the CPU Coeus backend for the same operations.
3. Registration metrics preserve autodiff tape connectivity; no host extraction
   is allowed on differentiable paths.
4. Parzen histogram GPU path has either a Coeus scatter-compatible
   implementation or a fused WGPU kernel with CPU differential tests.
5. RITK model 3-D convolution and pooling forward/backward tests pass on CPU and
   WGPU.
6. Python binding tests verify value-semantic equivalence through the Rust core.

## Verified Increments

Landed, evidence-backed steps of the sequence below (most recent first):

- **Native writers — remaining 5 formats (Sprint 495, MIG-495).** mgh,
  metaimage, minc, tiff, jpeg gain native writers wrapping substrate-agnostic
  serialization cores + ritk-io `ImageWriter` adapters. All 9 formats now
  read+write natively via the unified contract. Next: the cli/python cutover
  [major] that drops Burn from the format crates.

- **Native writers — nrrd + analyze (Sprint 494, MIG-494).** Extracted
  `write_*_flat` serialization cores; added `native` writers + ritk-io
  `ImageWriter` adapters. Byte-identical to the Burn writers. nrrd/analyze now
  have full native read+write parity. Remaining writers: mgh, metaimage, minc,
  tiff, jpeg.

- **Native-reader parity — nrrd + analyze (Sprint 493, MIG-493).** Extracted
  `decode_*` seams; added `native` readers + ritk-io unified-contract
  adapters so all 9 formats read natively through one generic trait.
  Prerequisite for the cli/python cutover (the next [major] that drops Burn
  from the format crates). nrrd 34/34, analyze 4/4, io 354/354.

- **Un-gating — coeus is the mainline, not a feature (Sprint 492, MIG-489
  slice 4).** The `coeus` cargo feature removed from all 14 crates; deps
  unconditional; 40 files un-gated. Default runs now cover the entire
  native/autodiff surface (2,714 tests, counts identical to the former
  feature-gated runs). MIG-489 fully closed.

- **De-branding slice 3 — zero `coeus` in ritk fn/struct names (Sprint 491,
  MIG-489).** Format-crate fns → per-crate `native` modules with plain
  names; ritk-filter wrappers consolidated (net −7 files); interpolation /
  tensor-ops / statistics / registration modules de-branded. 2,916 tests
  green across 13 crates. Remaining: the `coeus` feature name (config-only).

- **De-branding slice 2 — `ritk_image::coeus` → `ritk_image::native`
  (Sprint 490, MIG-489).** The widest-fan-out rename (30 files / 13 crates),
  one coordinated textual change, all touched suites re-run green (1,825
  tests). Remaining: format-crate `*_coeus` fn restructuring (incl.
  ritk-filter wrapper names), then the feature-name evaluation.

- **De-branding slice 1 — registration autodiff (Sprint 489, MIG-489).**
  `metric::coeus_autograd` → `metric::autodiff`; `*_coeus` fns de-suffixed;
  seams now `autodiff::{Transform, Metric}` (path-disambiguated). Fixed a
  latent external-crate shadowing hazard; `#[inline]` pass on trait impls.
  740/740 under the new names. Remaining: format-crate fns +
  `ritk_image::coeus` module (one coordinated change), feature name.

- **Correction — image-generic I/O contract, de-branded names (Sprint 488,
  ADR 0002 Amendment A1).** User review: substrate brands must not appear in
  component names, and backend variation belongs in traits. `ImageReader<I>`/
  `ImageWriter<I>` unified (Coeus trait pair deleted); adapters renamed to
  plain names in transitional `native` modules. Remaining renames tracked as
  MIG-489.

- **Cutover step 2 coverage — seven more Coeus reader implementors
  (Sprint 487).** jpeg/mgh/metaimage/minc/png/png-series/tiff trait readers,
  each proven identical to its Burn reader on the same file. The consumer-
  cutover gate is now exactly: Coeus read paths for VTK/NRRD/Analyze/DICOM
  (none exist yet) + per-format Coeus writers (only NIfTI has one).
- **Cutover step 2 — Coeus-typed `ritk-io` contract (Sprint 486).**
  `domain::coeus::{CoeusImageReader, CoeusImageWriter}` with
  `CoeusNiftiReader`/`CoeusNiftiWriter` as live first implementors, proven by
  a trait-dispatched round-trip. Additive ([minor]) behind the `coeus`
  feature; the [major] consumer cutover (`ritk-cli`/`ritk-python`) is now the
  head of the sequence, with the remaining 12 format implementors as
  parallel mechanical work.
- **Cutover prerequisite — first Coeus format writer (Sprint 485, ADR 0002
  step 1 write-side).** `write_nifti_coeus` over a serialization core shared
  with the Burn writer, byte-identical output proven by differential test;
  first production use of Sprint 484's `data_cow_on`. The Coeus-typed
  `ritk-io` contract (MIG-486) is now justified — NIfTI has both a Coeus
  reader and writer; remaining format writers are mechanical repeats.
- **Cutover prerequisite — Coeus `Image` host-extraction parity (Sprint 484,
  ADR 0002 step 1).** Grep-audited the Burn-`Image` methods the I/O consumers
  actually call and closed the one genuine gap: layout-independent host
  extraction (`data_cow_on`/`data_vec_on` + `Default`-backend conveniences),
  verified against a host-transpose oracle on a strided view. Metadata
  accessors already had parity; the index↔world call (1 site) rides with the
  MIG-485 `ritk-io` contract change.
- **Reverse-mode autodiff — gradient-descent registration driver (Sprint 482).**
  Added `driver::gradient_descent` (+ config/outcome types) composing the
  verified primitives into one runnable entry point, generic over any
  `CoeusMetric`/`CoeusTransform`. Verified end-to-end (known-translation
  recovery to <1e-8; generic over `Affine`). This makes the Coeus registration
  capability *usable as a unit*; wiring it behind the production (Burn) engine
  API — multi-resolution, stopping policy, caller migration — is the remaining
  phase.
- **Reverse-mode autodiff — `CoeusMetric` reduction seam (Sprint 481).** Added
  `traits::CoeusMetric` with `Mse`/`Ncc` implementors and generalized the
  composition to `metric::evaluate<M, Tf>` (`mse_metric` now a thin `Mse`
  wrapper). Completes the Coeus-native registration seam set
  (`CoeusTransform` + `CoeusMetric`); a third metric (MI) can now plug in as
  another reduction. Verified the seam actually dispatches (Ncc ≠ Mse value)
  and that NCC's additive-shift invariance holds end-to-end.
- **Reverse-mode autodiff — differentiable NCC loss reduction (Sprint 480).**
  Added `ncc::normalized_cross_correlation_coeus` (`−NCC`, single-pass
  algebraic moments on the autograd tape) — the second Coeus intensity-metric
  reduction, verified with closed-form correlation/anti-correlation oracles and
  finite-difference gradients. Unblocks the `CoeusMetric` reduction seam
  (MIG-478-02, now READY: `Mse` + `Ncc` implementors).
- **Reverse-mode autodiff — Coeus-native transform seam + generic metric
  (Sprint 478).** ADR 0001 + `traits::CoeusTransform` with `Translation`/
  `Affine` implementors and a generic `mse_metric` over them (composition
  SSOT); `affine_mse_coeus` delegates to it. A parallel Coeus trait family,
  `[N,3]`-canonical, entirely additive behind the `coeus` feature (burn path
  untouched). Completes the differentiable-registration path as a parallel
  capability (MIG-471…478). Not yet wired into the production (Burn)
  registration engine — that, plus a `CoeusMetric` trait (2nd metric, NCC) and
  engine-loop port, are the remaining tracked steps.
- **Reverse-mode autodiff — end-to-end affine-MSE metric (Sprint 477).**
  Added `metric::affine_mse_coeus` = `mse(sample(moving, affine(grid, R, t)),
  fixed)`, splitting the affine `[N,3]` output to the per-axis sampler via the
  differentiable `slice`+`reshape`. Verified forward vs a closed-form linear-
  field reference, all 12 (R,t) gradients vs finite differences, and a 200-step
  GD loop driving the loss to ~0. This completes the Coeus-autograd
  registration primitive set (loss, sampling, translation/affine transforms,
  composed metrics, optimizer step); nothing in the differentiable forward/
  backward path depends on Burn. The Coeus-native `Metric`/`Transform` trait
  surface (`MIG-478-01`, [arch]) is next — ADR first.
- **Reverse-mode autodiff — differentiable affine transform (Sprint 476).**
  Added `transform::affine_transform_coeus` (`coords·Rᵀ + t` via Coeus
  `matmul`), gradient to the `[3,3]` `R` and `[3]` `t`. Verified against a host
  reference under rotation+shear and 9-entry finite-difference gradients. The
  Atlas replacement for the Burn/nalgebra affine matrix path; composing it into
  an affine-MSE metric with rotation recovery is `MIG-477-01`.
- **Reverse-mode autodiff — gradient-descent optimizability (Sprint 475).**
  Added `metric::coeus_autograd::optim::sgd_step_var` and proved end-to-end
  that `translation_mse_coeus` optimizes: a 20-step descent loop from a known
  offset drives the loss monotonically to ~0 and the translation parameter to
  the true offset. The Coeus registration objective is now demonstrably usable,
  not merely differentiable. Differentiable affine transform split to
  `MIG-476-01`; the Coeus-native `Metric`/`Transform` trait ADR is now
  well-supported but deliberately awaits the affine parameter shape.
- **Reverse-mode autodiff — end-to-end MSE-over-a-translation metric
  (Sprint 474).** Composed the loss kernel, trilinear sampler, and a new
  differentiable translation (`transform::translate_axis_coeus`) into
  `metric::translation_mse_coeus` = `mse(sample(moving, translate(grid, t)),
  fixed)`, gradient reaching the translation parameters. Verified end-to-end
  (closed-form loss/gradient at a known offset, zero at alignment,
  self-consistent finite-difference), proving the tape is intact through all
  three seams. The first usable Coeus-native registration objective; a
  gradient-descent convergence demonstration + affine transform is the next
  step (`MIG-475-01`) before the trait-surface ADR.
- **Reverse-mode autodiff — differentiable trilinear sampling (Sprint 473).**
  Added `metric::coeus_autograd::sampling::sample_trilinear_coeus`, extending
  the 1-D mechanism to 3-D (8-corner `gather`, per-axis fractional-weight
  products, coordinate gradient to each of the three axis leaves). Verified
  against a host trilinear reference, separable-ramp per-axis analytical
  gradients, and per-axis finite differences. With the MSE kernel and a trivial
  differentiable translation, an end-to-end MSE-over-a-transform metric can now
  be composed (backlog `MIG-474-01`).
- **Reverse-mode autodiff — differentiable 1-D linear sampling (Sprint 472).**
  Added `metric::coeus_autograd::sampling::sample_linear_1d_coeus`. Resolved the
  gather-semantics blocker (index is a non-differentiable integer-valued float
  `Var`; gradient flows through gathered values via `scatter_add`), so the
  coordinate gradient flows through the fractional weights: `∂out/∂x =
  signal[i1] − signal[i0]`. Verified against the closed-form ramp slope, a
  `gather` value-gradient check, an edge-clamp zero-gradient case, and a
  finite-difference cross-check. This is the mechanism that makes the MSE loss
  (below) a function of transform parameters; 3-D trilinear is the mechanical
  extension (backlog `MIG-473-01`).
- **Reverse-mode autodiff — MSE loss kernel (Sprint 471).** Added
  `ritk_registration::metric::coeus_autograd::mean_squared_error_coeus`, a
  differentiable `mean((moving − fixed)²)` built entirely from Coeus autograd
  `Var` ops (`sub`/`mul`/`mean`) with no host extraction on the path
  (satisfies gate #3 in miniature). Verified against a closed-form value
  oracle, closed-form gradients w.r.t. both inputs (`±(2/N)(moving − fixed)`),
  and a central finite-difference cross-check. This establishes that Coeus
  reverse-mode autodiff produces analytically correct gradients for the
  terminal intensity-loss node every intensity metric (MSE, NCC moments)
  reduces to. Behind the `coeus` feature; deterministic `SequentialBackend`.
  Gap it does **not** yet close: differentiable *sampling* (interpolating the
  moving image at transform-dependent coordinates), which makes the loss a
  function of transform parameters — filed as the next increment (depends on
  Coeus `gather` index semantics; see backlog `MIG-472-01`).
- **Filter compute paths (Sprints 466–470).** Coeus-native trilinear
  interpolation and the Euclidean-distance-transform + binary-morphology
  family (erode/dilate/closing/opening) via `ritk_filter::coeus_support`,
  each verified bitwise-identical to its Burn counterpart. These are
  non-differentiable boundary wrappers over already substrate-agnostic cores,
  distinct from the autodiff path above.

## Development Sequence

1. Keep Burn as the production backend.
2. Maintain `burn-migration-audit` as the authoritative migration surface.
3. Introduce a RITK-owned tensor contract only after Coeus exposes all required
   CPU operations.
4. Add CPU Coeus differential tests behind an explicit feature.
5. Add WGPU Coeus differential tests after CPU parity is green.
6. Replace Burn call sites by crate boundary, starting with I/O construction and
   host extraction, then filters/statistics, then registration metrics, then
   `ritk-model`.
7. Remove Burn dependencies only after all call sites are migrated and the audit
   reports zero manifest dependencies and zero source references.

## Non-goals

- No Burn compatibility shim.
- No placeholder Coeus backend in RITK.
- No fallback branch that silently swaps Coeus failures back to Burn.
- No Python domain logic during the migration.
