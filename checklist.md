<!-- Compacted 2026-09-21: this board keeps each item's open signal only. The per-sprint delivery prose, evidence and ticked boxes are per-PR record, and the superseded 'next increment' scope of the old `### Residual Risk` sub-sections is history; recover any of it with `git log -p -- <this file>`. -->

## ATLAS-RITK-DICOM-ORIENTATION-070 — Provider tag SSOT [minor] — local closure 2026-08-14

- [ ] Merge the provider head, then let Helios consume the named constant and complete the cross-repo exact-head integration gate.

## AEQUITAS-AEQ-MET-68 — Eunomia 0.8 provider compatibility [patch] — local closure; external release gates open

- [ ] Run hosted security/indexing gates at the exact delivery head and obtain owner review before merge; registry indexing and package archive checks are external to this local closure.

## RELEASE-689-01 — Publish the Rust library closure

- [ ] Verify metadata, formatting, lint, focused native tests, documentation, and each packaged source archive. Exact local evidence: locked metadata reports 39 workspace packages and the manifests contain 29 explicit `publish = true` packages, alongside nine explicit nonpublishable packages and zero packages with implicit publish policy. Rustfmt and workspace dependency alignment pass; warning-denied all-target Clippy for `ritk-filter` passes; Nextest passes 1,123/1,123 `ritk-filter` tests in 8.183 seconds and 484/484 `ritk-segmentation` tests in 6.643 seconds; and focused doctests pass. Non-publishing archive preparation is blocked before `.crate` creation: `ritk-filter` requires `gaia ^0.3.0`, while crates.io currently indexes Gaia only through 0.2.1. The remaining archive set is therefore not claimed verified.
- [ ] Merge hosted gates and publish every package in dependency order.
- [ ] Verify crates.io indexing, trusted-publishing-only enforcement, and a matching GitHub Release for every published package version.

## SAFE-687-01 — Reject truncated JPEG 2000 marker tails

- [ ] Complete independent review and exact-head hosted gates.
- [ ] Reconcile PM evidence, commit, publish, and merge after hosted gates pass.

## CI-664-01 — Atlas-owned provider checkout

- [ ] Merge PR #45 after the final documentation head repeats required hosted checks.

## MIG-660-01 — Remove stale Burn contract text from native owner crates

- [ ] Commit only the claimed files. Blocker: a peer is actively editing and has staged the broader migration on the shared `main` tree. Reopen when that increment commits or moves to its migration branch; do not include peer-owned staged files in this patch.

## MIG-657-01 — Native extended label-shape statistics

- [ ] Re-run the same gate set after the compatibility-mode root cause is removed. Completion condition: the current branch's registration targets and workspace formatting compile without the legacy feature changing the public image type.

## MIG-658-01 — Remove relocated Burn compatibility surfaces

- [ ] Port every active consumer of `burn_compat_types` to its native Coeus operation, then delete that module and the `burn-compat` feature in the same breaking cutover. Completion condition: `xtask burn-migration-audit` reports no relocated compatibility surfaces and the source count falls without an allowlist expansion.

## SEC-656-01 — Workspace license metadata

- [ ] Merge the metadata and dependency updates. Completion condition: the RITK default branch contains both security commits.

## SEC-656-02 — DICOM JPEG XL security update

- [ ] Merge the upstream security update and refresh the Kwavers provider pin. Completion condition: Kwavers resolves no vulnerable `jxl-grid` 0.5.3.

## MIG-500-01 — Reject hidden Burn dependency relocation

- [ ] Replace each affected consumer with its native Coeus/Leto operation, delete the Burn aliases, refresh the allowlist only after real source removal, and rerun the same gates.

## DEP-497-01 — Dead `burn` production-dep strip (17 leaf crates)

### Residual risk (gap_audit.md candidates)

- Closed by MIG-499-01: the 2 pre-existing private intra-doc links were removed
  with the redundant erosion surface and the Euclidean native-doc correction.
- `ritk-snap::app::pacs_ops` full-workspace-nextest-only timeout is
  resource-contention flakiness under full-parallel load, not a code
  hang (isolated run: 2.1s pass) — worth a nextest per-binary
  parallelism cap if it recurs, not a correctness defect.

## Sprint 464 — PERF-432-01 Precise Op-Level Profiling, One Prior Claim Retracted

- [ ] PERF-432-01 [patch] remains OPEN. Precisely localized (gather+weighted- sum block, `crates/ritk-transform/src/transform/bspline/interpolation/ dim3.rs`) but the real fix — a custom fused gather+weighted-sum kernel, or bypassing burn's generic autodiff for this hot path with a hand-derived analytic backward — is an architectural change beyond a scoped patch; filed as an investigation target (quantify whether a hand-written CPU gather-weighted-sum is worth its correctness-verification cost), not a ready increment.
- [ ] MIG-456-04 [minor]: Color-volume Coeus variants; DICOM Coeus reader.
- [ ] MIG-433-06 / MIG-437-04 / MIG-439-03 [minor]: burn→Atlas backend migration.
- [ ] BACKLOG: Wire `ritk-snap::ui::coordinate_system` into a UI feature or remove.

## Sprint 463 — PERF-432-01 Profiling: Bottleneck Located, One Approach Rejected

- [ ] PERF-432-01 [patch] remains OPEN. Next increment (see backlog.md for full detail): (1) cache the iteration-invariant fixed-image grid in `MeanSquaredError::forward` instead of recomputing it 200×/call — requires a design decision on trait-level caching vs. hoisting, since it touches every `Metric` implementor; (2) hoist the 5 static index/mask tensors in `transform_3d_chunk` to a per-`BSplineTransform` cache (zero value-risk, removes 5 of ~30 ops/call); (3) further fusion of the basis-weight outer product and gather-weighted-sum, the same direction as the prior partial "fused MSE interpolation" win.
- [ ] MIG-456-04 [minor]: Color-volume Coeus variants; DICOM Coeus reader.
- [ ] MIG-433-06 / MIG-437-04 / MIG-439-03 [minor]: burn→Atlas backend migration.
- [ ] BACKLOG: Wire `ritk-snap::ui::coordinate_system` into a UI feature or remove.

## Sprint 462 — Workspace-Wide Orphaned-Module Sweep (SEC-461-04)

- [ ] BACKLOG: Wire `ritk-snap::ui::coordinate_system` (LPS/RAS conversion + DICOM patient-position formatting, fully tested) into an actual coordinate readout UI feature, or remove if the display feature is never built.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-456-04 [minor]: Color-volume Coeus variants; DICOM Coeus reader.
- [ ] MIG-433-06 / MIG-437-04 / MIG-439-03 [minor]: burn→Atlas backend migration.

## Sprint 461 — Restore Orphaned DICOM color_multiframe Module

- [ ] SEC-461-04 [patch]: Tooling-based orphaned-module sweep (see note above).
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-456-04 [minor]: Color-volume Coeus variants; DICOM Coeus reader.
- [ ] MIG-433-06 / MIG-437-04 / MIG-439-03 [minor]: burn→Atlas backend migration.

## Sprint 460 — Workspace Unblock + DICOM Multiframe Alloc Bound

- [ ] SEC-460-03 [patch]: Bound the DICOM color/color-multiframe `vec![0.0; total_samples]` full allocations (checked_mul present; eager-alloc-from-header remains — needs incremental build or a pixel-data-length bound).
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-456-04 [minor]: Color-volume Coeus variants; DICOM Coeus reader.
- [ ] MIG-433-06 / MIG-437-04 / MIG-439-03 [minor]: burn→Atlas backend migration.

## Sprint 459 — MINC Shape-Exceeds-Data Regression (TEST-447-05)

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect (oldest open perf item; prior fused-MSE/identity-direction attempts did not close the 30s budget).
- [ ] SEC-459-02 [patch]: Audit the DICOM-level `PixelLayout` (Rows×Columns) construction for an upstream pixel-count bound feeding the codecs.
- [ ] MIG-456-04 [minor]: Color-volume Coeus variants; DICOM Coeus reader.
- [ ] MIG-433-06 / MIG-437-04 / MIG-439-03 [minor]: burn→Atlas backend migration.

## Sprint 458 — JPEG/J2K Decode Dimension Bounds (SSOT)

- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-456-04 [minor]: Color-volume Coeus variants; DICOM Coeus reader.
- [ ] MIG-433-06 / MIG-437-04 / MIG-439-03 [minor]: burn→Atlas backend migration.

## Sprint 457 — JPEG-LS Decode DoS Hardening

- [ ] MIG-456-04 [minor]: Color-volume Coeus variants; DICOM Coeus reader.
- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 / MIG-437-04 / MIG-439-03 [minor]: burn→Atlas backend migration.

## Sprint 456 — TIFF Coeus Reader Path (grayscale frontier complete)

- [ ] MIG-456-04 [minor]: Color-volume Coeus variants across jpeg/png/tiff; DICOM Coeus reader (separate API surface).
- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Registration N4 bias correction onto Coeus/Leto/Hephaestus.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases.

## Sprint 455 — PNG Coeus Reader Paths

- [ ] MIG-455-04 [minor]: Coeus reader path for ritk-tiff; color-volume Coeus variants across jpeg/png/tiff. (Grayscale: mgh/nifti/metaimage/minc/jpeg/png done.)
- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Registration N4 bias correction onto Coeus/Leto/Hephaestus.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases.

## Sprint 454 — JPEG Coeus Reader + Decode Optimization

- [ ] MIG-454-04 [minor]: Coeus reader paths for ritk-png and ritk-tiff (same `decode_* + into_raw` pattern); JPEG/PNG/TIFF color-volume variants.
- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Registration N4 bias correction onto Coeus/Leto/Hephaestus.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases.

## Sprint 453 — MINC Coeus-Backed Reader Path

- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] MIG-453-04 [minor]: Coeus NIfTI label-map reader (`read_nifti_labels`).
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Registration N4 bias correction onto Coeus/Leto/Hephaestus.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases.

## Sprint 452 — MINC HDF5 Writer Round-Trip Fix

- [ ] MIG-451-04 [minor]: MINC Coeus reader path — now unblocked by the round-trip fix (the write→read test gives a value-semantic oracle for it); was reverted once by a concurrent agent, re-attempt with a fast commit.
- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Registration N4 bias correction onto Coeus/Leto/Hephaestus.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases.

## Sprint 451 — MetaImage Coeus-Backed Reader Path

- [ ] MIG-451-04 [minor]: Coeus reader path for ritk-minc (HDF5); Coeus NIfTI label-map reader. Single-volume image readers (mgh, nifti, metaimage) now done.
- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Registration N4 bias correction onto Coeus/Leto/Hephaestus.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases.

## Sprint 450 — NIfTI Coeus-Backed Reader Path

- [ ] MIG-450-04 [minor]: Coeus NIfTI label-map reader (`read_nifti_labels` currently Burn/Vec only); and apply the pattern to ritk-metaimage, ritk-minc.
- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Registration N4 bias correction onto Coeus/Leto/Hephaestus.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases.

## Sprint 449 — MGH Coeus-Backed Reader Path (burn→Atlas migration begin)

- [ ] MIG-449-05 [minor]: Apply the same `decode_*` split + Coeus reader path to the sibling readers (ritk-nifti, ritk-metaimage, ritk-minc) following the ritk-mgh pattern.
- [ ] TEST-447-05 [patch]: MINC format-level hostile-fixture regression.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Registration N4 bias correction onto Coeus/Leto/Hephaestus.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases.

## Sprint 448 — NIfTI Header SoC Decomposition

- [ ] TEST-447-05 [patch]: Format-level hostile-fixture regression for the MINC reader (HDF5 shape > backing bytes).
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 447 — Centralized Bounded Reads Across Format Parsers

- [ ] TEST-447-05 [patch]: Format-level hostile-fixture regression for the MINC reader (requires forging an HDF5 dataset with shape > backing bytes; the `read_bounded_with` primitive is unit-tested in ritk-core).
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 446 — VTK Reader Untrusted-Input Allocation Hardening

- [ ] SEC-446-05 [patch]: Apply the same untrusted-input allocation hardening to the remaining format-parser crates (ritk-nrrd, ritk-nifti, ritk-metaimage, ritk-mgh, ritk-minc) whose readers reserve from header count/size fields.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 445 — MAD Noise Work-Buffer Reuse

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 444 — Histogram Matching Allocation Cleanup

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 443 — Nyul-Udupa Output Buffer Reuse

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 442 — Statistics Full-Image Owned Extraction

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 441 — Statistics Masked-Buffer Allocation Cleanup

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 440 — Coeus Image Flat-Buffer Boundary

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 439 — I/O Workspace Dependency Cleanup

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.
- [ ] MIG-439-03 [minor]: Replace remaining `burn_ndarray` backend aliases and tests with Atlas-backed Coeus/Leto surfaces where each crate boundary is migrated.

## Sprint 438 — Registration Leto Dependency Cleanup

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.

## Sprint 437 — CLI Leto MI Boundary Cleanup

- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.
- [ ] MIG-437-04 [minor]: Replace the CLI-wide Burn NdArray backend alias with an Atlas-backed backend after the image/filter/IO command boundaries are migrated.

## Sprint 436 — Fused Coordinate-Convention Coverage

- [ ] PERF-432-01 [patch]: Continue reducing `bspline_registers_offset_sphere`; latest focused row is 80.456s and still exceeds the strict runtime budget.
- [ ] PERF-432-01 [patch]: Remaining B-spline registration runtime defect.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.

## Sprint 435 — Fused MSE Interpolation Cleanup

- [ ] PERF-435-05 [patch]: Continue optimizing `bspline_registers_offset_sphere`; focused nextest improved to 76.441s but still exceeds the strict 60s termination budget.
- [ ] PERF-432-01 [patch]: Continue with the remaining MSE B-spline runtime defect; this slice removed one intermediate materialization path but did not bring the row below the 60s budget.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.

## Sprint 434 — Registration Convergence Runtime Budget

- [ ] PERF-434-05 [patch]: Optimize `bspline_registers_offset_sphere`; this MSE B-spline row remains above the strict 60s termination budget at 87.615s and needs a production hot-path fix rather than convergence-window truncation.
- [ ] PERF-432-01 [patch]: Continue with the remaining MSE B-spline runtime defect after this convergence-policy slice.
- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation.

## Sprint 433 — Coeus Preprocessing Smoothing

- [ ] MIG-433-06 [minor]: Migrate registration N4 bias correction to a Coeus/Leto/Hephaestus-backed bias-field implementation before the Coeus preprocessing executor can run every preprocessing step.
- [ ] PERF-432-01 [patch]: Profile and reduce long-running registration integration tests currently covered by `.config/nextest.toml` 600s overrides; the latest full package run passed but still violates the stricter 30s/60s AGENTS.md budget.

## Sprint 432 — Coeus Registration Preprocessing Scalar Consumer

- [ ] PERF-432-01 [patch]: Profile and reduce long-running registration integration tests currently covered by `.config/nextest.toml` 600s overrides; the full package run passed but violates the stricter 30s/60s AGENTS.md budget.

## Sprint 431 — Coeus Statistics Image Consumer

- [ ] MIG-431-06 [minor]: Migrate the next production image consumer from the legacy Burn `Image<B, D>` helper path to a Coeus image path, prioritizing consumers with existing slice-level SSOTs.

## Sprint 427 — Coeus Tensor-Ops Contract Tests

- [ ] MIG-387-01 [arch]: Continue replacing production Burn tensor boundaries with Coeus only where a complete tensor/image contract and focused tests are available. This slice strengthens the Coeus contract tests and does not claim production Burn removal.

## Sprint 426 — NIfTI Fixture Provenance and Import Coverage

- [ ] MIG-425-01 [minor]: Add paired NIfTI `ni1`/`ni2` `.hdr`/`.img` support if a caller needs NIfTI pairs; keep Analyze 7.5 routed through `ritk-analyze`.

## Sprint 425 — Native NIfTI-2 Single-File Codec

- [ ] MIG-425-01 [minor]: Add paired NIfTI `ni1`/`ni2` `.hdr`/`.img` support if a caller needs NIfTI pairs; do not route Analyze 7.5 through `ritk-nifti`.
- [ ] MIG-424-02 [minor]: Extend native NIfTI datatype coverage beyond Float32 images and UInt32/Float32 labels when a caller needs additional scalar kinds.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.

## Sprint 424 — Native RITK NIfTI Codec

- [ ] MIG-424-02 [minor]: Extend native NIfTI datatype coverage beyond Float32 images and UInt32/Float32 labels when a caller needs additional scalar kinds.
- [ ] MIG-424-03 [minor]: Add NIfTI-2 and header/img pair support if those file variants become required by an integration contract.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.

## Sprint 423 — NIfTI Shape Bounds SSOT

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.
- [ ] PERF-419-01 [patch]: Profile registration integration tests that exceed the 30s slow-test budget.

## Sprint 422 — PACS Worker Send Signal

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.
- [ ] PERF-419-01 [patch]: Profile registration integration tests that exceed the 30s slow-test budget.

## Sprint 421 — Direct Moirai DICOM Series Loading

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.
- [ ] PROVIDER-420-01 [patch]: Land the Hermes provider dispatch-bound cleanup separately; the local Hermes tree remains dirty outside this RITK branch.

## Sprint 420 — Direct Moirai Filter Diffusion Enumeration

- [ ] PROVIDER-420-01 [patch]: Land the Hermes provider dispatch-bound cleanup separately; the local Hermes tree is already dirty, and full `hermes-simd` rustfmt is blocked by unrelated pre-existing `axpy.rs` formatting drift.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.

## Sprint 419 — Direct Moirai Registration Enumeration

- [ ] PERF-419-01 [patch]: Profile registration integration tests that exceed the 30s slow-test budget; this gate passed functionally but did not prove the selected paths are performance-clean.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.

## Sprint 418 — Direct Moirai Segmentation Enumeration

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.

## Sprint 417 — Level-set Safe Moirai Metrics

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.

## Sprint 416 — GrowCut Safe Moirai Assignment

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.

## Sprint 415 — SLIC Safe Moirai Assignment

- [ ] MIG-415-06 [patch]: Continue removing the same raw-pointer side-write pattern from level-set kernels with focused value-semantic gates.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.

## Sprint 414 — Gaia MeshBuilder Array API Migration

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.

## Sprint 413 — BinShrink Moirai Chunk Write Cleanup

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.
- [ ] MIG-387-02 [arch]: Continue mesh-only spatial cleanup where Gaia-backed mesh paths still expose `nalgebra::Point3` through the Gaia contract.
- [ ] MIG-413-05 [patch]: Continue stale Rayon wording cleanup in registration Parzen/CMA-ES docs after verifying each path's Moirai execution surface.

## Sprint 412 — Statistics Atlas Dependency Cleanup

- [ ] MIG-387-01 [arch]: Continue Burn/Coeus tensor replacement as a separate contract-preserving slice.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary removal in NIfTI, CLI, registration, and I/O packages.
- [ ] MIG-387-02 [arch]: Continue mesh-only spatial cleanup where Gaia-backed mesh paths still expose `nalgebra::Point3` through the Gaia contract.

## Sprint 411 — SNAP Spatial Dependency Cleanup

- [ ] MIG-387-02 [arch]: Continue mesh-only spatial cleanup. Gaia-backed mesh paths still use Gaia's current `Point3r`/`nalgebra::Point3` public contract, so those require either a Gaia API extension or a mesh-bounded RITK slice.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus and `ndarray` boundary migration as separate contract-preserving slices.

## Sprint 410 — PNG Spatial Dependency Cleanup

- [ ] MIG-387-02 [arch]: Continue SNAP and mesh-only spatial cleanup. Gaia-backed mesh paths still use Gaia's current `Point3r`/`nalgebra::Point3` public contract, so those require either a Gaia API extension or a mesh-bounded RITK slice.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus and `ndarray` boundary migration as separate contract-preserving slices.

## Sprint 409 — DICOM/MINC/Filter Spatial Leto Slice

- [ ] MIG-387-02 [arch]: Continue PNG/SNAP and mesh-only spatial cleanup in separate bounded-context slices; `ritk-io` still has VTK mesh test `nalgebra` use and keeps its manifest dependency until that mesh slice is migrated.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus migration as a separate tensor-boundary redesign; Burn remains a public backend/tensor contract across image, filter, registration, model, IO, and Python crates.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary audit for NIfTI/file-format conversion and Python/numpy boundary code.

## Sprint 408 — Spatial Leto SSOT Slice

- [ ] MIG-387-02 [arch]: Continue DICOM IO geometry, MINC, PNG/SNAP/filter spatial call-site cleanup in a separate slice after this spatial SSOT merge.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus migration as a separate tensor-boundary redesign; Burn remains a public backend/tensor contract across image, filter, registration, model, IO, and Python crates.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary audit; current remaining direct use includes NIfTI/file-format conversion and Python/numpy boundary code.

## Sprint 407 — Leto Classical Registration Slice

- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra` surfaces in `ritk-spatial`, DICOM IO geometry, and medical-image spatial metadata only after the spatial SSOT has a provider-backed fixed-math representation and all call sites are migrated in one slice.
- [ ] MIG-387-01 [arch]: Continue Burn/Coeus migration as a separate tensor-boundary redesign; Burn remains a public backend/tensor contract across image, filter, registration, model, IO, and Python crates.
- [ ] MIG-387-01 [arch]: Continue `ndarray` boundary audit; current remaining direct use includes NIfTI/file-format conversion and Python/numpy boundary code.

## Sprint 406 — Global Format Gate

- [ ] `cargo test --doc -p ritk-core -p ritk-filter -p ritk-interpolation -p ritk-registration -p ritk-segmentation -p ritk-tensor-ops` -> blocked by dirty `D:\atlas\repos\coeus` provider compile errors in `coeus-autograd`
- [ ] `cargo doc -p ritk-core -p ritk-filter -p ritk-interpolation -p ritk-registration -p ritk-segmentation -p ritk-tensor-ops --no-deps` -> blocked by the same `coeus-autograd` compile errors
- [ ] PERF-406-02 [patch]: Profile and reduce slow registration tests observed in Sprint 406 (`test_bspline_cr_registration_small` 161s, `test_multires_cr_registration` 116s, `bspline_registers_offset_sphere` 81s, plus several 30s-40s rigid/affine/versor rows).
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with public VTK cell-list storage only after an ADR/migration plan, because `VtkPolyData`/`VtkUnstructuredGrid` expose nested cell vectors as public fields.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.
- [ ] MIG-406-01 [patch]: Remove stale `rayon` wording from comments/docs where production paths already use Moirai-backed helpers.
- [ ] COEUS-406-01 [patch]: Fix dirty `coeus-autograd` provider compile errors blocking RITK doctest/doc gates after the Coeus `0.2.6` lock refresh.

## Sprint 405 — FFT Padding Bounds

- [ ] `cargo fmt --check` -> blocked by pre-existing unrelated formatting drift outside this slice (`ritk-core`, `ritk-filter` deconvolution/diffusion, `ritk-interpolation`, `ritk-segmentation`, `ritk-tensor-ops`)
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with public VTK cell-list storage only after an ADR/migration plan, because `VtkPolyData`/`VtkUnstructuredGrid` expose nested cell vectors as public fields.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 404 — Apollo FFT Dependency Cleanup

- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with public VTK cell-list storage only after an ADR/migration plan, because `VtkPolyData`/`VtkUnstructuredGrid` expose nested cell vectors as public fields.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 403 — Vector Confidence Fallibility

- [ ] `cargo semver-checks -p ritk-segmentation` -> blocked because `ritk-segmentation` is not published on crates.io for registry baseline comparison
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with public VTK cell-list storage only after an ADR/migration plan, because `VtkPolyData`/`VtkUnstructuredGrid` expose nested cell vectors as public fields.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 402 — VTU Exact Cell Arrays

- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with public VTK cell-list storage only after an ADR/migration plan, because `VtkPolyData`/`VtkUnstructuredGrid` expose nested cell vectors as public fields.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 401 — VTK Cell Streaming and Parse Errors

- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with public VTK cell-list storage only after an ADR/migration plan, because `VtkPolyData`/`VtkUnstructuredGrid` expose nested cell vectors as public fields.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 400 — NIfTI Spatial Field Validation

- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 399 — MINC Exact Dimension Attributes

- [ ] SAFE-399-01 [patch]: Continue hostile-header/value audit in the remaining NIfTI parser for exact shape/affine field consumption and bounded allocation on malformed fields.
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 398 — MetaImage Exact Payload Bounds

- [ ] SAFE-398-01 [patch]: Continue hostile-header/value audit in remaining sibling image parsers (MINC, NIfTI) for exact vector/matrix field consumption and bounded allocation on malformed fields.
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 397 — RT Plan Exact Sequence Numerics

- [ ] SAFE-397-01 [patch]: Continue hostile-header/value audit in remaining sibling image parsers (MetaImage, MINC, NIfTI) for exact vector/matrix field consumption and bounded allocation on malformed fields.
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 396 — RT Dose Exact Grid Fields

- [ ] SAFE-396-01 [patch]: Continue hostile-header/value audit in remaining sibling image and RT parsers (MetaImage, MINC, NIfTI, RT Plan) for exact vector/matrix field consumption and bounded allocation on malformed fields.
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 395 — RT Struct Exact ContourData

- [ ] `cargo fmt --check` workspace gate still blocked by pre-existing unrelated formatting drift recorded in Sprint 388.
- [ ] SAFE-395-01 [patch]: Continue hostile-header/value audit in sibling image and RT parsers (MetaImage, MINC, NIfTI, RT Dose/Plan) for exact vector/matrix field consumption and bounded allocation on malformed fields.
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 394 — NRRD Exact Vector Fields

- [ ] `cargo fmt --check` workspace gate still blocked by pre-existing unrelated formatting drift recorded in Sprint 388.
- [ ] SAFE-394-01 [patch]: Continue hostile-header audit in sibling image format parsers (MetaImage, MGH, MINC, NIfTI) for exact vector/matrix field consumption and bounded allocation on malformed fields.
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 393 — NRRD Unterminated Vector Rejection

- [ ] `cargo fmt --check` workspace gate still blocked by pre-existing unrelated formatting drift recorded in Sprint 388.
- [ ] SAFE-393-02 [patch]: Continue hostile-header audit for NRRD and sibling format parsers: reject trailing unparsed tokens where the file format requires an exact vector list, and preserve existing permissive behavior only where a compatibility contract requires it.
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 392 — NRRD Fixed-Vector Header Parsing

- [ ] `cargo fmt --check` workspace gate still blocked by pre-existing unrelated formatting drift recorded in Sprint 388.
- [ ] PERF-392-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. VTK cell-list storage remains a public model change.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 391 — Binary VTI Appended Streaming

- [ ] `cargo fmt --check` workspace gate still blocked by pre-existing unrelated formatting drift recorded in Sprint 388.
- [ ] PERF-391-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK public cell-list storage. The VTK cell-list model remains a broader public API/storage change and needs an ADR before breaking `Vec<Vec<u32>>` fields.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 390 — TIFF Flat Page Accumulation

- [ ] `cargo fmt --check` workspace gate still blocked by pre-existing unrelated formatting drift recorded in Sprint 388.
- [ ] PERF-390-02 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK cell-list storage as next candidates.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 389 — Inverse Displacement Coefficient Flattening

- [ ] `cargo fmt --check` workspace gate still blocked by pre-existing unrelated formatting drift recorded in Sprint 388.
- [ ] PERF-389-01 [patch]: Continue flat-buffer audit with `VectorConfidenceConnected` channel buffers and VTK cell-list storage as next candidates.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 388 — Linear Kernel Slice Semantics

- [ ] `cargo fmt --check` workspace gate still blocked by pre-existing unrelated formatting drift in `ritk-core`, `ritk-filter`, non-linear `ritk-interpolation`, `ritk-registration`, `ritk-segmentation`, and `ritk-tensor-ops` files.
- [ ] PERF-387-02 [patch]: Continue flattening small matrix/vector hot paths where API-compatible: `vector_confidence_connected` channel buffers, `inverse_displacement` derivative matrices, and VTK cell lists remain next audit candidates.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 387 — Region-Growing Matrix Flattening + Legacy Cleanup

- [ ] `cargo fmt --check` workspace gate blocked by pre-existing formatting drift in unrelated files; not applied to avoid unrelated churn.
- [ ] PERF-387-02 [patch]: Continue flattening small matrix/vector hot paths where API-compatible: `vector_confidence_connected` channel buffers, `inverse_displacement` derivative matrices, and VTK cell lists are the next audit candidates.
- [ ] MIG-387-01 [arch]: Continue replacing remaining `nalgebra`/`ndarray`/`burn` production surfaces with `leto`/`coeus`/`hephaestus` only where the Atlas crate has an equivalent verified contract.

## Sprint 386 — CurvatureFlow f64, Interior Peel Perf, Laplacian Bug Fix, cmake Parity +18

| Run | cmake-data | Broad suite | Rust filter | Rust seg | Rust reg | Notes |
|-----|-----------|------------|------------|---------|---------|-------|
| Sprint 385 exit | 430 | 1078 | 928 | 430 | 654 | |
| Sprint 386 (this) | **448** | **1096** | **928** | **431** | **654** | +18 cmake; 2 correctness; 2.2× CF perf |
- [ ] cmake-data: ContourExtractor2D (2 tests) skip because `sitk.ContourExtractor2DImageFilter` unavailable in test environment — environment limitation, not a code gap.
| Run | cmake-data | Broad suite | Rust filter | Rust seg | Rust reg | Notes |
|-----|-----------|------------|------------|---------|---------|-------|
| Sprint 384 exit | 429 | 1077 | 928 | 430 | 654 | |
| Sprint 385 (this) | **430** | **1078** | **928** | **430** | **654** | +1 cmake (shift_scale); 5 correctness+perf fixes |
- [ ] PERF-381-01 [partial]: `cargo bench` baseline timings for `separable_box_3d` / EDT not yet recorded.
- [ ] FRANGI-QA-01: Frangi/Sato parity tests against sitk `ObjectnessMeasure`/`Hessian` outputs at multiple σ not yet added; further validation needed.
- [ ] CHAN-VESE-QA-01: ScalarChanAndVese pixel-exact comparison against sitk after mu+dt fix; current test is structural only.
- [ ] ISOLATED-WS-QA-01: Isolated watershed with complex 3D real images — the flat-region plateau handling may still diverge for certain topologies.
| Run | cmake-data | Broad suite | Rust filter | Rust seg | Rust reg | Notes |
|-----|-----------|------------|------------|---------|---------|-------|
| Sprint 383 exit | 421 | 1068 | 920 | 431 | 2002 | |
| Sprint 384 (this) | **429** | **1077** | **926** | **1356** | **652** | +8 cmake, 14 perf+correctness fixes |
- [ ] NEW-384-01 [minor]: `shift_scale` Python binding not yet exposed. 1 cmake test skips cleanly.
- [ ] PERF-381-01 [partial]: `cargo bench` baseline timings for separable_box_3d / EDT not yet recorded. Requires `cargo bench` on release build.
- [ ] CORR-384-01 [major]: Frangi vesselness Hessian via finite-diff on sampled Gaussian vs ITK's 2nd-order Deriche IIR. Audit C-1 — fix is to call `recursive_gaussian_directional(Second)` per axis; existing IIR machinery is available. Significant correctness improvement.
- [ ] CORR-384-03 [major]: `ScalarChanAndVeseDenseLevelSet` — 19% match; SharedData region-mean propagation + adaptive dt needed.
- [ ] PERF-384-01 [high]: `window_cc_stats` O(N·w³) 2-pass scan → O(N) centered-residual integral image form. At r=3 default, ~114× reduction. Algorithmic fix, not a parallelism patch.
- [ ] PERF-384-02 [high]: `geodesic_active_contour` convergence — max|Δφ|/dt vs ITK RMS. Different stopping behavior; ITK RMS is more numerically stable.

## Sprint 383 — cmake Coverage, Perf/Memory, Clippy/Doc Cleanup (Active)

| Run | cmake-data | Broad suite | Rust tests | Notes |
|-----|-----------|------------|-----------|-------|
| Sprint 382 exit | 404 | 818 | 910 | |
| Stale binary fix | 416 | — | — | +12 (InverseDisplacement+others) |
| Sprint 383 (this) | **421** | **1082** | **1351** | +7 new filter tests; broad +264 |
- [ ] PERF-381-01 [partial]: Benchmark scaffold (separable_box, euclidean_dt) added in Sprint 382; baseline timings not yet recorded. Requires `cargo bench` on release build.
- [ ] NEW-383-02 [minor]: 3 cmake tests currently `skip` (AntiAliasBinary, CannySegmentationLevelSet, ContourExtractor2D) because the installed SimpleITK wheel doesn't expose these filters in the test environment. Tests are wired to the real implementations and will activate automatically when a compatible sitk wheel is installed.

## Sprint 382 — Deconvolution Crop Fix, cmake Coverage Expansion (Active)

| Run | cmake-data | Broad suite | Notes |
|-----|-----------|------------|-------|
| Sprint 381 exit | 400 | 814 | |
| f78197de (VectorConfidenceConnected) | 401 | — | +1 |
| Sprint 382 (this) | **404** | **818** | +3 blurred-deconv, +4 broad |
- [ ] DOC-381-02 [patch]: **16 pre-existing intra-doc-link warnings** — 16 rustdoc unresolved links to private items. Non-blocking. Target Sprint 383 cleanup.
- [ ] PERF-381-01 [partial]: Benchmark scaffold added; actual baseline timings not yet recorded (require `cargo bench` run on release build). Record before claiming speedup.

## Sprint 381 — Wiener Formula Fix, Parallel Box/EDT, cmake CoherenceEnhancingDiffusion

| Run | cmake-data | Broad suite | Notes |
|-----|-----------|------------|-------|
| Sprint 380 exit | 375 | 1034 | |
| Sprint 380→381 pyd sync | 394 | — | +19 from Toboggan/LabelMapContourOverlay/MedianProjection |
| Sprint 381 (this) | **400** | **814** | +6 CED; broad excludes scipy-missing |
- [ ] GAP-381-01 [patch]: **Wiener/Inverse deconvolution crop-position scale divergence** — Root-cause identified (Sprint 381): `ifft_and_crop` crops from [0,0,0] of the padded IFFT output, yielding values ~400–3000× larger than sitk's for band-limited blurred input. ITK applies a different crop region (or different boundary condition). Fix requires careful analysis of ITK's `FFTConvolutionImageFilter::CropOutput` region computation; would close the Pearson≈0 divergence for WienerDeconvolution and improve InverseDeconvolution from Pearson≈0.42–0.66 to Pearson≥0.90.
- [ ] PERF-381-01 [patch]: **Verify separable_box_3d and EDT Phase 3 speedups with criterion benchmarks** — Both parallelizations are bit-identical to serial (verified by existing tests) but no benchmark baseline recorded yet for Phase 3 or separable_box_3d. Add benches/separable_box.rs before merging parallel claim.
- [ ] DOC-381-02 [patch]: 16 pre-existing intra-doc-link warnings (unresolved rustdoc links to private items). Non-blocking; target next sprint cleanup pass.
| Run | cmake-data | Broad suite | Notes |
|-----|-----------|------------|-------|
| Sprint 379 exit | 354 | 983 | stale pyd resolved |
| Sprint 380 (this) | **375** | **1034** | +21 cmake, +51 broad, 0 failures |
- [ ] PERF-380-04 [patch]: **euclidean_dt Phase 3 parallelism** — Z-columns non-contiguous in z-major layout; requires transposed intermediate buffer. Deferred: phases 1+2 already give ~2/3 of the serial savings; Phase 3 is the minor remainder.
- [ ] PERF-380-05 [patch]: **separable_box_3d moirai parallelism** — X/Y/Z passes each have embarrassingly parallel row/column structure; would accelerate all grayscale morphological filters (dilation, erosion, opening, closing, gradient, top-hat).
- [ ] GAP-380-01 [patch]: **Wiener deconvolution parameter-semantic investigation** — ritk `noise_to_signal` and sitk `noiseVariance` appear to parameterise the same filter with incompatible units; measured Pearson ≈ 0 across all test values. Needs root-cause analysis (see gap_audit.md).
- [ ] GAP-380-02 [patch]: **MinMaxCurvatureFlow ComputeThreshold divergence** — documented in SITK_CMAKE_EXCLUSIONS.md; test commented out until resolved.
| Run | Passed | Failed | Notes |
|-----|--------|--------|-------|
| Sprint 378 exit | 315 | 25 | stale-pyd, sign, displacement |
| Sprint 379 (this) | 354 | 0 | all resolved |
- [ ] PERF-377-01-HUANG3D — reopen conditions unchanged (>10⁶-voxel workload or SSOT promotion)
- [ ] PERF-377-02-RANGE-LUT — gate-blocked by test contract (728k bins/unit required)
- [ ] FIX-transform_to_displacement_field — pre-existing world-axis ordering vs sitk convention
- [ ] FIX-signed_distance_map — PyO3 ndarray conversion for 4D input shape

### Deferred / carry-forward (next increments)

- [ ] PERF-377-01-HUANG3D [patch→[minor]?] (deferred-with-rationale): **Huang 3-D sliding-histogram MedianFilter** — Perreault-Hebert 2007 §3.2 with `window_hist[n_bins]` + row_in/row_out column-histogram updates. Reopen condition: (a) >10⁶-voxel workload where the algorithm is the bottleneck, or (b) algorithm promotion into the `rank::kernel::neighborhood_rank_3d` SSOT to amortise across rank/percentile. Existing brute-force parallelism is already 10ms at 64³ r=2; 2-D Huang would be a regression (O(r²·n_bins) > current O(r³) at typical n_bins). See `benches/median.rs` for the per-size baseline threshold.
- [ ] PERF-377-02-RANGE-LUT [patch→[minor]?] (gate-blocked by test contract): **BilateralFilter range LUT** — module-level doc on `bilateral.rs` carries the ε-bound derivation; see commit `462a6b63`. A quantised `range_w[|dr|]` LUT over the full intensity range would need qscale > 728k bins/unit to hold the existing 1e-5 test epsilon for σ_r=50. Three options documented in code: 1. Hybrid exp + LUT (≤ 2× at typical σ_r) 2. Loosen test tolerance to a derived ~0.05 HU bound (test-contract change, [minor]) 3. Keep current `exp`-per-neighbour path (default) Reopen when (1) or (2) are justified by an explicit workload or test-contract change.
- [ ] DOC-377-01 [patch]: 16 pre-existing intra-doc-link warnings (rustdoc unresolved link, public docs → private items) accumulated from Sprint 393-395 commits; gated but non-blocking.
- [ ] FMT-377-01 [patch]: working-tree fmt-only diffs from cumulative agent updates (long-line rewraps). Now ~30 files per current `git status`; pure whitespace; next `cargo fmt --all` by next agent or this session will close.

## Sprint 376 — DRY Closure, Build Hardening & Carry-Forward Reconciliation

- [ ] VAR-375-01 [upstream]: `PhantomData<B>` → `PhantomData<fn() -> B>` BLOCKED at `burn-core-0.19.1`
- [ ] CONST-375-02 [toolchain]: const-assert companion for `BSplineTransform` blocked on const_panic_fmt
- [ ] NAMING-362-23 [arch]: sealed trait `DimInterpolation<B>` BLOCKED — ADR required
- [ ] SRP-362-20 [minor]: `FilterKind` ValueEnum separation — partial (slice delivery done; per-family Args structs remain)
- [ ] NAMING-FILTER-01 [major]: `FftConvolution3DFilter` const-generic unification — concurrent-crate changes required
- [ ] N-375-08 [arch]: DRY cross-crate parse utils — promotion trigger requires `ritk-io` → `ritk-core` migration

## Sprint 375 — Architecture Hardening Round 8: SSOT · DRY · NAMING · ENUM · SRP · COMPAT

- [ ] DRY-374-01: `make_image_*`/`make_mask_*` — 68 occurrences [minor] (next round)
- [ ] NAMING-362-23: `transform_1d/_2d/_3d/_4d` [arch] BLOCKED — ADR required
- [ ] SRP-362-20: `FilterArgs` → `FilterKind` [major] BLOCKED
- [ ] NAMING-FILTER-01: `FftConvolution3DFilter` const-generic unification [major] BLOCKED
- [ ] N-375-08: DRY cross-crate parse utils (shared IO codec layer) [arch] BLOCKED

## Sprint 374 — Architecture Hardening Round 7: SSOT · DRY · NAMING · ENUM · SRP · COMPAT

- [ ] NAMING-362-23: `transform_1d/_2d/_3d/_4d` [arch] BLOCKED
- [ ] SRP-362-20: `FilterArgs` → `FilterKind` [major]
- [ ] DRY-374-01: `make_image_*`/`make_mask_*` 35+ copies (next round)
- [ ] SRP-374-03: 21 test blocks in ritk-filter (next round)
- [ ] SRP-374-04: 25 test blocks in ritk-snap (next round)
- [ ] NAMING-374-02, ENUM-374-06, DRY-374-07/08, NAMING-374-05: carry-forward (next round)

## Sprint 373 — J2K interop closure (MQ root cause fixed)

### Open (next increment)

- [ ] SITK-PARITY (codec e2e, automated): add the SimpleITK-written J2K DICOM round-trip as a pytest in `test_simpleitk_parity.py` once the concurrent agent's `fix/sitk-parity-mi-sampling` branch merges (file currently has uncommitted edits on that branch)
- [ ] J2K-LOSSY-97, JLS-INTEROP, CODEC-PERF, REG-MI-FLAKY: carry-forward

## Sprint 371 — J2K multi-code-block tier-2 (J2K-MULTI-CBLK delivered)

- [ ] J2K-INTEROP [patch]: differential decode vs OpenJPEG-encoded reference corpus — now unblocked (conformant tag trees + multi-cblk in place); NEXT
- [ ] J2K-LOSSY-97, JLS-INTEROP, CODEC-PERF, REG-MI-FLAKY: carry-forward

## Sprint 370 — J2K multi-level DWT (J2K-DECODE-DWT delivered)

- [ ] REG-MI-FLAKY [investigate]: carry-forward (in-flight registration wave)
- [ ] J2K-MULTI-CBLK, J2K-LOSSY-97, J2K-INTEROP, JLS-INTEROP, CODEC-PERF: carry-forward

## Sprint 369 — Native JPEG-LS codec: CharLS elimination + NEAR support

- [ ] REG-MI-FLAKY [investigate]: `translation_recovery_shifted_gaussian` fails deterministically (est 1.0 vs true 3.0) in the in-flight NGF/RSGD registration wave — owned by the concurrent registration effort; not in codec blast radius
- [ ] J2K-DECODE-DWT [minor]: carry-forward (Sprint 368)
- [ ] J2K-LOSSY-97, J2K-INTEROP: carry-forward (Sprint 368)

## Sprint 368 — RITK-native JPEG 2000 codec (pure-Rust ISO 15444-1, C/FFI elimination)

- [ ] J2K-DECODE-DWT [minor]: multi-level 5/3 DWT decode (wavelet.rs idwt groundwork in place; `decode_tile_part` currently bails on `num_decomp_levels > 0`)
- [ ] J2K-LOSSY-97 [minor]: 9/7 irreversible wavelet (lossy TS .91 full support)
- [ ] J2K-INTEROP [patch]: differential decode test against an OpenJPEG-encoded reference codestream (real-world DICOM corpus)

## Sprint 367 — Architecture Hardening Round 6: ENUM · NAMING · SRP · SSOT · DRY · COMPAT + ritk-core Crate Extraction

- [ ] NAMING-362-23 [arch]: `transform_1d/_2d/_3d/_4d` — BLOCKED; `DimInterpolation<B>` sealed trait design needed
- [ ] SRP-362-20 [major]: `FilterArgs` → `FilterKind` ValueEnum — carry forward
- [ ] NAMING-FILTER-01 [major]: `FftConvolution3DFilter`/`FftNormalizedCorrelation3DFilter` → const-generic unification
- [ ] TIMEOUT-367: ritk-interpolation 4-test timeout cluster (`dim4`, `dim3_extended`) — investigate under performance_engineering protocol

## Sprint 366 — Architecture Hardening Round 5: NAMING · SSOT · COMPAT · DRY · SRP · ENUM · PRIM

- [ ] NAMING-362-23 [arch]: `transform_1d/_2d/_3d/_4d` — BLOCKED; `DimInterpolation<B>` sealed trait design needed
- [ ] SRP-362-20 [major]: `FilterArgs` → `FilterKind` ValueEnum — carry forward
- [ ] NAMING-FILTER-01 [major]: `FftConvolution3DFilter`/`FftNormalizedCorrelation3DFilter` → const-generic unification

## Sprint 365 — Architecture Hardening Round 4: COMPAT · NAMING · SSOT · SRP · DRY · DIP · ENUM

- [ ] NAMING-362-23 [arch]: `transform_1d/_2d/_3d/_4d` — BLOCKED; `DimInterpolation<B>` sealed trait design needed
- [ ] SRP-362-20 [major]: `FilterArgs` → `FilterKind` ValueEnum — carry forward
- [ ] ENUM-365-03 [minor]: `ResampleArgs.interpolation: String` → `InterpolationMode` ValueEnum
- [ ] NAMING-CORE-01 [patch]: `gaussian_kernel_1d` → `gaussian_kernel` (cross-crate callers)
- [ ] NAMING-FILTER-01 [major]: FftConvolution*3DFilter → const-generic unification

## Sprint 364 — Architecture Hardening Round 3: COMPAT · NAMING · SSOT · CACHE · SRP · PRIM · ENUM

- [ ] DIP-362-13 [minor]: `RegistrationCallbackSet` DIP — deferred; requires surveying `src/progress/` first
- [ ] NAMING-362-23 [patch]: `transform_1d/_2d/_3d/_4d` — **BLOCKED** [arch] — duplicate method names on same type
- [ ] SRP-362-20 [major]: `FilterArgs` (46 fields) → `FilterKind` ValueEnum — carry forward
- [ ] ENUM-365-01 [minor]: `StatsArgs.metric: String` → `StatMetric` ValueEnum — **Done** (Patch 19)
- [ ] ENUM-365-02 [minor]: `RegisterArgs.method: String` → `RegistrationMethod` ValueEnum — **Done** (Patch 20)
- [ ] ENUM-365-03 [minor]: `ResampleArgs.interpolation: String` → `InterpolationMethod` ValueEnum

## Sprint 363 — Architecture Hardening Round 2: DRY · SRP · PRIM · NAMING · CACHE

- [ ] DIP-362-13 [minor]: `RegistrationCallbackSet` DIP — deferred; requires surveying `src/progress/` ProgressTracker internals first
- [ ] NAMING-362-23 [patch]: `transform_1d/_2d/_3d/_4d` — **BLOCKED**: duplicate method names on same type; [arch] refactor required
- [ ] SRP-362-20 [major]: `FilterArgs` (46 fields) → `FilterKind` ValueEnum — carry forward

## Sprint 362 — Architecture Hardening: SSOT · DRY · SRP · DIP · Naming

- [ ] DRY-362-04 [minor]: `UnaryImageFilter<Op>` + `UnaryPixelOp` trait; collapse `abs/sqrt/exp/log/square` (5 files, ~570L → ~100L + type aliases); generalize `D=3` → `const D: usize`
- [ ] PRIM-362-12 [minor]: `EarlyStoppingPolicy::Enabled { patience, min_improvement }` — bundle orphaned fields into enum variant
- [ ] DIP-362-13 [minor]: `Registration::with_config` DIP fix — `RegistrationCallbackSet` builder decouples engine from concrete callback types
- [ ] SRP-362-18 [patch]: Split `dicom/seg/tests/convert.rs` (554L) → 4 test modules
- [ ] SRP-362-19 [patch]: Split `dicom/series.rs` → `series/{types,scan,loader}.rs`; replace `Arc<Mutex>` scan pattern with collect-and-merge
- [ ] SRP-362-20 [major]: `FilterArgs` (46 fields) → `FilterKind` `ValueEnum` + `#[command(flatten)]` per-family structs; `SegmentArgs` same treatment
- [ ] NAMING-362-23 [patch]: `transform_1d/_2d/_3d/_4d` in `bspline/interpolation/` → `transform_points_impl` dispatching on `D` — BLOCKED: duplicate method names on same type across impl blocks; requires [arch] refactor
- [ ] NAMING-362-24 [patch]: `spatial_gradient_2d/_3d`, `spatial_laplacian_2d/_3d` → move to `deformable_field_ops/`, surface only through `dispatch.rs`
- [ ] PRIM-362-25 [minor]: `IntensityRange { min, max }` validating newtype; adopt in `MinMaxNormalizer.target_{min,max}` and `ZScore` params
- [ ] PRIM-362-27 [minor]: `DicomSeriesInfo` — replace `ArrayString<64>` public fields with `&str` accessor; keep `ArrayString` internal

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

## gap-audit-2026-08-20 (owner: atlas-gap-audit)

### RITK-GAP-2026-08-20-01 — collapse `X` / `X_native`

- [ ] Enumerate the closure: every `pub fn *_native`, every `*Native*` type, and their callers across `crates/`, `examples/`, and the book samples. Record the count per crate as the ratchet baseline.
- [ ] Draft the ADR. This supersedes ADR 0002's transitional naming, so revise 0002 in place with a dated note rather than adding a parallel record.
- [ ] Increment 1: leaf format crates (`ritk-nifti`, `ritk-nrrd`, `ritk-metaimage`, `ritk-mgh`, `ritk-analyze`, `ritk-minc`, `ritk-png`, `ritk-tiff`, `ritk-jpeg`) — base name takes the Coeus signature, the old one is deleted, every call site in the same commit.
- [ ] Increment 2: `ritk-image`, `ritk-transform`, `ritk-interpolation`, `ritk-tensor-ops`.
- [ ] Increment 3: `ritk-filter` (129 `apply_native` methods) and `ritk-morphology`, `ritk-segmentation`, `ritk-statistics`.
- [ ] Increment 4: `ritk-registration`, `ritk-model`.
- [ ] Increment 5: consumers — `ritk-io`, `ritk-cli`, `ritk-python`, `ritk-snap`, examples, book samples, `.pyi` stubs.
- [ ] Final: `grep -rn 'native' --include='*.rs' crates` shows no identifier carrying the marker; CHANGELOG records the `[major]` mapping.

### RITK-GAP-2026-08-20-02 — fuzz the parsers

- [ ] Add a non-published `fuzz/` workspace member; confirm it stays out of every published crate's dependency graph.
- [ ] Seed corpora from existing `test_data/` headers: whole, truncated at each field boundary, and bit-flipped.
- [ ] Targets in dependency order: `ritk-trx`, `ritk-tck`, `ritk-trk`, `ritk-mif` (smallest surface, and the four crates holding the production `unwrap()` sites) — then `ritk-nifti`, `ritk-nrrd`, `ritk-metaimage`, `ritk-analyze`, `ritk-mgh`, `ritk-minc` — then `ritk-tiff`, `ritk-png`, `ritk-jpeg`, `ritk-codecs` — then `ritk-vtk` and `ritk-dicom`.
- [ ] Every panic or unbounded allocation found becomes a typed error plus a regression test carrying the offending bytes. Never widen a bound to make a finding go away.
- [ ] Convert the 39 proven-invariant `unwrap()` sites to `expect("invariant: ...")` so the proof ships at the panic site.
- [ ] Wire a scheduled CI job with a committed per-target time budget.

### RITK-GAP-2026-08-20-03 — GPU naming and accelerator claims

- [ ] Decide and record: wire a real accelerator `ComputeBackend` upstream in Coeus, or retire the device vocabulary here. Draft the ADR with the recommendation; do not pose it as a question.
- [ ] If retiring: rename `GpuFieldSmoother` and `CpuOrGpu` for what they do (pre-allocated staging versus in-place), update all callers.
- [ ] Delete or replace the three unbacked performance paragraphs (`smooth.rs:281-285`, `atlas/mod.rs:130-131`, `lddmm/geodesic.rs:137`). A retained sentence cites a stored criterion baseline.
- [ ] Confirm `README.md:19-21` and the type names agree afterwards.

### RITK-GAP-2026-08-20-04 — evict the tracked payload

      `scratch/check_restart.rs`; `scratch/` is now free of tracked artifacts.
      the four stale `target_*` entries.
      the measured groups, duplicate copies, and unresolved corpus provenance
      in `gap_audit.md` F5.
      canonical `ants_example/` and `openneuro/` paths, migrate all consumers,
      and remove the superseded files; the canonical pair is now the only
      source used by `xtask`, Python registration tests, and NIfTI source tests.
- [ ] Decide retention or externalisation for the unreferenced `paired_mri_ct/` corpus after provenance and authorization are established.
- [ ] Split referenced public datasets into small committed goldens and an on-demand checksummed set without removing files consumed by tests.
- [ ] Move the on-demand set behind the existing `externals/` fetch harness; re-point every consuming test and confirm each still resolves its input.
- [ ] Record the committed-fixture budget so the next addition is measured against it.

### RITK-GAP-2026-08-20-05 — nextest budgets

- [ ] Delete the two dead filters (`test(bspline_cr)`, `test(multires_cr)`) and the stale "NdArray CPU time" comment.
- [ ] Profile the six escalated groups. For each, decide: optimise the production code until it fits 30 s / 60 s, or move it to a dedicated profile with a derived, recorded budget.
- [ ] Remove every override above the standard budget from `profile.default` and `profile.ci`.
- [ ] Verify every remaining filter expression matches at least one test.

### RITK-GAP-2026-08-20-06 — lint and doc floor (sequence after 01)

- [ ] Add `[workspace.lints]` and `lints.workspace = true` per member.
- [ ] Record the per-crate `missing_docs` debt as a non-increasing baseline.
- [ ] Add `#![deny(missing_docs)]` crate by crate, burning the baseline down.
- [ ] Write `README.md` for the 15 publishable crates lacking one.

### RITK-GAP-2026-08-20-07 — CHANGELOG version axis

- [ ] Map each of the 167 `[Unreleased]` blocks to its landing version from `git log` and the per-crate manifest history.
- [ ] Fold, collapse completed entries to one line plus a commit link, leave exactly one open `[Unreleased]`.

### RITK-GAP-2026-08-20-08 — book chapters (sequence after 01)

- [ ] For each of the twelve thin chapters: write the promised content, or delete the promise. No placeholder prose.
- [ ] Registration chapters get the MI expression, the gradient-descent update rule, and the convergence criterion, with resolved citations.
- [ ] Add `mdbook test` to the Pages workflow.

### RITK-GAP-2026-08-20-09 — MI subsample stride

- [ ] Derive the sample count from bin occupancy versus histogram variance, or make it a caller parameter with a documented default.
- [ ] Add a test showing MI is stable across volume sizes straddling the threshold.
- [ ] Refresh the stale Correlation-Ratio line in `crates/ritk-registration/docs/REGISTRATION_OPTIMIZATION_ANALYSIS.md:12`.
