# RITK execution backlog

Unresolved delivery items are kept as executable records. Closed history is indexed below; full prose remains in git.

<a id="RITK-METIS-LOCK-008"></a>
## RITK-METIS-LOCK-008 — Advance the WebGPU recovery provider pins [patch]
- Status: review; priority: P1; owner: RITK viewer + integration; integrator: root; last-update: 2026-09-18.
- Scope: advance the six Metis package sources to merged Metis `5e892245ac52c6455bbb57244fa654e6eb3cc9c1` and the fifteen Moirai package sources to `ae282117fd962f4b7c66d722aad9d3c2906320bb`, then replay the real MRI surface and the hosted browser workflow.
- Acceptance: standalone lock resolves with 61 first-party sources; locked native/WASM checks, strict Clippy, format and provenance gates pass; the 1280×800 real MRI PNG remains byte-identical; browser workflow pins the exact Metis revision and records WebGPU capability/recovery evidence without a raster fallback.
- Dependency: Metis PR [#267](https://github.com/ryancinsight/metis/pull/267) merged with the hosted Windows gate; Moirai main includes PR [#401](https://github.com/ryancinsight/Moirai/pull/401); DICOM parsing and clinical presentation remain RITK-owned.
- Delivery: lock commit `c1d4eed49`; six Metis sources resolve at `5e892245ac52c6455bbb57244fa654e6eb3cc9c1`; fifteen Moirai sources resolve at `ae282117fd962f4b7c66d722aad9d3c2906320bb`; standalone lock SHA-256 `a602ec28d9fb3d2c73b6245067ec33e4648d66434de07087b0153c90aea12fbb`.
- Evidence: native locked nextest 449/449 (lib) and 487/487 (package), native/WASM check and strict Clippy, format, lockfile and metadata checks pass; browser trace validation now accepts complete window-level and interaction attribute groups and rejects partial or unknown extensions; the real 94-file MRI replay exits 0, invalid-study probe exits 1, and produces the byte-identical 1280×800 PNG SHA-256 `259dd79103482756c4e688621bebafc841cc40f1df10ff2bbd7f9d04b7b4d401` with 411,589 non-black pixels; executable SHA-256 `3eb1d7b7b181b9a1600d17e38be35fff7b5ba93b6c8492a17df2a3ea3a4db7bd` (26,361,344 bytes). Hosted browser workflow evidence remains open; WebKit bounded Blob reads and Chromium WebGPU capability are residuals until the new dispatch.

<a id="RITK-METIS-LOCK-007"></a>
## RITK-METIS-LOCK-007 — Advance the current Metis text provider pin [patch]
- Status: done; priority: P1; owner: RITK viewer + integration; integrator: root; last-update: 2026-09-18.
- Scope: advance the six Metis package sources from `8f33126f23b8dc327bea45c4fe513a6b60b73c99` to merged Metis `0e449856ade677860fd6866c3855ae2ba527e33a`, then replay the real MRI surface and rebind the standalone provenance.
- Acceptance: standalone `Cargo.lock` resolves with 61 first-party sources; locked `ritk-snap` tests, native/WASM checks and strict Clippy pass; the 1280×800 real MRI PNG remains byte-identical; docs name the exact lock digest and provider revisions.
- Dependency: Metis PR #258 merged with hosted Windows/artifact gates; DICOM parsing and clinical presentation remain RITK-owned.
- Delivery: lock commit `e635baf90f99cc0ae09df06c42424ba5c4c6faec`; six Metis sources resolve at `0e449856ade677860fd6866c3855ae2ba527e33a`; standalone lock SHA-256 `49381c64b5751b5c07bf571c66a31205ebf3ccdc780afe3b4b102c0792a5bc85`.
- Evidence: rebuilt `ritk-snap` exits 0 on the 94-file public MRI-DIR study and the invalid-study probe exits 1; PNG SHA `259dd79103482756c4e688621bebafc841cc40f1df10ff2bbd7f9d04b7b4d401`, 1280×800, 411,589 non-black pixels; executable SHA `210a6a0e0f6efa59d76ad91de8a6d068d12b9e7796df17c00d3be270e829208c` (25,973,760 bytes). Locked native nextest exits 0 for 449 tests; strict native Clippy, WASM check/Clippy, format, lockfile and provenance JSON checks pass.

<a id="RITK-BROWSER-READ-001"></a>
## RITK-BROWSER-READ-001 — Reproduce and close WebKit study reads
- Status: blocked; compacted 2026-09-18; full delivery history remains in git.
- Scope: saved public MRI-DIR chooser replay and cross-engine gallery; provider/host fixes stay with their owners.
- Acceptance: read-path cause and production fix, 94 file hashes, three exact pixel oracles, bounded rejections and clean sessions on Chromium, Firefox and WebKit.
- Risk: [patch]; dependency: [Metis read diagnosis](../metis/backlog.md#METIS-BROWSER-READ-001).
- Delivery: RITK PR [#422](https://github.com/ryancinsight/ritk/pull/422), merge `18ee4e55b`; the current failure evidence is recorded and the external WebKit authorization residual remains blocked.
- Evidence: current hosted [run 35133971196](https://github.com/ryancinsight/ritk/actions/runs/35133971196) builds RITK `db390b8616e4cc58f2555f49a816e61eed1fadd1` against Metis `88c60a0b6410c0e07700e965bcdbea43b7b20789`; Chromium/Firefox pass 94 file hashes, three exact pixel oracles, cine-rate/repeat checks, picker r…
- Diagnosis: WebKit sandbox denies reads and read-extension issuance on the selected real file despite verified host bytes; four browser APIs and isolated one-file/full-batch inputs fail. DICOM stays in RITK.
- Verification: locked `ritk-snap` nextest 841/841, strict native/WASM Clippy and checks, formatting and lockfile validation pass; [proof and log hashes](docs/manual/images/dicom-metis-real-browser-mri-cross-engine.json).
- Blocker: exact SafariDriver/WebKit selected-file authorization defect remains external; current run 35133971196 reproduces the denial after SafariDriver accepts the chooser request. Re-open when the corrected browser/runner path grants real-file reads; application byte-read APIs cannot grant that access.

<a id="RITK-DOCS-EVIDENCE-SYNC-002"></a>
## RITK-DOCS-EVIDENCE-SYNC-002 — Rebind current real MRI replay provenance [patch]
- Status: done; delivery: RITK PR #493, merge `73b01be1a3fcb6bb755ffdd302cb412d218fcd44`; compacted 2026-09-18.
- Outcome: standalone-lock provenance binds RITK `e635baf90f99cc0ae09df06c42424ba5c4c6faec`, Metis `0e449856ade677860fd6866c3855ae2ba527e33a`, Moirai `a2f21496d1d09b2abe6523e3c8cdbf751dcd560a` and lock `49381c64b5751b5c07bf571c66a31205ebf3ccdc780afe3b4b102c0792a5bc85`; the 94-file MRI replay remains byte-identical at `259dd79103482756c4e688621bebafc841cc40f1df10ff2bbd7f9d04b7b4d401` (1280×800, 411,589 non-black pixels), with JSON/image/hash and locked gates validated.

<a id="RITK-BROWSER-WEBGPU-001"></a>
## RITK-BROWSER-WEBGPU-001 — Demonstrate browser WebGPU presentation [arch] [minor]
- Status: blocked; compacted 2026-09-18; full delivery history remains in git.
- Scope: replay the saved public MRI-DIR study through the RITK-owned `?renderer=webgpu` page and retain actual canvas/window evidence; RITK owns DICOM decoding and clinical pixels, while Métis remains the format-neutral canvas host.
- Acceptance: a configured browser runner reports an adapter, presents the three saved-study canvases, records revision-bound PNGs and semantic attributes, and completes bounded teardown without a raster fallback.
- Blocker: hosted Chromium in run [35133971196](https://github.com/ryancinsight/ritk/actions/runs/35133971196) reports no WebGPU adapter; the existing setup error and failure capture are preserved in [`dicom-metis-real-browser-mri-webgpu-failure.png`](docs/manual/images/dicom-metis-real-browser-mri-webgpu-failure.png)…

## Atlas Batch #3 sub-batches (ritk Burn-trait rebind — 6 atomic commits per `atlas/docs/adr/0012-ritk-burn-trait-rebind.md`)

- Status: in-progress; sub-batches #3, #5 and #6 remain reserved per ADR 0012.

- Scope: Burn-trait rebind and per-crate migration are dependency-ordered; each increment must preserve locked tests and the xtask allowlist.


<a id="MIG-439-03"></a>

## MIG-439-03 — Replace remaining Burn NdArray backend aliases with Atlas-backed surfaces

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Atlas-backed surfaces. RESCOPED (was READY; original acceptance criteria do not hold — see Sprint 465 finding).** **Correction (Sprint 465, evidence-based)**: investigated the strongest candidate crate (`ritk-jpeg`, smallest burn_ndarray footprint, already has a parallel Coeus reader) to execute th…

<a id="PERF-435-01"></a>

## PERF-435-01 — Route MSE through fused interpolation. PARTIAL.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Generalized `ritk_interpolation::transform_and_interpolate` over spatial dimensionality, generalized the OOB mask helper over the image shape length, and routed `MeanSquaredError` through the fused transform-to-index-to-linear interpolation path. Evidence tier: value-semantic nextest and focused ti…

<a id="PROVIDER-420-01"></a>

## PROVIDER-420-01 — Hermes complex dispatch bound cleanup. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: The local Atlas provider graph exposed that Hermes complex SIMD operations still require `Neg` at the complex-operation dispatch surface after broader unsigned scalar support. A minimal local fix passes `cargo check -p hermes-simd --all-targets`; full provider rustfmt is still blocked by unrelated…

<a id="PERF-419-01"></a>

## PERF-419-01 — Registration test runtime budget breach. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 419's `ritk-registration` nextest gate passed but exposed integration tests above the 30s slow budget, including 100s, 146s, and 193s rows. Treat this as a real performance defect to profile; do not weaken or skip those tests.

<a id="COEUS-406-01"></a>

## COEUS-406-01 — Fix dirty Coeus autograd provider compile break. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: RITK doctest/doc gates against the current local Atlas stack are blocked after refreshing Coeus path packages to `0.2.6`: `D:\atlas\repos\coeus` is dirty on `test/cuda-parity-suite`, and `coeus-autograd` fails to compile in shape/reduction ops. This must be fixed in Coeus before RITK can claim docs…

<a id="PERF-406-02"></a>

## PERF-406-02 — Registration test runtime budget breach. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 406's touched-package `nextest` gate passed but exposed registration tests above the 30s slow budget, including 93s, 129s, and 183s rows. Treat this as a real performance defect to profile; do not weaken or skip those tests.

<a id="PERF-387-02"></a>

## PERF-387-02 — Continue flat-buffer memory-efficiency audit. IN PROGRESS.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 387 flattened `VectorConfidenceConnected` covariance/inverse matrices and removed the B-spline legacy placeholder. Sprint 389 flattened `InverseDisplacementField` TPS spline/affine coefficient blocks after the solve. Sprint 390 flattened TIFF grayscale/RGB page accumulation by removing `Vec<…

<a id="MIG-387-01"></a>

## MIG-387-01 — Atlas crate migration audit.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Continue replacing production `nalgebra`/`ndarray`/`burn` surfaces with `leto`/`coeus`/ `hephaestus` only after each target operation has a verified equivalent contract and focused differential tests. Do not remove boundary dependencies used only for file-format interop or external framework contra…

<a id="MIG-387-02"></a>

## MIG-387-02 — Spatial Leto SSOT migration. IN PROGRESS.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 408 migrates `ritk-spatial` storage to Leto fixed vectors/matrices and removes direct `nalgebra` dependencies from `ritk-core`, `ritk-metaimage`, `ritk-nrrd`, `ritk-nifti`, and `ritk-mgh` spatial direction setup. Sprint 409 moves DICOM IO, MINC, and filter spatial-transform consumers onto `D…

## Sprint 377 — Performance Review, Memory Efficiency & Carry-Forward Reconciliation

- Status: in-progress; historical sprint retained only for unresolved rows.

- | FMT-377-01 | `cargo fmt --check` clean (staged files); 22 working-tree diffs from cumulative agent updates remain pending rewash | Pending |

- | DOC-377-01 | 16 intra-doc-link warnings accumulated from Sprint 393-395 commits; non-blocking | Pending |

- | PERF-377-01 | **MedianFilter O(N·n³·log n) → O(N·r²)** via Huang's sliding column histogram — bit-exact equivalence to naive reference on every radius | Next |

- | PERF-377-02 | **BilateralFilter memory-bandwidth review** — current LUT/SIMD-friendly; headroom: drop `exp` into a second LUT, separable approximation | Deferred (depends on benchmark) |

- | PERF-377-03 | **Rank/Percentile filter** — same naive O(N·n³·log n) pattern as median; bundle if algorithm portable | Deferred |



## Sprint 372 — J2K conformance fixes + differential interop harness (in progress)

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: Conformance fixes delivered; interop acceptance gate pending

- | J2K-372-HARNESS | Differential interop suite vs openjp2 (pure-Rust c2rust port, dev-dep): encode+decode both directions, marker/packet dump diagnostic | **Done (gate pending)** |

- ### Open defects (P1)



## Sprint 362 — Architecture Hardening: SSOT · DRY · SRP · DIP · Naming

- Status: in-progress; historical sprint retained only for unresolved rows.

- | SSOT-362-02 | `ritk-io::ImageFormat` enum + `from_path` resolver; replace CLI `infer_format` (20L) and Python `io/mod.rs` if-chains (27L) [minor] | Planned |

- | DRY-362-03 | Remove `FftDir` shim from `filter/fft/convolution/helpers.rs`; update all call sites to `ForwardFft`/`InverseFft` ZSTs [patch] | Planned |

- | DRY-362-04 | `UnaryImageFilter<Op>` + `UnaryPixelOp` sealed trait; collapse abs/sqrt/exp/log/square ~570L → ~100L; type aliases preserve public names; `D=3` → `const D` [minor] | Planned |

- | DRY-362-06 | Complete `SamplingConfig` migration: `MutualInformation.sampling_percentage: Option<f32>` + `CorrelationRatio` + `compute_image/mod.rs` [patch] | Planned |

- | DRY-362-08 | `SharedCache<T>` newtype in `metric/cache_slot.rs`; adopt in Parzen (×3) + MutualInformation [patch] | Planned |

- | SRP-362-09 | `bspline_ffd/basis.rs` (445L) → `basis/{scalar,cache,evaluate}.rs` [patch] | Planned |

- | SRP-362-10 | `dl_registration_loss.rs` → `dl/losses/{lncc,grad,combined,mod}.rs` (6 concerns separated) [patch] | Planned |

- | SRP-362-11 | `regularization/trait_::utils` → `regularization/spatial_ops.rs`; make `pub(crate)` [patch] | Planned |

- | PRIM-362-12 | `EarlyStoppingPolicy::Enabled { patience, min_improvement }`: bundle orphaned fields; eliminate impossible `Disabled + non-zero patience` state [minor] | Planned |

- | DIP-362-13 | `Registration::with_config` DIP: `RegistrationCallbackSet` builder owns callback construction; engine receives set [minor] | Planned |

- | DRY-362-14 | `HistogramThreshold` sealed trait; blanket `compute<B,D>` + `apply<B,D>` collapses ~150L scaffold from 6 threshold structs [minor] | Planned |

- | DRY-362-15 | `smooth_or_borrow(data, dims, sigma) -> Cow<[f64]>` in `level_set/helpers.rs`; 3× Cow conditional collapsed [patch] | Planned |

- | PRIM-362-16 | `Connectivity { Six, TwentySix }` enum in `ConnectedComponentsFilter`; eliminate `assert!` on u32 [patch] | Planned |

- | SRP-362-17 | `UnionFind` extracted from `labeling/mod.rs` → `labeling/union_find.rs` [patch] | Planned |

- | SRP-362-18 | `dicom/seg/tests/convert.rs` (554L) → 4 test modules [patch] | Planned |

- | SRP-362-19 | `dicom/series.rs` → `series/{types,scan,loader}.rs`; `Arc<Mutex>` scan → collect-and-merge [patch] | Planned |

- | SRP-362-20 | `FilterArgs` (46 fields) → `FilterKind` ValueEnum + `#[command(flatten)]` per-family structs; `SegmentArgs` (32 fields) same [major] | Planned |

- | DRY-362-21 | `Backend` alias: `commands/viewer.rs` → `use super::Backend` [patch] | Planned |

- | DRY-362-22 | `scales: String`, `cpr_points: Vec<String>` deferred parsing → `value_delimiter` typed Clap fields [patch] | Planned |

- | NAMING-362-23 | `transform_1d/_2d/_3d/_4d` in `bspline/interpolation/` → `transform_points_impl` dispatching on `D` [patch] | Planned |

- | NAMING-362-24 | `spatial_gradient_2d/_3d`, `spatial_laplacian_2d/_3d` → `deformable_field_ops/`; surface only through `dispatch.rs` [patch] | Planned |

- | PRIM-362-25 | `IntensityRange { min, max }` validating newtype; `MinMaxNormalizer.target_{min,max}` + `ZScore` adopt it [minor] | Planned |

- | PRIM-362-26 | `// PRECISION:` justification comment in `normalize.rs` f64 accumulator path [patch] | Planned |

- | PRIM-362-27 | `DicomSeriesInfo`: `ArrayString<64>` public fields → `&str` accessor; `arrayvec` leaves public API surface [minor] | Planned |



## Sprint 342 (Phase 20) — Coeus Migration Readiness Audit

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: In Progress

- | MIG-342-04 | RITK-owned tensor contract over Coeus CPU backend | **Open** |

- | GPU-342-05 | Coeus WGPU differential test harness for RITK operation subset | **Open** |

- | REG-342-06 | Registration autodiff tape continuity proof/test under Coeus | **Open** |

- | MODEL-342-07 | `ritk-model` Coeus module/parameter/3-D convolution migration design | **Open** |

- | PY-342-08 | Python binding conversion plan over Coeus-backed Rust core | **Open** |

- The next implementation stage is not a dependency swap. It is the RITK tensor

- ### Residual risks



## Sprint 332 (0.50.95) — Documentation Compaction + Structural Audit + Benchmark

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: In Progress

- | BENCH-332-03 | `STACK_WEIGHTS_CAPACITY=32` Criterion benchmark — measure AVX2 speedup vs 8-entry version | **Open** |

- | GPU-332-04 | Evaluate `sparse.rs` GPU-backend potential (Burn autodiff scatter compatibility, custom kernel feasibility) | **Open** |

- | CRLF-332-05 | Git CRLF normalization (`git add --renormalize`) — blocked by missing test data files | **Blocked** |



<a id="RITK-GAP-2026-08-20-01"></a>
## RITK-GAP-2026-08-20-01 [major][arch] — collapse the dual `X` / `X_native` surface
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-02"></a>
## RITK-GAP-2026-08-20-02 [minor] — fuzz the sixteen format parsers
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-03"></a>
## RITK-GAP-2026-08-20-03 [patch] — retire the GPU naming and the unbacked accelerator claims
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-04"></a>
## RITK-GAP-2026-08-20-04 [patch] — evict the 1.6 GB tracked binary payload
- Status: in-progress; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-05"></a>
## RITK-GAP-2026-08-20-05 [patch] — derive the escalated test budgets and sweep the dead filters
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-06"></a>
## RITK-GAP-2026-08-20-06 [patch] — raise the lint and documentation floor
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-07"></a>
## RITK-GAP-2026-08-20-07 [patch] — restore the CHANGELOG version axis
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-08"></a>
## RITK-GAP-2026-08-20-08 [patch] — write the registration and dispatch book chapters
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-09"></a>
## RITK-GAP-2026-08-20-09 [patch] — derive or remove the MI subsample stride
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

## Archive — closed items

Closed items are indexed by ID with delivery SHAs and dates; consult git history for their full acceptance records.

- **RITK-BROWSER-ASPECT-001** — Preserve physical slice proportions (2026-09-16) `e6be3d53621a9643331c3259d3d52654cc101072`
- **RITK-BROWSER-SLIDER-001** — Select browser slices directly (2026-09-16) `bf54c6cf6` `abc83b485`
- **RITK-BROWSER-LOCAL-BOX-001** — Map custom embedded canvas boxes (2026-09-16) `c8a323c56746e9bbbb20e52dd48cb9b101b842eb` `c08749927` `8ca04d5`
- **RITK-BROWSER-GALLERY-001** — Own the DICOM browser consumer page [arch] [patch] (2026-09-16) `35133971196`
- **RITK-BROWSER-GALLERY-HARNESS-001** — Own the browser slice-control harness [arch] [patch] (2026-09-16) `db390b8616e4cc58f2555f49a816e61eed1fadd1` `88c60a0b` `35133971196`
- **RITK-BROWSER-CINE-001** — Expose browser cine controls [minor] (2026-09-18) `3695ea220cdd8a630c604b1c6ccb67fb12f61d35` `35294499612`
- **RITK-BROWSER-TOOLS-001** — Expose browser diagnostic tools [minor] (2026-09-18) `1050b7730d85b4b047062e4053817a6d46f962cd` `35298831502`
- **RITK-GALLERY-CYCLES-001** — Repeated saved-study browser lifecycle (2026-09-16) `86632ab0afc667f83e746ab5c8ccb8ce8ca8a32b`
- **RITK-SNAP-BROWSER-TRUST-2026-09-15** — Reject synthetic canvas input [arch] [minor] (2026-09-15) `96bda4f9d3b38f4071ba31946f7e8f131da30f8b` `fb8b0a0b8169a580eb80028d0bbd519cf0057473` `a1e83a15`
- **RITK-SNAP-FILE-TARGETS-2026-09-10** — Four ritk-snap files past the 500-line target [patch] (2026-09-10) `58d42b87b` `7f9b0ce0c` `00ad74875`
- **RITK-VALUE-ASSERTIONS-2026-09-08** — Tests that cannot fail on the defect they name [patch] (2026-09-08)
- **RITK-SNAP-FIXTURES-001** — Required synthetic study workflows [patch] `c38ca276`
- **RITK-SNAP-OPEN-001** — Selected DICOM input and series identity [major] [arch] `4a9f6eb1`
- **RITK-SNAP-DICOM-SUBSTRATE-001** — Keep DICOM opening in RITK [arch] [minor]
- **RITK-SNAP-RESOURCES-001** — Confined and bounded study ingestion [arch] [minor] (2026-09-09) `83a562b1e`
- **RITK-SNAP-DIRECTORY-001** — Validate media-directory record semantics [patch] `0a8367ab`
- **RITK-SNAP-ASPECT-001** — Preserve physical image aspect ratios [patch]
- **RITK-SNAP-FRAMES-001** RITK-SNAP-FRAMES-001 `26074f364`
- **RITK-SNAP-FUSION-001** — Compare volumes in patient coordinates [patch] `e5815378a`
- **RITK-SNAP-COORDINATES-001** — Preserve transformed measurement coordinates [patch] `f60057b47`
- **RITK-SNAP-COLOR-001** — Preserve decoded DICOM color in display [patch] `ce6b60429`
- **RITK-SNAP-GRAYSCALE-001** — DICOM grayscale presentation semantics [patch] `e52a4d28f`
- **RITK-SNAP-REAL-DICOM-DOCS-001** — Lead with real image evidence [patch] `3c0ea6a00`
- **RITK-SNAP-REAL-DICOM-REPLAY-001** — Rebind the real MRI replay [patch] (2026-09-17) `e33d8268a` `f1d7d458` `79f20e2c` `a0f4fd29`
- **RITK-SNAP-EFRAME-MRI-001** — Record a real eframe MRI baseline [patch] `b4d30980f`
- **RITK-SNAP-EVIDENCE-SURFACES-001** — Identify measured presentation surfaces [patch] `261307f65`
- **RITK-SNAP-EFRAME-SURFACE-001** — Capture a matched eframe orthogonal surface [minor] `0fd885adc6124468aa745a2f548dd685b74f0774`
- **RITK-SNAP-EFRAME-MATCH-001** — Match the Métis capture extent [minor] `102909c8f67bca1128eba1438ec280ee2651656e`
- **RITK-SNAP-METIS-001** — Migrate the viewer shell to Métis [arch] [major] (2026-09-17) `57e11c92c080dc9c8716f9218d55c1344a75b412` `5a68a8e725280a3a94387dd4c9f01524201fdb8f`
- **RITK-METIS-LOCK-001** — Advance first-party provider pins [patch] (2026-09-17) `1ab0a3685c58b8d0d6131d9ad5556729f7a838ac` `2358e3003ab625a058be042b6e0f6d562bab7ac9` `b94f3ed7a0faa436ebe993dbfec49726cef853fa` `35232317634`
- **RITK-METIS-LOCK-002** — Advance the post-merge viewer pins [patch] (2026-09-17) `132251fa57046241e55cb9189126d6ae7fb98eb9` `8e566af9a37dc0382e8e919c593d3838f5b08186` `a2f21496d1d09b2abe6523e3c8cdbf751dcd560a`
- **RITK-METIS-LOCK-003** — Advance the current Metis viewer pin [patch] (2026-09-17) `5166dfad833db2d7b2ee3d63a07a52ad66416c3a` `47193af81491d3b2e5e236f8fe69dba85e42ded4` `a2f21496d1d09b2abe6523e3c8cdbf751dcd560a`
- **RITK-METIS-LOCK-004** — Advance the landed Metis asset pin [patch] `b2535b139d4f0df132e0ffd157efe89b445a13e2` `619832d15ac18a188ce6f79a0d810fcab7315891` `35322831603`
- **RITK-METIS-LOCK-005** — Advance the post-merge Metis pin [patch] `e634fe925d8edb18cea4399d43255fdaa2d444a5`
- **RITK-METIS-LOCK-006** — Advance the current public Metis pin [patch] `3ac246728d34a06d9998c21951dc76ad961c8efb` `a563f4517d2ef74f6b50707687f7f28d341ac111`
- **RITK-DOCS-EVIDENCE-SYNC-001** — Sync current provider revisions in replay docs [patch] (2026-09-17) `198fc38bfa1d144f8c95fe66127a96d65c562172` `f971773edb79f987f0a977971e80366ad5001f39` `2358e3003ab625a058be042b6e0f6d562bab7ac9` `b94f3ed7a0faa436ebe993dbfec49726cef853fa`
- **RITK-CLIPPY-RENDER-CFG-001** — Scope the native MIP export to Windows [patch] (2026-09-17) `92a3c06be03727a35a0b91bee2f87100d500c7fa` `35235536323`
- **RITK-PYTHON-DENOISE-ULP-001** — Reproduce hosted denoising parity variance [patch] (2026-09-18) `b2535b139d4f0df132e0ffd157efe89b445a13e2` `35322829144` `35322831603`
- **RITK-CI-DICOM-WORKFLOW-001** — Build the viewer workflow example with its shell [patch] (2026-09-17) `daef9acfc32d517dcfda8aef7183b7f2807f1c6c` `ee0ded9e4` `35242533864` `35242532842`
- **RITK-SNAP-EFRAME-BOUNDARY-001** — Separate the eframe compatibility shell [arch] [major] (2026-09-17) `57e11c92c080dc9c8716f9218d55c1344a75b412`
- **RITK-SNAP-METIS-MIP-001** — Match the native Métis projection layout [arch] [minor] (2026-09-12) `de7b8d8ed0f754020ddb6a7607f24df3e6d652b8`
- **RITK-SNAP-WASM-TOPOLOGY-001** — Scope native shell modules out of the WASM library [arch] [major] (2026-09-13) `d4c928c56`
- **RITK-SNAP-METIS-002** — Validate RITK browser semantic traces [arch] [minor] (2026-09-11) `d97e6523934a9485122d4dbb4afa1eacdf826353`
- **RITK-SNAP-PRESENTATION-CFG-001** — Keep native presentation helpers platform-scoped [patch] (2026-09-12) `030772033a2daa40657db1be093730a513c36c80`
- **RITK-PYTHON-PACKAGING-001** — Publishable Python metadata and tokenless release [patch] (2026-09-12) `8071c7d963095d80258832161a78b8809bb756f7`
- **RITK-PYTHON-DICOM-001** — Select DICOM series from Python [minor] (2026-09-12) `d36263467b106fe7d73fe90b374adae7524d8267`
- **RITK-SOFT-TISSUE-REGISTRATION-2026-09-03** — Soft-tissue multimodal registration [major] [arch] (2026-09-04)
- **RITK-MIND-REGISTRATION-2026-09-04** — Modality-independent neighborhood metric [minor] [arch] (2026-09-04) `47a5d8b05901a2afa77bb558abc86ac675c5e223`
- **RITK-STRUCTURAL-RADIUS-2026-09-04** — Bounded structural-refinement radius [minor] (2026-09-04)
- **RITK-RIGID-CAPTURE-INITIALIZER-2026-09-04** — Robust rigid capture initializer [major] [arch] (2026-09-04)
- **RITK-LINT-ALLOW-SITES-2026-08-31** — Remove reintroduced production allowances [patch] (2026-08-31)
- **ATLAS-RITK-BOOK-STAGING-2026-08-20** — Adopt hash-preserving shared book gate [patch] (2026-08-20) `b35c93313c06ea55fffa680a430378dda1df8e41` `20c93980` `35174713023`
- **RITK-GPU-SMOOTHER-REACH** RITK-GPU-SMOOTHER-REACH [major][arch] — retire the GPU smoother (2026-08-21)
- **ATLAS-RITK-WORKFLOW-PIN-2026-08-20** — Refresh shared book workflow pin [patch] (2026-08-20) `20544b405f68e542364da77492ee7a7ffcc44ae9` `aa48c471ac96eb81869437d84bab439e18d89038` `32344964253` `32344964345`
- **DOC-HUMAN-CONNECTOME** DOC-HUMAN-CONNECTOME [patch] — human tractography and connectomics (2026-09-17) `1a21c14a60736481485027a9949a40c10ae22fb8` `32339275860`
- **FIX-DTI-VOLUME-FRAME** FIX-DTI-VOLUME-FRAME [major][arch] — preserve diffusion coordinate frames (2026-08-20) `14a9c619` `2d159850636a6539db61109533f399d31cc7c6f4` `32387951529` `32387951635`
- **DTI-CONNECTOME-PARCELLATION** DTI-CONNECTOME-PARCELLATION [major] — weighted DTI, connectome measures, atlas parcellation (2026-08-20) `d2ff0b9d` `02d42e0` `d7065b40` `ef7c267c`
- **RITK-ULP-PATCH-DENOISE** RITK-ULP-PATCH-DENOISE [patch] — a 1-ULP parity tolerance that holds only on CI `44535369edbf1a74aeab6873c22c0c3b95910294` `40bbc60ca`
- **BUILD-BLOCK-MATCHING-LOCK** BUILD-BLOCK-MATCHING-LOCK [patch] — restore locked workspace resolution (2026-08-19)
- **DOC-TRACTOGRAPHY-VALIDATION** DOC-TRACTOGRAPHY-VALIDATION [patch] — creation and validation chapter (2026-08-19) `32272992103` `32272991999` `32272992744` `2db8dda8`
- **ATLAS-RITK-ZERO-FLUX-PAD-STRUCTURE** ATLAS-RITK-ZERO-FLUX-PAD-STRUCTURE [patch] — operation-family split (2026-08-19) `805b7216`
- **ATLAS-RITK-RECURSIVE-GAUSSIAN-HESSIAN-STRUCTURE** ATLAS-RITK-RECURSIVE-GAUSSIAN-HESSIAN-STRUCTURE [patch] — operation-family split (2026-08-19) `9034af11`
- **RITK-PARITY-171** — InverseDisplacementField SimpleITK parity broken on main [major] (2026-08-19) `18e5bc7f` `2323646545410156` `0820963382720947` `1707854270935059`
- **DEP-492-01 [patch] - Mnemosyne Eunomia scratch feature propagation. DONE.** RITK's workspace `mnemosyne` dependency enables `eunomia` so the local Apollo FFT dependency graph sees Mnemosyne's `ScratchElement` impls for `eunomia::Complex32/64`. Evidence tier: compile-time validation; `cargo check -p ritk-core` passes.
- **TEST-447-05 [patch] — MINC format-level hostile-fixture regression. DONE.** Commit `eb1a6e3b` uses the native MINC2 writer to forge a 64³ dataset backed by only eight voxels. The format-level regression confirms `read_minc` returns the contextual voxel-data error through `read_bounded_with` without reserving the declared payload. The earlier READY entry was stale relative to the merged test, changelog, and Sprin…
- **SEC-446-05 [patch] — Untrusted-input allocation hardening for the remaining format-parser crates. DONE (Sprint 447).** `ritk-mgh`, `ritk-metaimage`, and `ritk-minc` readers route through the new `ritk-core::io_bounds` SSOT helpers; `ritk-vtk` migrated onto the same module (per-crate copies removed). `ritk-nifti` already validates `volume_byte_range` against the input length and `ritk-nrrd` allocates from real pa…
- **SEC-446-01 [patch] — VTK reader untrusted-input allocation hardening. DONE.** `ritk-vtk` binary/ASCII VTK and PLY readers no longer reserve `count * size` bytes up front from header count fields. SSOT `read_exact_bounded` / `bounded_capacity` helpers cap speculative allocation at 16 MiB/chunk and report truncation; `read_binary_be` checks the length product for overflow. Evidence tier: value-semantic nextest plu…
- **PERF-432-01 [patch] — Registration integration tests exceed the strict nextest budget. SUPERSEDED.** This duplicate historical item is closed by the profiling-backed `PERF-432-01` implementation record above: the focused B-spline row passes in 17.279s and the package passes 740/740. The remaining text is retained as investigation history, not executable work. Acceptance: profile the slow registration integration…
- **MEM-445-01 [patch] — MAD noise work-buffer reuse. DONE.** MAD noise estimation now overwrites its mutable work buffer with absolute deviations after the median is known, avoiding the previous second `Vec<f32>` allocation for deviation sorting. Evidence tier: value-semantic nextest plus compile/lint/docs; `cargo nextest run -p ritk-statistics --features coeus mad` passed 9 tests, including borrowed-slice order-pr…
- **MEM-444-01 [patch] — Histogram matching allocation cleanup. DONE.** `HistogramMatcher::match_histograms` now reuses the extracted source voxel buffer as the transform output after landmark estimation, avoiding a separate output `Vec<f32>`. `quantile_landmarks` now emits landmarks during a single histogram-bin scan instead of allocating a cumulative histogram `Vec<u64>`. Evidence tier: value-semantic nextest plus…
- **MEM-443-01 [patch] — Nyul-Udupa output buffer reuse. DONE.** `NyulUdupaNormalizer::apply` still needs one sorted work buffer because percentile landmarks require sorted intensities while image reconstruction must preserve original voxel order. It now reuses the extracted original-order voxel buffer as the transform output after landmark computation, avoiding a separate output `Vec<f32>` allocation. Evidence tier…
- **MEM-442-01 [patch] — Statistics full-image owned extraction cleanup. DONE.** Routed Burn-backed full-image statistics from `extract_vec_infallible` directly into the owned-buffer statistics core, avoiding a redundant clone of the extracted tensor values before percentile selection. Evidence tier: value-semantic nextest plus compile/lint/docs; `cargo nextest run -p ritk-statistics --features coeus image_statistic…
- **MEM-441-01 [patch] — Statistics masked-buffer allocation cleanup. DONE.** Split the image-statistics core into the existing non-mutating borrowed-slice API and a crate-private owned-buffer path. Burn and Coeus masked statistics now consume the foreground vector directly for in-place percentile selection instead of cloning it before quickselect. Evidence tier: value-semantic nextest plus compile/lint/docs; `cargo…
- **MIG-440-01 [patch] — Coeus image flat-buffer boundary. DONE.** Added `ritk_image::coeus::Image::from_flat_on` and `from_flat` so Coeus image construction from flat buffers validates checked shape products and length mismatches at the image boundary before tensor construction. Routed existing Coeus statistics and registration preprocessing test helpers through the new constructor. Evidence tier: type-level rank e…
- **MIG-439-01 [patch] — I/O direct ndarray and workspace nalgebra cleanup. DONE.** Removed the unused direct `ndarray` dependency from `ritk-io` and removed the stale root workspace `ndarray` and `nalgebra` entries after auditing source and manifests for direct usage. Remaining matches are `burn_ndarray` backend/test aliases or Python `numpy::ndarray` boundary imports, not direct `ndarray`/`nalgebra` crate edges. E…
- **MIG-489 [minor] — De-brand the remaining substrate-named APIs. DONE (all slices; the feature-name slice was resolved by *removing* the feature, per the user's directive).** **Slice 4 (Sprint 492, DONE):** "coeus is not a feature" — the Atlas substrate is the mainline, not an opt-in. Removed the `coeus` cargo feature from all 14 crates: coeus-core/tensor/ops/autograd deps made unconditional, every `#[cfg(feature…
- **MIG-488-01 [major] — Correction: image-generic I/O contract + de-branded ritk-io types (user review). DONE.** User review caught two real defects in the Sprint 486/487 work: substrate brand names baked into component names (`Coeus*` — a violation of the workspace naming rule, and permanent noise given Burn is being completely removed), and a parallel branded trait pair where one generic trait belonged. Corrected…
- **MIG-487-01 [minor] — All seven remaining Coeus reader implementors for the `ritk-io` contract. DONE.** Broadened the Coeus I/O contract's format coverage from 1 to 8 reader implementors: added `CoeusJpegReader`, `CoeusMghReader`, `CoeusMetaImageReader`, `CoeusMincReader`, `CoeusPngReader`, `CoeusPngSeriesReader`, and `CoeusTiffReader` — each a cfg-gated module in its format's own `ritk-io/src/format/<fmt>/mod.rs…
- **MIG-486-01 [minor] — Coeus-typed `ritk-io` I/O contract + first implementors (ADR 0002 cutover step 2). DONE.** Added the parallel Coeus I/O contract to `ritk-io`: `domain::coeus::{CoeusImageReader, CoeusImageWriter}` (new leaf module, `coeus` feature), mirroring the Burn `ImageReader`/`ImageWriter` role interfaces over `ritk_image::coeus::Image<T, B, D>` — generic over the scalar `T` (implementors pin `f32`), b…
- **MIG-485-01 [minor] — First Coeus format writer: `write_nifti_coeus` + shared serialization core. DONE.** The write-side half of the ADR-0002 cutover prerequisite, and the direct validation of MIG-484's `data_cow` extraction. Refactored `ritk-nifti/src/writer.rs` to a substrate-agnostic serialization SSOT (`write_flat_with_version`: header-from-spatial + `[Z,Y,X]` byte stream) and made the Burn `write_nifti` a th…
- **MIG-484-01 [minor] — Coeus `Image` host-extraction parity (ADR 0002 cutover prerequisite, step 1). DONE (extraction gap); RESIDUAL filed.** Gap-audited the Coeus `Image` accessor surface against the actual method calls the `ritk-io` writers/CLI/Python boundary make on the Burn `Image` (grep-enumerated: `shape`/`spacing`/`origin`/`direction` metadata — already present; `with_data_slice`/`data_slice`/`try_data_vec…
- **MIG-483-01 [arch] — Core `Image`/tensor-substrate migration strategy (ADR 0002). DONE (design artifact; Foundation phase).** Audited the full migration surface (`burn-migration-audit` + manifest greps): confirmed `rayon`/`tokio`/`nalgebra`/`ndarray`/`rustfft` are already absent from RITK — Burn is the sole remaining substrate. Established that the 12 sprints of parallel Coeus registration capability (MIG-471…482…
- **MIG-482-01 [minor] — Coeus-native gradient-descent registration driver. DONE.** Addressed the standing risk that the Coeus registration primitives (built and verified individually over MIG-471…481) had no composed, usable end-to-end entry point — a parallel capability nothing could *run*. Added `metric::coeus_autograd::driver::gradient_descent` + `GradientDescentConfig` + `RegistrationOutcome`: the reusable "run…
- **MIG-479-01 [minor] — Consolidate per-axis translation onto the `CoeusTransform` seam. DONE.** Removed both superseded per-axis translation functions — `translation_mse_coeus` (composed metric) and `translate_axis_coeus` (per-axis primitive) — leaving a single authoritative translation path: the `Translation` `CoeusTransform` struct dispatched through the generic `mse_metric` (SSOT). Both had only test callers af…
- **MIG-480-01 [minor] — Coeus-native differentiable NCC loss reduction. DONE.** Added `metric::coeus_autograd::ncc::normalized_cross_correlation_coeus`, the second Coeus-native intensity-metric reduction (after MSE), computing `−NCC(moving, fixed)` via the single-pass algebraic-moments form (Lewis 1995) entirely on the autograd tape (`sum`/`mul`/`sub`/scalar ops/`sqrt`/`div`/ `neg`; `T: Float` bound). Evidence tier…
- **MIG-478-02 [minor] — Coeus-native `CoeusMetric` reduction seam over Mse/Ncc. DONE.** Introduced `traits::CoeusMetric` (`reduce(&self, sampled: &Var[N], fixed: &Var[N]) -> Var[1]`, `T: Float`) — the minimal role interface distinguishing metric types (the shared transform+sample step stays in the generic composition, not in implementors, per interface segregation). Added `Mse` (`mse.rs`) and `Ncc` (`ncc.rs`) ZST i…
- **MIG-478-01 [arch] — Coeus-native `CoeusTransform` trait surface + generic MSE metric. DONE.** Wrote ADR 0001 (`docs/adr/0001-coeus-native-registration-traits.md`) deciding: a **parallel** Coeus-native trait family (not substrate- generalization of the burn-bound `ritk_core` traits, which would be a workspace-wide breaking [major] change), unified on the `[N,3]` coordinate convention at the seam, with **one gener…
- **MIG-477-01 [minor] — End-to-end Coeus-autograd affine-MSE registration metric. DONE.** Composed `affine_transform_coeus` + `sample_trilinear_coeus` + `mean_squared_error_coeus` into `affine_mse_coeus(moving_flat, dims, fixed, grid[N,3], R[3,3], t[3])`, splitting the affine's `[N,3]` output into the three per-axis coordinate `Var`s the sampler consumes via the differentiable `slice` + `reshape` (their scatter bac…
- **MIG-476-01 [minor] — Coeus-autograd differentiable affine coordinate transform. DONE.** Added `transform::affine_transform_coeus(coords[N,3], R[3,3], t[3]) → [N,3]` = `coords·Rᵀ + t` (i.e. `R·point + t` per row), gradient flowing to `R` (via Coeus `matmul` + `transpose_2d`) and `t` (via `broadcast_to`'s summing backward). **Design decision (resolved by checking source):** both `slice` and `index_select` are diff…
- **MIG-475-01 [minor] — Coeus-autograd gradient-descent optimizability of the registration metric. DONE.** Proved the end-to-end Coeus registration objective is not merely differentiable but *usable for optimization*. Added `metric::coeus_autograd::optim::sgd_step_var` — a `Var`-level vanilla gradient-descent step (Coeus provides only a low-level fused `sgd_step` over raw buffers, no `Var`-level helper; this return…
- **MIG-474-01 [minor] — End-to-end Coeus-autograd MSE-over-a-translation registration metric. DONE.** Composed the three verified primitives into the first usable Coeus-native registration metric: `translation_mse_coeus` = `mean_squared_error_coeus(sample_trilinear_coeus(moving, translate(grid, t)), fixed)`, gradient flowing to the per-axis translation parameters. Added the differentiable transform primitive it nee…
- **MIG-473-01 [minor] — Coeus-autograd differentiable trilinear (3-D) image sampling. DONE.** Extended the proven 1-D gather+weight-gradient mechanism to 3-D trilinear: `sample_trilinear_coeus(signal_flat, [Z,Y,X], coords_z, coords_y, coords_x)` gathers all eight corners from the flattened moving-image `Var` (flat index `z·Y·X + y·X + x`, corners clamped per axis independently), weights each by the product of the t…
- **MIG-472-01 [minor] — Coeus-autograd differentiable 1-D linear image sampling. DONE.** Resolved the gather-semantics blocker first (read `coeus-autograd/src/ops/shape/select/gather.rs`): `gather(input, dim, index)` takes the index as a `Var<T,B>` of integer-valued floats and is differentiable through the gathered *values* (`scatter_add` backward) but not the index (piecewise-constant) — exactly the interpolation…
- **MIG-471-01 [minor] — Coeus-autograd differentiable MSE loss kernel. DONE.** First verified increment of the burn→coeus registration-metric autodiff path (`docs/coeus_migration.md` dev-sequence step 6, gate #3). Added `ritk_registration::metric::coeus_autograd::mean_squared_error_coeus` (`coeus` feature): `mean((moving − fixed)²)` built entirely from Coeus autograd `Var` ops (`sub`/`mul`/`mean`), no host extracti…
- **MIG-470-01 [minor] — Coeus-native binary dilation/closing/opening, shared differential-test harness. DONE.** Completed the binary-morphology family's Coeus boundary layer: added `binary_dilate_coeus`, `binary_closing_coeus`, and `binary_opening_coeus` alongside the existing `binary_erode_coeus`, each a thin `map_flat_image` wrapper over its already substrate-agnostic pure core (`dilate_binary_3d`; `erode∘dilate`…
- **MIG-469-01 [patch] — Retract false "Coeus has no autodiff" claim. DONE.** The user directly challenged Sprint 468's assertion that "Coeus does not provide autodiff." Checked rather than defended: `D:/atlas/repos/coeus/ coeus-autograd` is a real crate (`[package] name = "coeus-autograd"`, workspace member, depends on `coeus-core`/`coeus-tensor`/`coeus-ops`/ `coeus-sparse`/`apollo-fft`/`leto-ops`) implementing ful…
- **MIG-468-01 [minor] — Coeus-native binary erosion + shared boundary helper for `ritk-filter`. DONE.** Added `crates/ritk-filter/src/coeus_support.rs::map_flat_image`, a generic extract→compute→reconstruct helper for `ritk_image::coeus::Image` boundaries, and refactored `distance_transform_coeus` (MIG-467-01) to use it — the second occurrence of the identical five-line marshaling sequence (this sprint's `binary_er…
- **MIG-467-01 [minor] — Coeus-native Euclidean distance transform for `ritk-filter`. DONE.** Added a `coeus` feature to `ritk-filter` (it had none — the only compute-heavy crate in that state after MIG-466-01) and `distance_transform_coeus`, a thin `ritk_image::coeus::Image` boundary around the existing pure `euclidean_dt` core (Meijster–Roerdink– Hesselink, `#![forbid(unsafe_code)]`, already substrate-agnostic — n…
- **MIG-466-01 [minor] — Coeus-native trilinear interpolation path for `ritk-interpolation`. DONE.** Added a `coeus` feature to `ritk-interpolation` (workspace pattern already used by `ritk-jpeg`/`ritk-statistics`/`ritk-registration`) and a Coeus-native `trilinear_interpolation_coeus` operating on flat row-major buffers via `coeus_core::Scalar`, mirroring the Burn-generic `trilinear_interpolation` contract exactly —…
- **MIG-437-01 [patch] — CLI MI registration direct ndarray boundary. DONE.** Replaced the `ritk-cli` MI registration image conversion helpers with `leto::Array3<f64>` so the CLI hands Leto volumes directly to the classical registration engine and spatial warp. Removed the direct `ndarray` dependency from `ritk-cli`; the remaining `burn_ndarray::NdArray` alias is a separate CLI backend migration item. Evidence tier:…
- **PROVIDER-437-02 [minor] — Moirai stream module rename completion. DONE.** Completed the `moirai-iter` `parallel_stream` -> `stream` module rename that blocked RITK Coeus rustdoc, and verified the bounded concurrent stream API in Moirai. Evidence tier: compile/lint/docs plus value-semantic nextest; `cargo nextest run -p moirai-iter stream` passed 10 tests, and RITK `cargo doc -p ritk-registration --features coeus…
- **MIG-438-01 [patch] — Registration direct ndarray dependency cleanup. DONE.** Removed the unused direct `ndarray` dependency from `ritk-registration` after auditing production source for direct `ndarray` symbols. The remaining registration matches are `burn_ndarray` test/backend aliases. Updated the classical-engine Rustdoc from stale ndarray wording to the active Leto array substrate. Evidence tier: source audit…
- **TEST-436-01 [patch] — Fused identity-direction coordinate convention. DONE.** Added asymmetric-origin, anisotropic-spacing differential coverage comparing fused interpolation against the unfused transform -> world-to-index -> interpolation path. Evidence tier: value-semantic differential nextest; `cargo nextest run -p ritk-interpolation fused` passed 8/8.
- **PERF-434-01 [patch] — Correct CR registration convergence and expose multires loop config. DONE.** Fixed `ConvergenceChecker` so the current best loss is compared against the previous patience window instead of being included in the best-loss baseline. Added `MultiResolutionRegistration::with_registration_config` and used the corrected convergence policy for B-spline CR and multires CR integration tests. Evidenc…
- **MIG-433-01 [minor] — Coeus preprocessing Gaussian smoothing. DONE.** Route `PreprocessingPipeline::execute_coeus` `Smoothing` through the existing Moirai-backed Gaussian smoothing primitive, extended with per-axis voxel sigmas for spacing-aware images. Coeus extraction/rebuild remains centralized in `ritk_tensor_ops::coeus`; smoothing reuses executor-owned scratch storage and rejects non-finite sigma. Evidence t…
- **MIG-432-01 [minor] — Coeus registration preprocessing scalar consumer. DONE.** Add feature-gated `PreprocessingPipeline::execute_coeus` for scalar-safe preprocessing steps and consolidate scalar value semantics into one `value_ops` implementation shared with the legacy Burn executor. Evidence tier: compile/lint/docs plus value-semantic tests (`cargo nextest run -p ritk-registration --features coeus` -> 661/661 p…
- **MIG-431-01 [minor] — Coeus statistics image consumer. DONE.** Add feature-gated `ritk_statistics::image_statistics::coeus` entry points for Coeus-backed image statistics. The Coeus functions borrow image data through the Sprint 430 `ritk_tensor_ops::coeus` image helpers and reuse the existing slice-level statistics computation SSOT. Evidence tier: compile/lint plus value-semantic parity tests (`cargo nextest run…
- **MIG-430-01 [minor] — Coeus image tensor-ops boundary. DONE.** Add feature-gated `ritk_tensor_ops::coeus` helpers for `ritk_image::coeus::Image<T, B, D>`: borrowed contiguous extraction, owned extraction, and checked rebuild while preserving image metadata. The image helpers delegate to the existing Coeus tensor rank, contiguity, and shape-product validation SSOT. Evidence tier: compile/lint/docs plus value-seman…
- **MIG-429-01 [minor] — Coeus image contract. DONE.** Add a feature-gated `ritk_image::coeus::Image<T, B, D>` backed by `coeus_tensor::Tensor<T, B>`. Construction validates tensor rank against the const image dimensionality; metadata access and `into_parts` preserve ownership; contiguous host borrowing is available only for CPU-addressable Coeus backends and rejects non-contiguous layouts instead of materializing s…
- **MIG-428-01 [minor] — Coeus tensor-ops host boundary. DONE.** Add a feature-gated Coeus-native host-buffer boundary to `ritk-tensor-ops`: borrowed contiguous extraction for zero-copy read-only kernels, owned extraction when mutation/storage is required, and checked tensor rebuild that rejects overflowing or mismatched shape products before allocation. Evidence tier: compile/lint/docs plus value-semantic tests (`c…
- **MIG-427-01 [patch] — Coeus tensor-ops contract tests. DONE.** Consolidate `ritk-tensor-ops` Coeus feature tests so elementwise Coeus/Burn differential coverage runs through one table-driven fixture with explicit expected values. Shape-operation coverage now asserts reshape values and transpose logical indexing instead of shape-only success. Evidence tier: compile/lint/docs plus value-semantic tests (`cargo nexte…
- **MIG-426-01 [patch] — NIfTI fixture provenance and import coverage. DONE.** Add source-backed NIfTI import validation around `ritk-nifti`: the real repository NIfTI-1 gzip fixture (`test_data/ants_example/mni152.nii.gz`) is documented as an ANTs/MNI152 copy and imported in tests; deterministic generated NIfTI-2 gzip fixtures validate the native writer/reader path; and Analyze-style `.hdr` bytes are rejected by th…
- **MIG-425-01 [minor] — Native NIfTI-2 single-file codec. DONE.** Extend `ritk-nifti`'s native codec from NIfTI-1-only single-file support to automatic NIfTI-1/NIfTI-2 reads plus explicit NIfTI-2 image and label writers. The header module is now one versioned SSOT over datatype validation, endian detection, widened NIfTI-2 dimensions/spatial fields, checked payload ranges, and endian-aware payload lane reads. Analy…
- **MIG-424-01 [patch] — Native RITK NIfTI codec. DONE.** Replace `ritk-nifti`'s dependency on `nifti-rs` and direct ndarray conversion/writer handoff with a native NIfTI-1 single-file codec. The new vertical structure owns header parsing/serialization, checked dimensions, sform/qform spatial extraction, Float32 image decoding, Float32/UInt32 label decoding, and streamed `.nii` / `.nii.gz` writing without a full pay…
- **MIG-423-01 [patch] — NIfTI shape bounds SSOT. DONE.** Move NIfTI voxel-count arithmetic into one `ritk-nifti::shape` helper used by reader and writer paths. Label and image writers now validate shape products before constructing ndarray handoff buffers, and adversarial overflowing label shapes fail with a typed error instead of multiplication wraparound or allocation. Evidence tier: compile/lint/docs plus value-…
- **MIG-422-01 [patch] — PACS worker send signal and Tokio drift cleanup. DONE.** Remove the final stale Tokio reference from `ritk-snap` PACS worker docs, correct completed-response backpressure wording, and replace the discarded `SyncSender::send` result with one send-status helper covered by delivered and receiver-dropped value-semantic tests. The RITK source/manifests now have no `rayon`, `tokio`, `ParallelSlice…
- **MIG-421-01 [patch] — Direct Moirai DICOM series loading. DONE.** Replace `ritk-io` DICOM directory scan, series header parse, and pixel decode `ParallelSlice` extension-trait call sites with direct `moirai::map_collect_index_with::<moirai::Adaptive>` calls. This keeps file/slice ordering explicit by index and leaves no `ParallelSlice`, `.par()`, or `map_collect` matches in `crates/ritk-io/src/format/dicom`. Evid…
- **MIG-420-01 [patch] — Direct Moirai filter diffusion enumeration. DONE.** Replace `ritk-filter` Perona-Malik and coherence diffusion `ParallelSliceMut` extension-trait call sites with direct `moirai::enumerate_mut_with::<moirai::Adaptive>` and indexed collection calls. The touched filter source now has no `ParallelSliceMut`, `par_mut`, Rayon, or Tokio matches, and projection docs no longer describe Rayon. Evidenc…
- **MIG-419-01 [patch] — Direct Moirai registration enumeration. DONE.** Replace `ritk-registration` Parzen direct sparse-entry initialization and CMA-ES population fitness writes with direct `moirai::enumerate_mut_with::<moirai::Adaptive>` calls instead of the `ParallelSliceMut` extension trait. The touched registration contexts now have no `ParallelSliceMut`, `par_mut`, or stale Rayon wording. Evidence tier: compi…
- **COEUS-419-01 [patch] — Fix local Coeus provider blockers. DONE.** Repair the dirty local Coeus provider graph required by the RITK registration gate: restore the shape root `flat_to_nd` export for moved shape leaves, restore the real `embedding_backward_with_padding_idx` accumulation path, and restore the autograd reshape contiguous-function import. Evidence tier: compile plus value-semantic provider tests (`car…
- **MIG-418-01 [patch] — Direct Moirai segmentation enumeration. DONE.** Replace the last `ritk-segmentation` `ParallelSliceMut` extension-trait call sites in isolated watershed and STAPLE with direct `moirai::enumerate_mut_with::<moirai::Adaptive>` calls. This keeps the execution policy explicit at the call site and leaves no `ParallelSliceMut`, `par_mut`, `unsafe`, or `SendPtr` matches in `ritk-segmentation/src`.…
- **MIG-417-01 [patch] — Level-set safe Moirai convergence metrics. DONE.** Replace the five level-set raw-pointer `SendPtr` convergence-metric side writes with one shared helper that pairs each mutable z-slice with its metric slot under Moirai dispatch. This removes RITK-local unsafe code from Chan-Vese, geodesic active contour, shape detection, Laplacian, and threshold level-set PDE loops while preserving per-slic…
- **MIG-416-01 [patch] — GrowCut safe Moirai paired assignment. DONE.** Replace `ritk-segmentation` GrowCut's raw-pointer `SendPtr` side-write pattern with Moirai paired mutable chunk dispatch over `next_strengths` and `next_labels`. This removes unsafe code from the GrowCut assignment loop while preserving disjoint per-voxel writes and seed label stability. Evidence tier: compile/lint/docs plus value-semantic tests…
- **MIG-415-01 [patch] — SLIC safe Moirai paired assignment. DONE.** Replace `ritk-segmentation` SLIC assignment's raw-pointer `SendPtr` side-write pattern with Moirai paired mutable chunk dispatch over `distances` and `labels`. This removes unsafe code from the SLIC assignment hot path while preserving disjoint per-voxel writes and the existing SLIC distance contract. Evidence tier: compile/lint/docs plus value-sem…
- **MIG-414-01 [patch] — Gaia MeshBuilder array API migration. DONE.** Extend Gaia's `MeshBuilder` with coordinate-array and explicit xyz insertion APIs, then migrate RITK mesh construction sites to those provider APIs. Target outcome: `ritk-filter`, `ritk-vtk`, and `ritk-io` no longer declare direct `nalgebra` dependencies for Gaia mesh construction. Evidence tier: compile/lint/docs plus value-semantic provider and…
- **MIG-413-01 [patch] — BinShrink direct Moirai output writes. DONE.** Replace `ritk-filter::bin_shrink`'s intermediate `(offset, value)` result staging with direct disjoint output-chunk writes through Moirai. This preserves the row-major bin-average contract while removing an allocation proportional to the output voxel count and avoiding a scatter pass. Evidence tier: compile/lint/docs plus value-semantic tests (`…
- **MIG-412-01 [patch] — Statistics Atlas dependency cleanup. DONE.** Remove `ritk-statistics`' stale direct `nalgebra` dependency and correct Jacobian comments that still described Rayon even though the implementation already uses Moirai adaptive execution helpers. This is a dependency-surface and documentation cleanup only; it does not claim Burn/Coeus tensor migration or ndarray removal. Evidence tier: compile/li…
- **FMT-406-01 [patch] — Restore full-repo rustfmt gate. DONE.** Sprint 406 applies the committed rustfmt style to the formatting drift that blocked `cargo fmt --check` after Sprint 405. This is mechanical hygiene only; no behavior, allocation, or performance change is claimed.
- **MIG-411-01 [patch] — SNAP spatial metadata Leto cleanup. DONE.** Remove `ritk-snap`'s direct `nalgebra` dependency where the crate only needs default and row-major direction construction. Route those sites through `ritk_spatial::Direction` so Leto-backed spatial metadata remains the single RITK-owned API. This does not claim the broader Burn/Coeus, ndarray, or mesh migration complete.
- **SAFE-393-02 [patch] — Continue hostile format-header parser audit. IN PROGRESS.** Sprint 393 hardened NRRD spatial vector parsing so unterminated parenthesized groups return an error instead of accepting a parsed prefix. Sprint 394 hardened NRRD vector fields so trailing non-vector tokens and multiple `space origin` vectors are rejected. Sprint 395 hardened DICOM RT Structure Set `ContourData` so present contour…
- **SAFE-405-01 [patch] — FFT convolution padding bounds. DONE.** Sprint 405 centralizes 2-D/3-D FFT padding and boundary-extension shape arithmetic for `ritk-filter` convolution and normalized cross-correlation. The target is checked `usize` addition/multiplication and power-of-two extent validation before allocation, plus removal of `usize as isize` source-index casts in edge replication.
- **CLIPPY-387-01 [patch] — `ritk-interpolation` linear-kernel slice lint cleanup. DONE.** Focused Clippy was blocked by `clippy::single_range_in_vec_init` in `interpolation/kernel/linear/{dim2,dim3,dim4}.rs`; the kernels now route gathered 1-D corner-batch splits through the shared `linear::slice_batch` helper backed by `Tensor::slice_dim`. Evidence tier: compile/lint and value-semantic focused tests.
- **PERF-379-01 [patch] — Deriche recursive-Gaussian cross-line parallelism. DONE.** `iir::apply_deriche_1d` now parallelises the X/Y passes across Z-slices via `moirai::for_each_chunk_mut` (contiguous `nyx` chunks, one `LineScratch` per slice); the per-line IIR is factored into `deriche_line`. Output **bit-identical** to the serial form (exact array equality; float-exact sitk parity unchanged). Min-of-20 on 128³: s…
- **(unnumbered)** Sprint 376 — DRY Closure, Build Hardening & Carry-Forward Reconciliation `fc9d009e` `91991789`
- **RITK-BROWSER-WINDOW-LEVEL-001** — Expose Rust-owned browser window presets [minor] (2026-09-17) `13c7b40c4dd2342ae8149eda3a095c05c06c433f` `678e7024a0f0d19acecff3dd53cea834d74c9c92` `feb3ef827dbbd8b0e2d4d322a7ef1cbf53d64e32` `bdccdc569b57021613df8a82bc9ae99119eb6146`
