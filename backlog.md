# RITK execution backlog

<a id="RITK-SNAP-OBLIQUE-NATIVE-001"></a>
## RITK-SNAP-OBLIQUE-NATIVE-001: Native oblique MPR
- outcome: deliver a native four-plane oblique viewer with patient-space navigation and measurement.
- acceptance: all child items land; invalid geometry is rejected; the public phantom capture shows one complete, uncropped app window with visible menus or toolbar buttons and all four anatomical panes; native visual and value-semantic gates pass.
- status: todo
- priority: architecture
- needs: RITK-SNAP-INTERACTION-REGIONS-001, RITK-SNAP-INTERACTION-MEASUREMENTS-001, RITK-SNAP-INTERACTION-STATE-001, RITK-SNAP-INTERACTION-WINDOW-LEVEL-001, RITK-SNAP-OBLIQUE-APP-ADAPTER-001, RITK-SNAP-OBLIQUE-APP-TESTS-001, RITK-SNAP-OBLIQUE-SESSION-MODULES-001, RITK-SNAP-OBLIQUE-SESSION-WIRING-001, RITK-SNAP-OBLIQUE-ROUTING-001, RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001, RITK-SNAP-OBLIQUE-SESSION-TESTS-001, RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001, RITK-SNAP-OBLIQUE-MANUAL-001
- scope: `crates/ritk-snap/src/{app,presentation,render,tools/interaction,session}/`, `crates/ritk-snap/src/main.rs`, crate README, ADRs, manual, and provenance.
- next: deliver ready child items in dependency order; preserve the public MRI-DIR phantom as the shareable visual fixture.
- risk: [major] [arch]; patient-space annotations and session format 3; no registry release is authorized.
- basis: 4e2e797c6af199d0c33e37148e784b2becc04981

<a id="RITK-CI-MERGE-GATE-001"></a>
## RITK-CI-MERGE-GATE-001: Gate pull requests through one CI check
- outcome: preserve an accurate required gate across draft and ready pull requests.
- acceptance: one aggregate check covers verification jobs; draft pull requests run no heavy jobs and fail the aggregate; `ready_for_review` runs affected checks; the main ruleset requires only the aggregate; workflow contracts and hosted runs verify Rust, Python, docs, and board-only changes.
- status: todo
- priority: architecture
- needs: none
- scope: `.github/workflows/`, `scripts/tests/`, `docs/adr/`, and the main merge ruleset
- next: claim the item, map the required-check and reusable-workflow graph, then record the single-pipeline design in an indexed ADR before changing workflows.
- basis: bf589f94b9826be8b4b05c85e5da31c338f01bab

<a id="RITK-SNAP-METIS-SHARED-TEXT-001"></a>
## RITK-SNAP-METIS-SHARED-TEXT-001: Build display text from shared strings
- outcome: ritk-snap builds and passes against metis-ui-lang once its `DrawText` text and `ElementRect` ids become `Arc<str>` (ryancinsight/metis#458).
- acceptance: the lock advances metis past d93a456 to the commit landing metis#458; every `DisplayCommand::DrawText` construction in `crates/ritk-snap/src/presentation/native_session/` (`layout/overlay.rs`, `selection.rs`) passes `Arc<str>`, and text comparisons there and in `native_session/tests*` compare `&**text` or `text.as_ref()`; ritk-snap clippy `-D warnings` and nextest pass; native display tests keep their expected text.
- status: blocked
- blocker: metis#458 is not merged; re-open when it merges.
- priority: correctness
- needs: none
- scope: `crates/ritk-snap/src/presentation/native_session/`, `Cargo.lock`
- next: after metis#458 merges, `cargo update -p metis-ui-lang -p metis-platform -p metis-web` and fix the call sites the checker reports.
- basis: d6b9f79f

<a id="RITK-CASTFROM-MIGRATE"></a>
## RITK-CASTFROM-MIGRATE: Retire `CastFrom` from ritk
- outcome: no ritk source uses eunomia `CastFrom`/`CastTo`; each site converts through std or a named eunomia method.
- acceptance: `git grep -c -E '(cast_from|cast_to|CastFrom|CastTo)' -- '*.rs'` is empty; ritk builds against the eunomia that drops `NumericElement: CastFrom<i32>`.
- status: blocked
- blocker: the 15 remaining call sites convert a float to an integer (`u8`, `u32`, `usize`) with rounding or flooring, and eunomia has no method for that conversion; re-open when eunomia lands a named float-to-integer method.
- priority: architecture
- needs: none
- scope: `crates/ritk-filter/examples/`, `crates/ritk-io/examples/`, `crates/ritk-registration/{examples,src/metric/mind}/`, `crates/ritk-interpolation/src/native.rs` (its `usize: CastFrom<T>` bounds).
- next: replace each remaining site with the eunomia method once it exists, then delete the `CastFrom` imports.
- links: consumer slice of [EUNOMIA-CASTFROM-RETIRE](../eunomia/backlog.md#EUNOMIA-CASTFROM-RETIRE); 27 of the 56 original sites converted to `FloatElement::{from_count, from_integer, from_f64}`.

<a id="RITK-FORMAT-BULK-DECODE-001"></a>
## RITK-FORMAT-BULK-DECODE-001: Decode sample buffers through one bulk path
- outcome: format readers and writers convert whole sample buffers through one bulk byte-order path, choosing the byte order once per buffer, instead of per-type `chunks_exact` loops.
- acceptance: the ritk-vtk binary scalar reader, ritk-mif's float writer, the ritk-nifti and ritk-analyze voxel decoders and the JPEG 2000 QCD step sizes use one shared bulk decode and encode, with the same output bytes and values; each crate's tests pass unchanged, and an instruction-count or pinned run shows no regression on each reader's decode loop.
- status: todo
- priority: tightening
- scope: `crates/ritk-codecs/src/byte_decode.rs`, `crates/ritk-vtk/src/io/reader.rs`, `crates/ritk-mif/src/writer.rs`, `crates/ritk-nifti/src/header/types.rs`, `crates/ritk-analyze/src/reader.rs`, `crates/ritk-codecs/src/jpeg_2000/codestream.rs`
- needs: none
- next: profile these call sites, route required bulk operations through `EndianScalar`, and remove the duplicated loops without changing output bytes or values.
- basis: 78bcf856d5f58a8ed2a3943aba1cac845ff655f1

<a id="RITK-FORMAT-CONVERSION-001"></a>
## RITK-FORMAT-CONVERSION-001: Convert image formats through RITK
- outcome: keep every supported medical-image and scientific-image parser, writer, and conversion in RITK behind typed format capabilities.
- acceptance: DICOM, NIfTI, NRRD, MetaImage, MINC, MIF, MGH/MGZ, Analyze, VTK, PNG, TIFF, JPEG, and GIFTI declare readable and writable data models; each compatible conversion preserves samples, geometry, calibration, and acquisition metadata or returns a typed loss report before output; DICOM directory loading exposes every series; Metis consumes RITK APIs and contains no parser or converter.
- status: todo
- priority: architecture
- needs: RITK-TYPED-SAMPLES-006
- scope: `crates/ritk-image-io/`, `crates/ritk-io/`, every listed format crate, conversion tests, and the RITK user manual.
- next: after RITK-TYPED-SAMPLES-006 lands, publish the format capability table and implement loss-aware preflight and execution for compatible data models.
- basis: 78bcf856d5f58a8ed2a3943aba1cac845ff655f1

<a id="RITK-TYPED-SAMPLES-006"></a>
## RITK-TYPED-SAMPLES-006: Preserve NRRD stored samples and semantics
- outcome: read and write NRRD volumes through one shared stored-volume contract without implicit sample conversion.
- acceptance: all ten fixed-width types and float bits round-trip; axis units, non-orthogonal geometry, measurement frame, coordinate maps, bounds, centering, calibration, and diffusion axes are preserved or rejected with a typed error before output; conflicting headers and resource-limit violations fail explicitly; manual examples cover supported cases.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-image-io/`, `crates/ritk-nrrd/`, `crates/ritk-codecs/`, `crates/ritk-spatial/`, the NRRD format manual, and ADR 0054.
- next: resolve the independent-review findings for alias and spatial-field conflicts, duplicated coordinate-map parameters, diffusion-axis validation, and bounded payload allocation; run the focused RITK gates.
- basis: 78bcf856d5f58a8ed2a3943aba1cac845ff655f1

<a id="RITK-SNAP-INTERACTION-REGIONS-001"></a>
## RITK-SNAP-INTERACTION-REGIONS-001: Separate region interaction tests
- outcome: give region tools one test module without changing behavior.
- acceptance: every moved test remains present and passes with the same assertions.
- status: todo
- priority: tightening
- needs: none
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `tools/interaction/tests/regions.rs`
- next: move only region cases and keep the parent test module as a manifest.
- basis: 5950857874f154cbff145d18c359883e6b58615f

<a id="RITK-SNAP-INTERACTION-MEASUREMENTS-001"></a>
## RITK-SNAP-INTERACTION-MEASUREMENTS-001: Separate measurement tests
- outcome: give measurement tools one test module without changing behavior.
- acceptance: moved measurement tests preserve their existing value oracles and pass.
- status: todo
- priority: tightening
- needs: RITK-SNAP-INTERACTION-REGIONS-001
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `tools/interaction/tests/measurements.rs`
- next: move measurement cases after the region extraction lands.
- basis: 3fcdc3dd

<a id="RITK-SNAP-INTERACTION-STATE-001"></a>
## RITK-SNAP-INTERACTION-STATE-001: Separate interaction state tests
- outcome: give tool-state transitions one test module without changing behavior.
- acceptance: moved transition tests preserve valid and invalid state outcomes and pass.
- status: todo
- priority: tightening
- needs: RITK-SNAP-INTERACTION-MEASUREMENTS-001
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `tools/interaction/tests/state.rs`
- next: move the state cases after measurement tests.
- basis: 3fcdc3dd

<a id="RITK-SNAP-INTERACTION-WINDOW-LEVEL-001"></a>
## RITK-SNAP-INTERACTION-WINDOW-LEVEL-001: Separate window-level tests
- outcome: give window-level interaction tests one module without changing behavior.
- acceptance: moved window-level tests retain their value assertions and pass.
- status: todo
- priority: tightening
- needs: RITK-SNAP-INTERACTION-STATE-001
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `tools/interaction/tests/window_level.rs`
- next: move the remaining window-level cases and leave no implementation in the test manifest.
- basis: 3fcdc3dd

<a id="RITK-SNAP-OBLIQUE-APP-ADAPTER-001"></a>
## RITK-SNAP-OBLIQUE-APP-ADAPTER-001: Route oblique viewer actions
- outcome: apply pointer and navigation actions to the RITK oblique view model.
- acceptance: production oblique viewport mapping consumes the checked shared geometry; clicks link the correct voxel, wheel translates the plane, and orientation keys update its basis; invalid actions reject.
- status: todo
- priority: feature
- needs: none
- scope: `crates/ritk-snap/src/app/{action_adapter.rs,pointer_ops.rs,screen_image_geometry.rs,oblique_viewport.rs}` and oblique tests
- next: add the adapter with the smallest complete happy-path test.
- basis: 4e2e797c6af199d0c33e37148e784b2becc04981

<a id="RITK-SNAP-OBLIQUE-APP-TESTS-001"></a>
## RITK-SNAP-OBLIQUE-APP-TESTS-001: Verify oblique adapter boundaries
- outcome: test source changes, invalid pointers, and orientation limits through the app contract.
- acceptance: adversarial and boundary inputs return the specified actions and never mutate unrelated planes.
- status: todo
- priority: verification
- needs: RITK-SNAP-OBLIQUE-APP-ADAPTER-001
- scope: `crates/ritk-snap/src/app/tests/action_adapter/oblique.rs`
- next: add rotated, edge, and invalid-input cases.
- basis: 3fcdc3dd

<a id="RITK-SNAP-OBLIQUE-SESSION-MODULES-001"></a>
## RITK-SNAP-OBLIQUE-SESSION-MODULES-001: Separate native oblique session concerns
- outcome: place plane rendering and gesture reduction in cohesive session modules.
- acceptance: extraction preserves rendered frames, retained-valid-frame behavior, and event outcomes.
- status: todo
- priority: architecture
- needs: RITK-SNAP-OBLIQUE-APP-ADAPTER-001
- scope: `crates/ritk-snap/src/presentation/native_session/oblique.rs`
- next: separate render/rebuild state from the gesture reducer.
- basis: bf589f94b9826be8b4b05c85e5da31c338f01bab

<a id="RITK-SNAP-OBLIQUE-SESSION-WIRING-001"></a>
## RITK-SNAP-OBLIQUE-SESSION-WIRING-001: Wire the oblique session pane
- outcome: initialize and update the fourth pane from the active volume and patient cursor.
- acceptance: source replacement rebuilds the plane and presents current pixels; failure keeps the last valid frame and surfaces the error.
- status: todo
- priority: correctness
- needs: RITK-SNAP-OBLIQUE-SESSION-MODULES-001
- scope: `crates/ritk-snap/src/presentation/native_session/{composition,session,events}.rs`
- next: connect the plane lifecycle to native session state.
- basis: 3fcdc3dd

<a id="RITK-SNAP-OBLIQUE-ROUTING-001"></a>
## RITK-SNAP-OBLIQUE-ROUTING-001: Route native oblique input
- outcome: route keyboard, pointer, wheel, and repaint events to the active oblique pane.
- acceptance: navigation changes only the oblique plane; linked cursor updates orthogonal panes; invalid events preserve visible state.
- status: todo
- priority: feature
- needs: RITK-SNAP-OBLIQUE-SESSION-WIRING-001, RITK-SNAP-OBLIQUE-APP-TESTS-001
- scope: `crates/ritk-snap/src/presentation/native_session/routing.rs`, `events.rs`
- next: implement the bounded event state machine and user-visible controls.
- basis: 3fcdc3dd

<a id="RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001"></a>
## RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001: Present patient-space lengths
- outcome: render persisted length endpoints and millimetre labels over the active plane.
- acceptance: label positions use the same plane projection as pixels; a 3–4–5 measurement displays 5 mm.
- status: todo
- priority: correctness
- needs: RITK-SNAP-OBLIQUE-SESSION-WIRING-001
- scope: `crates/ritk-snap/src/presentation/native_session/layout/measurement.rs`, `crates/ritk-snap/src/ui/measurements/`
- next: project endpoints through the shared physical plane mapping.
- basis: ba37d37658c679ba354eccf904f8074a2a2aff5e

<a id="RITK-SNAP-OBLIQUE-SESSION-TESTS-001"></a>
## RITK-SNAP-OBLIQUE-SESSION-TESTS-001: Verify oblique rendering and startup
- outcome: cover real session initialization, frame changes, resize, and retained-frame failures.
- acceptance: tests assert frame pixels and semantic state for valid and rejected transitions.
- status: todo
- priority: verification
- needs: RITK-SNAP-OBLIQUE-SESSION-WIRING-001
- scope: `crates/ritk-snap/src/presentation/native_session/tests/oblique/`
- next: add rendering and startup cases against manufactured volumes.
- basis: 3fcdc3dd

<a id="RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001"></a>
## RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001: Verify complete native gestures
- outcome: test linked navigation, plane rotation, cine-independent event timing, and patient-length interaction.
- acceptance: end-to-end event sequences assert exact cursor, orientation, measurement, and repaint outcomes.
- status: todo
- priority: verification
- needs: RITK-SNAP-OBLIQUE-ROUTING-001, RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001
- scope: `crates/ritk-snap/src/presentation/native_session/tests/oblique/gestures.rs`, `measurement.rs`
- next: verify full pointer and keyboard workflows on rotated anisotropic data.
- basis: 3fcdc3dd

<a id="RITK-SNAP-OBLIQUE-MANUAL-001"></a>
## RITK-SNAP-OBLIQUE-MANUAL-001: Demonstrate oblique MPR in the user manual
- outcome: document native launch and controls with a genuine public MRI-DIR phantom capture.
- acceptance: CLI, README, and manual show the 94-file public phantom in a complete app window with working menus or toolbar controls, pane controls, and all four anatomical panes; provenance identifies the exact capture revision.
- status: todo
- priority: verification
- needs: RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001, RITK-SNAP-OBLIQUE-SESSION-TESTS-001
- scope: `crates/ritk-snap/src/{launch.rs,main.rs}`, crate README, `docs/manual/`, provenance tests
- next: capture the completed viewer from the public phantom and validate the manual artifacts.
- basis: 4e2e797c6af199d0c33e37148e784b2becc04981

<a id="RITK-BROWSER-READ-001"></a>
## RITK-BROWSER-READ-001 â€” Reproduce and close WebKit study reads
- Status: blocked; compacted 2026-09-18; full delivery history remains in git.
- Scope: saved public MRI-DIR chooser replay and cross-engine gallery; provider/host fixes stay with their owners.
- Acceptance: read-path cause and production fix, 94 file hashes, three exact pixel oracles, bounded rejections and clean sessions on Chromium, Firefox and WebKit.
- Risk: [patch]; dependency: [Metis read diagnosis](../metis/backlog.md#METIS-BROWSER-READ-001).
- Delivery: RITK PR [#422](https://github.com/ryancinsight/ritk/pull/422), merge `18ee4e55b`; the current failure evidence is recorded and the external WebKit authorization residual remains blocked.
- Evidence: current hosted [run 35759891764](https://github.com/ryancinsight/ritk/actions/runs/35759891764) builds RITK `db17163baf2947e6e8d163e2750c4fec7f93b671` against Metis `1b10541c2ef7a849e6ff66a3c778874bdf96de7b`; Chromium and Firefox pass the 94-file real MRI study, three exact pixel oracles and lifecycle checks; WebKit accepts the chooser but denies the first bounded read under Safari 26.6.2, with diagnostics artifact [10709802466](https://github.com/ryancinsight/ritk/actions/runs/35759891764/artifacts/10709802466) preserved in the provenance JSON.
- Diagnosis: the current WebKit trace shows the sandbox rejecting the bounded whole-file read after chooser acceptance; earlier isolated one-file/full-batch probes also failed across the available browser read paths. DICOM stays in RITK.
- Verification: locked `ritk-snap` nextest 487/487, strict native/WASM Clippy and checks, formatting, rustdoc and lockfile validation pass; [proof and log hashes](docs/manual/images/dicom-metis-real-browser-mri-cross-engine.json).
- Blocker: exact SafariDriver/WebKit selected-file authorization defect remains external; current run 35759891764 reproduces the denial after SafariDriver accepts all 94 files (artifact [10709642592](https://github.com/ryancinsight/ritk/actions/runs/35759891764/artifacts/10709642592), diagnostics [10709802466](https://github.com/ryancinsight/ritk/actions/runs/35759891764/artifacts/10709802466)). Re-open when the corrected browser/runner path grants real-file reads; application byte-read APIs cannot grant that access.

<a id="RITK-BROWSER-WEBGPU-001"></a>
## RITK-BROWSER-WEBGPU-001 â€” Demonstrate browser WebGPU presentation [arch] [minor]
- Status: blocked; compacted 2026-09-18; full delivery history remains in git.
- Scope: replay the saved public MRI-DIR study through the RITK-owned `?renderer=webgpu` page and retain actual canvas/window evidence; RITK owns DICOM decoding and clinical pixels, while MÃ©tis remains the format-neutral canvas host.
- Acceptance: a configured browser runner reports an adapter, presents the three saved-study canvases, records revision-bound PNGs and semantic attributes, and completes bounded teardown without a raster fallback.
- Blocker: hosted Chromium in run [35759891764](https://github.com/ryancinsight/ritk/actions/runs/35759891764) reports no WebGPU adapter; the setup error and failure capture artifact [10710217395](https://github.com/ryancinsight/ritk/actions/runs/35759891764/artifacts/10710217395) are preserved in [`dicom-metis-real-browser-mri-webgpu-failure.png`](docs/manual/images/dicom-metis-real-browser-mri-webgpu-failure.png).

<a id="MIG-439-03"></a>

## MIG-439-03 â€” Replace remaining Burn NdArray backend aliases with Atlas-backed surfaces

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Atlas-backed surfaces. RESCOPED (was READY; original acceptance criteria do not hold â€” see Sprint 465 finding).** **Correction (Sprint 465, evidence-based)**: investigated the strongest candidate crate (`ritk-jpeg`, smallest burn_ndarray footprint, already has a parallel Coeus reader) to execute thâ€¦

<a id="PERF-435-01"></a>

## PERF-435-01 â€” Route MSE through fused interpolation. PARTIAL.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Generalized `ritk_interpolation::transform_and_interpolate` over spatial dimensionality, generalized the OOB mask helper over the image shape length, and routed `MeanSquaredError` through the fused transform-to-index-to-linear interpolation path. Evidence tier: value-semantic nextest and focused tiâ€¦

<a id="PROVIDER-420-01"></a>

## PROVIDER-420-01 â€” Hermes complex dispatch bound cleanup. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: The local Atlas provider graph exposed that Hermes complex SIMD operations still require `Neg` at the complex-operation dispatch surface after broader unsigned scalar support. A minimal local fix passes `cargo check -p hermes-simd --all-targets`; full provider rustfmt is still blocked by unrelatedâ€¦

<a id="PERF-419-01"></a>

## PERF-419-01 â€” Registration test runtime budget breach. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 419's `ritk-registration` nextest gate passed but exposed integration tests above the 30s slow budget, including 100s, 146s, and 193s rows. Treat this as a real performance defect to profile; do not weaken or skip those tests.

<a id="COEUS-406-01"></a>

## COEUS-406-01 â€” Fix dirty Coeus autograd provider compile break. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: RITK doctest/doc gates against the current local Atlas stack are blocked after refreshing Coeus path packages to `0.2.6`: `D:\atlas\repos\coeus` is dirty on `test/cuda-parity-suite`, and `coeus-autograd` fails to compile in shape/reduction ops. This must be fixed in Coeus before RITK can claim docsâ€¦

<a id="PERF-406-02"></a>

## PERF-406-02 â€” Registration test runtime budget breach. OPEN.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 406's touched-package `nextest` gate passed but exposed registration tests above the 30s slow budget, including 93s, 129s, and 183s rows. Treat this as a real performance defect to profile; do not weaken or skip those tests.

<a id="PERF-387-02"></a>

## PERF-387-02 â€” Continue flat-buffer memory-efficiency audit. IN PROGRESS.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 387 flattened `VectorConfidenceConnected` covariance/inverse matrices and removed the B-spline legacy placeholder. Sprint 389 flattened `InverseDisplacementField` TPS spline/affine coefficient blocks after the solve. Sprint 390 flattened TIFF grayscale/RGB page accumulation by removing `Vec<â€¦

<a id="MIG-387-01"></a>

## MIG-387-01 â€” Atlas crate migration audit.

- Status: todo; retained from the historical execution section; full detail remains in git.

- Scope: Continue replacing production `nalgebra`/`ndarray`/`burn` surfaces with `leto`/`coeus`/ `hephaestus` only after each target operation has a verified equivalent contract and focused differential tests. Do not remove boundary dependencies used only for file-format interop or external framework contraâ€¦

<a id="MIG-387-02"></a>

## MIG-387-02 â€” Spatial Leto SSOT migration. IN PROGRESS.

- Status: in-progress; retained from the historical execution section; full detail remains in git.

- Scope: Sprint 408 migrates `ritk-spatial` storage to Leto fixed vectors/matrices and removes direct `nalgebra` dependencies from `ritk-core`, `ritk-metaimage`, `ritk-nrrd`, `ritk-nifti`, and `ritk-mgh` spatial direction setup. Sprint 409 moves DICOM IO, MINC, and filter spatial-transform consumers onto `Dâ€¦

## Sprint 342 (Phase 20) â€” Coeus Migration Readiness Audit

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: In Progress

- | MIG-342-04 | RITK-owned tensor contract over Coeus CPU backend | **Open** |

- | GPU-342-05 | Coeus WGPU differential test harness for RITK operation subset | **Open** |

- | REG-342-06 | Registration autodiff tape continuity proof/test under Coeus | **Open** |

- | MODEL-342-07 | `ritk-model` Coeus module/parameter/3-D convolution migration design | **Open** |

- | PY-342-08 | Python binding conversion plan over Coeus-backed Rust core | **Open** |

- The next implementation stage is not a dependency swap. It is the RITK tensor

- ### Residual risks



## Sprint 332 (0.50.95) â€” Documentation Compaction + Structural Audit + Benchmark

- Status: in-progress; historical sprint retained only for unresolved rows.

- **Status**: In Progress

- | BENCH-332-03 | `STACK_WEIGHTS_CAPACITY=32` Criterion benchmark â€” measure AVX2 speedup vs 8-entry version | **Open** |

- | GPU-332-04 | Evaluate `sparse.rs` GPU-backend potential (Burn autodiff scatter compatibility, custom kernel feasibility) | **Open** |

- | CRLF-332-05 | Git CRLF normalization (`git add --renormalize`) â€” blocked by missing test data files | **Blocked** |



<a id="RITK-GAP-2026-08-20-01"></a>
## RITK-GAP-2026-08-20-01 [major][arch] â€” collapse the dual `X` / `X_native` surface
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-02"></a>
## RITK-GAP-2026-08-20-02 [minor] â€” fuzz the sixteen format parsers
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-03"></a>
## RITK-GAP-2026-08-20-03 [patch] â€” retire the GPU naming and the unbacked accelerator claims
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-04"></a>
## RITK-GAP-2026-08-20-04 [patch] â€” evict the 1.6 GB tracked binary payload
- Status: in-progress; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-05"></a>
## RITK-GAP-2026-08-20-05 [patch] â€” derive the escalated test budgets and sweep the dead filters
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-06"></a>
## RITK-GAP-2026-08-20-06 [patch] â€” raise the lint and documentation floor
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-07"></a>
## RITK-GAP-2026-08-20-07 [patch] â€” restore the CHANGELOG version axis
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-08"></a>
## RITK-GAP-2026-08-20-08 [patch] â€” write the registration and dispatch book chapters
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-GAP-2026-08-20-09"></a>
## RITK-GAP-2026-08-20-09 [patch] â€” derive or remove the MI subsample stride
- Status: todo; compacted 2026-09-18; full delivery history remains in git.
- Scope: historical item contract retained in the archived source block; re-open with the original DoR.

<a id="RITK-PYTHON-VTK-DIRECTION-001"></a>
## RITK-PYTHON-VTK-DIRECTION-001: Python images cannot be written to legacy VTK
- outcome: a Python-created image reaches a VTK file with its geometry intact, or the restriction is a recorded, accepted product decision rather than a silent dead path.
- acceptance: either `rio.write_image` handles the permutation direction `numpy_array_direction()` produces (axis permutation, no resampling needed) and a roundtrip test proves values + geometry survive, or the restriction is documented at the `write_image` contract with the typed error asserted by `test_write_image_vtk_rejects_non_identity_direction`.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-python/src/image.rs` (constructor direction), `crates/ritk-vtk/src/io/writer.rs` (representability check), `crates/ritk-python/tests/test_coverage_gaps.py`.
- next: decide whether the legacy writer gains permutation support or the Python surface gains a direction parameter; until then the rejection test (not a roundtrip) is the correct gate.
- risk: silent axis misorientation in medical images; the fail-closed rejection exists precisely to prevent it, so any writer change must preserve it for genuinely unrepresentable geometry.
- basis: f095910f
