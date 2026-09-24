# RITK execution backlog

<a id="RITK-SNAP-OBLIQUE-NATIVE-001"></a>
## RITK-SNAP-OBLIQUE-NATIVE-001: Native oblique MPR
- outcome: deliver a native four-plane oblique viewer with patient-space navigation and measurement.
- acceptance: all child items land; invalid geometry is rejected; the public phantom capture and manual show the real workflow; native visual and value-semantic gates pass.
- status: todo
- priority: architecture
- needs: RITK-SNAP-OBLIQUE-PIXEL-PROJECTION-001, RITK-SNAP-RESLICE-PIXEL-MODULE-001, RITK-SNAP-RESLICE-ORIENTATION-001, RITK-SNAP-RESLICE-ORIENTATION-TESTS-001, RITK-SNAP-PATIENT-LENGTH-001, RITK-SNAP-INTERACTION-REGIONS-001, RITK-SNAP-INTERACTION-MEASUREMENTS-001, RITK-SNAP-INTERACTION-STATE-001, RITK-SNAP-INTERACTION-WINDOW-LEVEL-001, RITK-SNAP-NATIVE-OVERLAY-001, RITK-SNAP-NATIVE-MPR-COMPOSITION-001, RITK-SNAP-OBLIQUE-VIEWPORT-001, RITK-SNAP-OBLIQUE-APP-ADAPTER-001, RITK-SNAP-OBLIQUE-APP-TESTS-001, RITK-SNAP-OBLIQUE-SESSION-MODULES-001, RITK-SNAP-OBLIQUE-SESSION-WIRING-001, RITK-SNAP-OBLIQUE-ROUTING-001, RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001, RITK-SNAP-OBLIQUE-SESSION-TESTS-001, RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001, RITK-SNAP-OBLIQUE-MANUAL-001
- scope: `crates/ritk-snap/src/{app,presentation,render,tools/interaction,session}/`, `crates/ritk-snap/src/main.rs`, crate README, ADRs, manual, and provenance.
- next: deliver ready child items in dependency order; preserve the public MRI-DIR phantom as the shareable visual fixture.
- risk: [major] [arch]; patient-space annotations and session format 3; no registry release is authorized.
- basis: 3fcdc3dd

<a id="RITK-SNAP-OBLIQUE-PIXEL-PROJECTION-001"></a>
## RITK-SNAP-OBLIQUE-PIXEL-PROJECTION-001: Project patient points to pixels
- outcome: project patient millimetres into continuous output pixels with signed plane distance.
- acceptance: rotated, anisotropic, and skew mappings round-trip; outward-rounded componentwise enclosures keep large normal offsets from widening unrelated pixel axes; boundary clamping requires an enclosure that contains the edge; the unit-plane point [-0.125, 0.5, 1e13] rejects as out of bounds; unresolved and overflowing projections return typed errors.
- status: todo
- priority: correctness
- needs: RITK-SNAP-OBLIQUE-PIXEL-MAPPING-001
- scope: `crates/ritk-snap/src/render/reslice/pixel.rs`, `crates/ritk-snap/src/render/reslice/pixel/interval.rs`, `crates/ritk-snap/src/render/tests_reslice.rs`
- next: implement componentwise interval propagation and the out-of-bounds regression.
- basis: c665c5075a6e712fa11e13d65955ecb3a05f6c13

<a id="RITK-SNAP-RESLICE-PIXEL-MODULE-001"></a>
## RITK-SNAP-RESLICE-PIXEL-MODULE-001: Extract pixel projection operations
- outcome: place reslice pixel sampling and projection operations in their canonical module.
- acceptance: the module extraction preserves every pixel value and rejection case; the shared pixel mapping remains the single implementation.
- status: todo
- priority: tightening
- needs: RITK-SNAP-OBLIQUE-PIXEL-PROJECTION-001
- scope: `crates/ritk-snap/src/render/reslice.rs`, `crates/ritk-snap/src/render/reslice/pixel.rs`
- next: move the pixel operation family without changing its contract.
- basis: 3fcdc3dd

<a id="RITK-SNAP-RESLICE-ORIENTATION-001"></a>
## RITK-SNAP-RESLICE-ORIENTATION-001: Build physical oblique planes
- outcome: construct bounded reslice planes from volume geometry and an in-plane orientation.
- acceptance: orthogonal bases, physical extents, sampling, and pixel mapping use one plane contract; invalid geometry never replaces valid output.
- status: todo
- priority: correctness
- needs: RITK-SNAP-RESLICE-PIXEL-MODULE-001
- scope: `crates/ritk-snap/src/render/reslice.rs`, `crates/ritk-snap/src/render/reslice/orientation.rs`, render exports
- next: add plane construction and orientation operations behind the reslice contract.
- basis: 3fcdc3dd

<a id="RITK-SNAP-RESLICE-ORIENTATION-TESTS-001"></a>
## RITK-SNAP-RESLICE-ORIENTATION-TESTS-001: Verify orientation boundaries
- outcome: cover source changes, plane bounds, and invalid orientation inputs.
- acceptance: rotated, non-square, anisotropic manufactured data verifies orientation and bounds with value-semantic assertions.
- status: todo
- priority: verification
- needs: RITK-SNAP-RESLICE-ORIENTATION-001
- scope: `crates/ritk-snap/src/render/tests_reslice.rs`
- next: add the remaining edge and source-geometry cases.
- basis: 3fcdc3dd

<a id="RITK-SNAP-PATIENT-LENGTH-001"></a>
## RITK-SNAP-PATIENT-LENGTH-001: Persist patient-space lengths
- outcome: store validated endpoint coordinates and millimetre length in viewer annotations.
- acceptance: format 3 writes patient endpoints; formats 1 and 2 still load; a 3–4–5 segment reports 5 mm through snapshots and UI.
- status: todo
- priority: correctness
- needs: RITK-SNAP-RESLICE-ORIENTATION-001
- scope: `crates/ritk-snap/src/{session,tools/interaction,ui}/` and snapshot tests
- next: complete the persisted-format migration and public enum documentation.
- risk: [major] public exhaustive annotation matches require migration.
- basis: 3fcdc3dd

<a id="RITK-SNAP-INTERACTION-REGIONS-001"></a>
## RITK-SNAP-INTERACTION-REGIONS-001: Separate region interaction tests
- outcome: give region tools one test module without changing behavior.
- acceptance: every moved test remains present and passes with the same assertions.
- status: todo
- priority: tightening
- needs: RITK-SNAP-PATIENT-LENGTH-001
- scope: `crates/ritk-snap/src/tools/interaction/tests.rs`, `tools/interaction/tests/regions.rs`
- next: move only region cases and keep the parent test module as a manifest.
- basis: 3fcdc3dd

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

<a id="RITK-SNAP-NATIVE-OVERLAY-001"></a>
## RITK-SNAP-NATIVE-OVERLAY-001: Extract native pane overlays
- outcome: place pane labels, tool state, and measurement presentation in one layout module.
- acceptance: existing native captures and interaction outcomes remain unchanged after the pure extraction.
- status: todo
- priority: architecture
- needs: RITK-SNAP-PATIENT-LENGTH-001
- scope: `crates/ritk-snap/src/presentation/native_session/layout/`
- next: extract overlay composition from the native session layout.
- basis: 3fcdc3dd

<a id="RITK-SNAP-NATIVE-MPR-COMPOSITION-001"></a>
## RITK-SNAP-NATIVE-MPR-COMPOSITION-001: Compose the four native panes
- outcome: compose three orthogonal panes and the physical oblique pane in one layout path.
- acceptance: pane geometry remains non-overlapping and responsive; existing three-pane layout remains unchanged.
- status: todo
- priority: architecture
- needs: RITK-SNAP-NATIVE-OVERLAY-001, RITK-SNAP-RESLICE-ORIENTATION-001
- scope: `crates/ritk-snap/src/presentation/native_session/layout/composition.rs`
- next: extend composition only after the overlay module exists.
- basis: 3fcdc3dd

<a id="RITK-SNAP-OBLIQUE-VIEWPORT-001"></a>
## RITK-SNAP-OBLIQUE-VIEWPORT-001: Map oblique viewport interactions
- outcome: map viewport points and transforms through the rendered plane geometry.
- acceptance: resize, pan, zoom, and pixel-to-patient mappings agree on rotated anisotropic test data.
- status: todo
- priority: correctness
- needs: RITK-SNAP-RESLICE-ORIENTATION-001, RITK-SNAP-NATIVE-MPR-COMPOSITION-001
- scope: `crates/ritk-snap/src/app/oblique_viewport.rs`, app exports
- next: implement the viewport mapping as an independent app value.
- basis: 3fcdc3dd

<a id="RITK-SNAP-OBLIQUE-APP-ADAPTER-001"></a>
## RITK-SNAP-OBLIQUE-APP-ADAPTER-001: Route oblique viewer actions
- outcome: apply pointer and navigation actions to the RITK oblique view model.
- acceptance: clicks link the correct voxel, wheel translates the plane, and orientation keys update its basis; invalid actions reject.
- status: todo
- priority: feature
- needs: RITK-SNAP-OBLIQUE-VIEWPORT-001, RITK-SNAP-PATIENT-LENGTH-001
- scope: `crates/ritk-snap/src/app/action_adapter.rs`, `app/pointer_ops.rs`, and oblique modules
- next: add the adapter with the smallest complete happy-path test.
- basis: 3fcdc3dd

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
- needs: RITK-SNAP-NATIVE-MPR-COMPOSITION-001, RITK-SNAP-OBLIQUE-APP-ADAPTER-001
- scope: `crates/ritk-snap/src/presentation/native_session/oblique.rs`
- next: separate render/rebuild state from the gesture reducer.
- basis: 3fcdc3dd

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
- needs: RITK-SNAP-OBLIQUE-SESSION-WIRING-001, RITK-SNAP-PATIENT-LENGTH-001
- scope: `crates/ritk-snap/src/presentation/native_session/layout/measurement.rs`, `crates/ritk-snap/src/ui/measurements/`
- next: project endpoints through the shared physical plane mapping.
- basis: 3fcdc3dd

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
- acceptance: CLI, README, manual, lossless image, and provenance identify the 94-file public phantom and exact capture revision; visual gate confirms all four anatomical panes.
- status: todo
- priority: verification
- needs: RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001, RITK-SNAP-OBLIQUE-SESSION-TESTS-001
- scope: `crates/ritk-snap/src/{launch.rs,main.rs}`, crate README, `docs/manual/`, provenance tests
- next: capture the completed viewer from the public phantom and validate the manual artifacts.
- basis: 3fcdc3dd

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
