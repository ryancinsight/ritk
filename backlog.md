# RITK execution backlog

<a id="RITK-SNAP-OBLIQUE-NATIVE-001"></a>
## RITK-SNAP-OBLIQUE-NATIVE-001: Native oblique MPR
- outcome: deliver a native four-plane oblique viewer with patient-space navigation and measurement.
- acceptance: all child items land; invalid geometry is rejected; the public phantom capture shows one complete, uncropped app window with visible menus or toolbar buttons and all four anatomical panes; native visual and value-semantic gates pass.
- status: todo
- priority: architecture
- needs: RITK-SNAP-INTERACTION-REGIONS-001, RITK-SNAP-INTERACTION-MEASUREMENTS-001, RITK-SNAP-INTERACTION-STATE-001, RITK-SNAP-INTERACTION-WINDOW-LEVEL-001, RITK-SNAP-OBLIQUE-APP-ADAPTER-001, RITK-SNAP-OBLIQUE-APP-TESTS-001, RITK-SNAP-OBLIQUE-SESSION-MODULES-001, RITK-SNAP-OBLIQUE-SESSION-WIRING-001, RITK-SNAP-OBLIQUE-ROUTING-001, RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001, RITK-SNAP-OBLIQUE-SESSION-TESTS-001, RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001, RITK-SNAP-WINDOW-CONTROLS-001, RITK-SNAP-OBLIQUE-MANUAL-001
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

<a id="RITK-TYPED-SAMPLES-002"></a>
## RITK-TYPED-SAMPLES-002: Preserve NIfTI stored samples and rescale
- outcome: NIfTI-1 and NIfTI-2 read and write supported stored sample types in RITK and apply `scl_slope`/`scl_inter` under an explicit conversion policy.
- acceptance: ten-type value-semantic round trips cover both byte orders; NIfTI-1/2 rescale and series behavior match the format contract; labels reject fractional, negative, and out-of-range values; the NIfTI crate docs and example describe the policy.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-nifti/`, `docs/book/nifti_format.md`, its examples and tests
- next: revalidate the stale PR #699 work against the merged sample API, then complete its reader/writer and scaling tests.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-003"></a>
## RITK-TYPED-SAMPLES-003: Read and write MGH in the stored sample type
- outcome: MGH/MGZ reads its four voxel types into `Image<T, B, 3>` and writes the type of `T`, rejecting types MGH cannot store with a typed error (ADR 0053).
- acceptance: generic round trip over u8, i16, i32, f32; an i32 sample above 2^24 survives; `voxel_decode.rs`'s per-type f32 functions are deleted.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-mgh/`
- next: map `VoxelType` onto `SampleType` and decode through `SampleBuffer::decode`.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-004"></a>
## RITK-TYPED-SAMPLES-004: Read and write Analyze 7.5 in the stored sample type
- outcome: Analyze reads u8, i16, i32, f32, f64 into `Image<T, B, 3>`, carries `funused1` as the shared rescale, and writes the type of `T` (ADR 0053).
- acceptance: generic round trip over the five types; a scaled file reads physical values in f32/f64; the `AnalyzeVoxel` trait and `decode_payload` are deleted.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-analyze/`
- next: replace `AnalyzeDatatype` with a `SampleType` code map.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-005"></a>
## RITK-TYPED-SAMPLES-005: Read and write MIF in the stored sample type
- outcome: MIF reads and writes all MRtrix integer and float types including Int64/UInt64 in the stored type and byte order, fixing the width-only decoder that reads int32/uint32 as float bits, uint16 as i16 and int8 as u8 (ADR 0053).
- acceptance: generic round trip over ten types in LE and BE; `decode.rs` and the local `parse_f64_vec` are deleted; the float writer emits the type of `T`.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-mif/`
- next: make `parse_datatype` return `(SampleType, ByteOrder)`.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-006"></a>
## RITK-TYPED-SAMPLES-006: Read and write NRRD in the stored sample type
- outcome: NRRD reads every NRRD scalar type including `int64`/`uint64` into `Image<T, B, 3>` and writes the type of `T`; an unknown `endian` value is a typed error (ADR 0053).
- acceptance: generic round trip over ten types, raw and gzip, both byte orders; type names match the NRRD file format specification's full list.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-nrrd/`
- next: make `read_nrrd` generic over `T` using `SampleBuffer::into_vec`.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-007"></a>
## RITK-TYPED-SAMPLES-007: Read and write MetaImage in the stored sample type
- outcome: MetaImage reads `MET_CHAR` through `MET_ULONG_LONG` and both float types into `Image<T, B, 3>` and writes the `ElementType` of `T` (ADR 0053).
- acceptance: generic round trip over ten types, inline and detached, raw and zlib, both byte orders.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-metaimage/`
- next: extend `element_sample_type` with the missing `MET_*` names, then make the reader generic.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-008"></a>
## RITK-TYPED-SAMPLES-008: Read and write legacy VTK scalars in the stored sample type
- outcome: the VTK binary and ASCII structured-points readers keep the stored scalar type and the writer emits the type of `T` (ADR 0053).
- acceptance: generic round trip over the VTK scalar types; the per-type f32 decode in `io/reader.rs` and the count-free `xml_helpers::parse_floats` are deleted.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-vtk/src/io/`
- next: map `VtkScalarType` onto `SampleType`.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-009"></a>
## RITK-TYPED-SAMPLES-009: Read and write MINC 2 in the stored sample type
- outcome: MINC reads its stored HDF5 type with `valid_range`/`image-min`/`image-max` as the shared rescale, and writes the type of `T` (ADR 0053).
- acceptance: generic round trip over the stored types; per-slice scaling reproduces the existing `tests_scaling.rs` values in f32 and f64; `convert.rs`'s f32 decode is deleted.
- status: todo
- priority: correctness
- needs: none
- scope: `crates/ritk-minc/`
- next: express `IntegerScaling` as per-slice rescale coefficients.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-010"></a>
## RITK-TYPED-SAMPLES-010: Dispatch image I/O over the sample type
- outcome: `ritk-io` reads and writes `Image<T, NativeBackend, 3>` for every `T: Sample` under the caller's `Conversion`, adds MINC and MIF to `ImageFormat`, and exposes a stored-type read returning `SampleBuffer`; its `f32` surfaces stop defaulting to `Cast` (ADR 0053).
- acceptance: the capability tables derive from the dispatch match; a u16 file round-trips through `read_image`/`write_image` as u16 for every format that stores u16; CLI and Python keep `f32` call sites compiling with explicit annotations.
- status: todo
- priority: correctness
- needs: RITK-TYPED-SAMPLES-002, RITK-TYPED-SAMPLES-003, RITK-TYPED-SAMPLES-004, RITK-TYPED-SAMPLES-005, RITK-TYPED-SAMPLES-006, RITK-TYPED-SAMPLES-007, RITK-TYPED-SAMPLES-008, RITK-TYPED-SAMPLES-009
- scope: `crates/ritk-io/src/dispatch.rs`, `crates/ritk-io/src/format/`, `crates/ritk-io/src/domain/mod.rs`, CLI and Python call sites
- next: decide the stored-type image carrier: the `VoxelImage` of draft PR #700 against `SampleBuffer` plus geometry, keeping `SampleType` the one runtime descriptor; then probe with `cargo check --message-format=json` after making `read_image_native` generic.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-TYPED-SAMPLES-011"></a>
## RITK-TYPED-SAMPLES-011: Read and write DICOM pixels in the stored sample type
- outcome: DICOM decodes stored samples after `BitsStored` masking into their integer type, carries Rescale Slope/Intercept as `f64`, and writes integer images at their own bit depth without re-quantizing (ADR 0053).
- acceptance: an i16 CT series round-trips bit-exact with its rescale; f32 input still writes with a computed rescale whose error bound is derived; `decode_native_pixel_bytes_checked` returns `SampleBuffer`.
- status: todo
- priority: correctness
- needs: RITK-TYPED-SAMPLES-010
- scope: `crates/ritk-codecs/src/pixel_layout*`, `crates/ritk-io/src/format/dicom/`, `crates/ritk-dicom/`
- next: change `PixelLayout` rescale fields to `f64` and route the native decode through `SampleBuffer`.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
<a id="RITK-IO-FORMAT-CONVERSION-001"></a>
## RITK-IO-FORMAT-CONVERSION-001: Plan loss-aware image conversions
- outcome: `ritk-io` owns format discovery, conversion planning, and execution across the RITK image formats; GUI, CLI, and Python code call this API.
- acceptance: the registry covers every RITK-supported reader/writer, including DICOM, NIfTI, NRRD, MetaImage, MGH, Analyze, VTK, TIFF, PNG, JPEG, MIF, and MINC; preflight reports sample, geometry, metadata, and lossy-encoding changes before opening output; DICOM series selection is explicit and DICOM output reports derived metadata and quantization; round trips assert typed samples and supported geometry; CLI and user manual use the same RITK API.
- status: todo
- priority: correctness
- needs: RITK-TYPED-SAMPLES-010, RITK-TYPED-SAMPLES-011
- scope: `crates/ritk-io/`, `crates/ritk-cli/`, and the RITK user manual and format examples
- next: repair the preserved conversion candidate against typed dispatch and pixel contracts; deliver preflight before writing and verify every supported lossless pair.
- basis: ed9e3834a5251a932959a0bdb88c922910a8eae6
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
- needs: RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001, RITK-SNAP-OBLIQUE-SESSION-TESTS-001, RITK-SNAP-WINDOW-CONTROLS-001
- scope: `crates/ritk-snap/src/{launch.rs,main.rs}`, crate README, `docs/manual/`, provenance tests
- next: capture the completed viewer from the public phantom and validate the manual artifacts.
- basis: 4e2e797c6af199d0c33e37148e784b2becc04981

<a id="RITK-SNAP-WINDOW-CONTROLS-001"></a>
## RITK-SNAP-WINDOW-CONTROLS-001: Add functional controls to the app window
- outcome: show RITK menus, tool controls, and study status around real views in the Métis native window.
- acceptance: File/View/Tools menus and toolbar controls perform their named viewer actions; chrome input never reaches image panes; the pane-only pixel capture remains unchanged; the public MRI manual image shows the complete app window and working controls.
- status: todo
- priority: feature
- needs: none
- scope: `crates/ritk-snap/src/presentation/native_session/`, `docs/manual/`, screenshot provenance.
- next: implement chrome layout and event reduction in RITK, then capture and inspect the public MRI workflow.
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
