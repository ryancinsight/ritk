# ADR 0026: Viewer presentation migration to Métis

Status: Accepted
Date: 2026-09-06
Revision 2026-11-19: consolidate current constraints, remove delivery history, and reconcile executable ownership with ADRs 0056/0061; source/reference review in [PR #1057](https://github.com/ryancinsight/ritk/pull/1057), with no new runtime or performance evidence.

## Intent and scope

The user selects Métis as the viewer presentation framework, with correct DICOM opening and viewing required for acceptance.
This record states requirements, not delivered capabilities.
The [feature comparison](../manual/features.md) and [board](../../backlog.md) own current coverage and remaining work; [shell removal](../../backlog/RITK-SNAP-EFRAME-REMOVAL-001.md) owns the remaining cutover.
Historical delivery evidence remains in Git and its originating changes.
A passing earlier revision does not certify the current application.

## Ownership and migration

RITK owns DICOM parsing and codec selection, study/series/frame identity, decoded pixels, physical geometry, medical display transforms, navigation, analysis, measurements, overlays, PACS, persistence and export.
Métis owns generic presentation, input, window/browser lifecycle and scoped host access.
Iris retains its visualization contracts; Moirai owns reusable execution, transport and filesystem primitives.
Move GUI carriers out of viewer state where needed; never move DICOM policy into Métis, duplicate decoders or introduce a parallel host-owned DICOM volume.
Implement framework gaps in Métis and reusable runtime gaps in Moirai.
The [Métis V09 contract](../../../metis/docs/VERIFICATION.md#V09) governs the shared application demonstration, without certifying viewer availability.
RITK translates provider input into format-neutral events, reduces them to typed viewer actions, and applies the same viewer transitions across hosts; host IPC transports actions, not clinical state.
RITK produces validated RGBA frames and owns image transforms.
Generic display-list commands carry chrome; they do not transfer DICOM metadata, selection policy or viewer state into the host.
The viewer remains in the workspace under [ADR 0056](0056-viewer-crate-returns-to-the-workspace.md): propagate fallible pixel access, own overlay commands until one display-list conversion, and use provider accessors rather than exposed storage or compatibility shims.
While gaps remain, retain the complete eframe adapter behind the explicit optional `eframe-shell` dependency boundary; require the default active graph to exclude egui, eframe and native-dialog dependencies.
Preserve explicit shell selection, the Windows Métis default and the Métis browser entry; other native hosts retain their working shell until replacement acceptance.
[ADR 0061](0061-asterslice-executable.md) owns the single `asterslice` executable and stable package/library identities; a separate legacy executable is not this migration's contract.
Preserve required behavior in independently verified vertical increments.
Keep the working shell until replacement acceptance; complete cutover removes superseded dependencies, feature gates and affected callers, without a forwarding layer or duplicated viewer state.
The [migration guide](../migration_selected_dicom.md) owns changed caller surfaces and persisted-format transitions.

## Acquisition selection and restoration

One RITK discovery descriptor carries series identity and exact member paths through primary/secondary selection, highlighting and pending loads.
A folder is a discovery location, not an acquisition identifier.
A selected instance supplies its UID before neighbouring members are collected.
Unselected directory or byte-batch loading requires one unambiguous image series; population, dimensions and tied counts never authorize a choice or merge.
Validate image identity before accumulating metadata or assembling geometry.
Never rediscover from folder plus UID in place of retaining discovered membership.
A selected DICOMDIR, or one present for directory discovery, owns its referenced file set.
Only reachable active IMAGE records are admitted.
Validate RecordInUseFlag, next/lower links, incoming-link uniqueness, cycles and first/last root-chain termination; check referenced SOP class, SOP instance and transfer syntax against the index.
Malformed indices, missing references, absolute components, parent traversal and root escape fail the load; invalid contents never trigger an unrelated flat-directory scan.
Discovery and decoding use the same resolved members.
Explicit selection is revalidated before decode; unknown UIDs and invalid source kinds fail closed.
Ambiguous native folders require an RITK-owned selector, never an arbitrary acquisition.
Cancellation retains the current study/frame; failed replacement preserves them and surfaces a diagnostic, with selection/recovery still usable.
Session documents record selection intent, not trusted metadata.
Restore exact recorded members and verify UID before replacing study or presentation state.
Older path-based sessions remain readable at the persistence boundary and follow corrected opening rules; ambiguous old directories require explicit selection.
Public callers use the discovery descriptor directly, without folder-only forwarding APIs.
Selection tests distinguish same-folder series, selected minority data, reversed input order, unselected-first-file metadata, inactive/unreachable index records and identity mismatch.
Missing external files remain rejection cases; complete synthetic indexed studies supply success oracles instead of dropping references from incomplete datasets.

## Host behavior and lifecycle

Inventory the application at a pinned revision: opening/series browser, PACS, orthogonal/projection views, window/level, pan/zoom/cursor, overlays/annotations, filters/segmentation, persistence/export, shortcuts, errors and recovery.
Preserve pointer buttons, coordinates, modifiers, key repeat, Unicode text, IME, resize, DPI, focus and lifecycle semantics, with bounded event batches and surfaced allocation failures.
Keep held-button state bounded, multi-button deltas deterministic, and click/drag release classification consistent.
Validate nonzero DPI and preserve bounded framebuffer behavior through resize/minimize.
Malformed event sequences, coordinate overflow, overlong composition and allocation failure leave dispatch state unchanged.
Focus loss cancels reducer and viewer gestures; close/destruction terminates the host lifecycle.
Preserve multi-click anchors across releases and accumulate repaint requests across each batch.
Losing the final pointer coordinate clears held gesture state.
Use finite display-pixel `f64` coordinates and checked deltas; subtract viewport origin and transform before narrowing.
Painting, cursor updates and sampling consume mapped image points rather than reconstructing from narrowed screen positions.
RITK owns Ctrl/Command wheel zoom, plain vertical wheel slice steps and modifier interpretation.
Space toggles cine; ignore repeated toggle/rate key presses, reanchor timing on rate changes, and feed shared 1–60 FPS cine state from host clocks.
Share modality-selected window/level presets across hosts.
Valid selections invalidate all affected planes; nonfinite, fractional or out-of-range indices reject before state/pixels change.
Browser tests use trusted input and verify semantic, RGBA and cleanup results.
Browser frame scheduling follows the owned animation-frame lifetime, validates timestamps and cancels registration on drop; unchanged bitmap extents do not require resetting the canvas.
Browser handles/bytes confer neither native filesystem nor PACS authority.
Bounded named-byte ingress reaches RITK classification/loading through a format-neutral carrier.
Per-target load generations and cooperative cancellation prevent superseded, cancelled or closed tasks from publishing volume, status or frames.
Current failure retains the previous study; the task bridge owns no DICOM logic.
Native pickers return selection or explicit failure to RITK; cancellation preserves state.
Native non-repeated Ctrl+O reopens through RITK loading; browser Ctrl+O remains browser-owned.
Cutover requires verified native/browser input and ingress, focus/text/IME, widgets, text, clipboard, context menus, accessibility, overlays, multi-viewport dispatch and recovery, not merely a matching canvas.
GPU-capable presentation must cover three orthogonal views and projections without replacing Iris contracts or copying DICOM ownership into Métis.
Required unsupported host cases remain open gaps.
Complete operating-system application-window readback and appropriate packaged-host evidence remain cutover obligations.
Source-frame or application-content capture excludes OS chrome and cannot substitute for full-window evidence.
Capture must be finite, require successful study publication, and report bounded failure rather than accepting an empty first frame.
Cross-platform installers require their own release evidence.
Packaging excludes DICOM/patient-data resources; producing an executable, inventory or installer grants no publication, signing or patient-data inclusion authority.

## Ingestion resources and filesystem boundary

Assets are selected identity, pixels/geometry and files outside the imported set.
Untrusted Part 10 bytes cross the parser, index references the filesystem, and persisted selectors the restore boundary.
Use explicit UID/member validation, bounded parsing, reference rejection and validation on restore.
Public screenshots, fixtures and diagnostics contain no patient data; this decision introduces no remote upload or service.
Enforce encoded-byte, element-count and nesting dimensions of Consus `ParseBudget` at `ritk-dicom`; `ritk-io` owns cumulative retained-study and separate decoded-workspace budgets through [DicomReadBudget](../../crates/ritk-io/src/format/dicom/reader/budget.rs).
Validate remaining input and allocation allowances before materializing meta headers, values, sequences or fragments.
Encoded size alone does not bound parser allocation; logical-length limits do not prove reserved capacity or peak memory.
Deflated datasets require a bounded decoder before admission; the [current backend](../../crates/ritk-dicom/src/backend/dicom_rs.rs) rejects them explicitly after structural preflight, before object construction.
Charge retained bytes before storage; check shape arithmetic and the planned peak frame/resample/contiguous-volume workspace before allocation.
Bound tasks, pending requests and buffers independently.
Retain the exact validated Part 10 input through decoding so scan/decode cannot substitute changed files; retention beyond reconstruction is not required by this boundary.
Root-confined opens belong to Moirai's filesystem layer: validate relative components, resolve from directory handles, reject link/traversal escape, and read the returned handle.
RITK owns parser/member policy and consumer budgets.
A prior path check does not prove safety against replacement; require independent filesystem tests for handle-based race protection.
Unsupported platform operations return explicit errors, never an unconfined fallback.
Exercise allocation requests at boundaries, malformed declared lengths, retained/decoded ceilings, missing references, links and concurrent replacement.
Successful parsing or viewing is not peak-memory evidence.

## Rendering resources and publication

Keep `SnapApp` host-neutral.
The eframe `EguiApp` wrapper owns `EguiRenderState`, including texture handles, RT-DOSE overlays, secondary-view cache keys and `RenderBufferPool` scratch.
A monotonic RITK visual revision identifies presentation-invalidating transitions.
The shell records the revision represented by its resources, releases stale textures/overlays on change, and preserves reusable scratch capacity.
Other hosts construct the same viewer state and consume RITK presentation values; they do not acquire eframe texture ownership or separate clinical state machines.
Before GPU allocation/submission, compute scalar-buffer size and compare both device buffer limits.
Unsupported requests select the existing CPU projection path with an explicit diagnostic identifying the limit and shape.
Distinguish pending, ready, unsupported and failed requests: pending readback keeps projection dirty, never masquerades as a completed frame or triggers CPU selection.
Surface submission/readback failures; CPU recovery applies to terminal unsupported/failed states.
Collect only readback matching the current request.
Changes to volume, window/level, presentation, colormap, opacity or extent invalidate stale frames, pending/mapped staging and affected buffers.
Do not block the render thread awaiting readback.
Reuse compatible buffers and preserve RITK's scalar/RGB, grayscale, physical placement, transform and projection semantics across hosts.

## Acceptance and comparisons

Fix opening/display defects before using the previous shell as a differential oracle: agreement with a defective baseline does not establish correctness.
Use synthetic studies with known decoded values, frame identities and physical landmarks; pin transfer syntax, photometric interpretation and frame organization per host.
Assert ordering, anisotropy, orientation, RGB and admitted grayscale behavior independently.
Run native and actual browser/desktop input traces under finite committed budgets, asserting both state and pixels through navigation, switching, cancellation, failure and close/reopen.
Untrusted DOM events cannot establish physical input acceptance.
Inspect rendered images and interaction outcomes after numeric checks.
The manual receives actual opening, selection, three-view and rejection/recovery captures when those workflows run; no missing fixture or unavailable GPU counts as passing coverage.
Compare memory and latency only under matched inputs, workloads and host conditions, using the orthogonal physical-spacing fixture rather than unequal shell chrome or MIP layouts.
Require saved-study capture and semantic-surface provenance, matched logical/physical extents with DPI recorded, and no post-capture pixel rescaling.
Record executable/provider revisions and distinguish local execution from packaging-only evidence.
Do not infer speed, memory use or completed migration from architecture, listener cleanup, canvas resemblance, successful compilation, screenshots alone or historical test counts.

## Alternatives and consequences

Keeping egui indefinitely contradicts the selected framework; removing it before replacement acceptance loses behavior.
Reimplementing DICOM in Métis duplicates its owner.
All three alternatives are rejected.
Selecting the largest acquisition or retaining folder-plus-UID instead of exact discovery membership cannot preserve user intent and is rejected.
The migration changes host boundaries and can require explicit public caller/persisted-format migration; it does not itself authorize a release or claim improved performance.
Changed requirements or independent evidence that the stated ownership/cutover contract cannot preserve required viewer behavior require revising this decision; delivery progress alone does not weaken it.
