# ADR 0026: Viewer presentation migration to Métis

Status: Accepted

Date: 2026-09-06

Driver: [RITK-SNAP-METIS-001](../../backlog.md#RITK-SNAP-METIS-001).

Revision 2026-09-08: [RITK-SNAP-RESOURCES-001](../../backlog.md#RITK-SNAP-RESOURCES-001)
now has a `ritk-dicom` structural Part 10 preflight backed by the existing
Consus `ParseBudget`. All selected, exact-member, SCP, named-byte, and
DICOMDIR-index reads route through it before dicom-rs object construction. The
tested syntax set includes implicit little-endian, explicit little-endian,
explicit big-endian, and encapsulated pixel data. Deflated datasets remain an
explicit unsupported case because the locked dicom-rs registry has no bounded
deflate decoder. `DicomReadBudget` now separates parser, retained-study, and
decoded-workspace ceilings; retained bytes are charged before storage and the
loader rejects the planned peak frame/resample/volume workspace before
allocation. Budgeted reads compare the opened handle with resolved path
metadata and consume that handle, while scanned slices retain the validated
bytes for later decode. The implementation detects replacement during the
resolution/inspection window and rejects a final symlink/reparse point on Unix
and Windows respectively. At that revision, parent-directory traversal through
directory handles remained a platform boundary; complete DICOMDIR record-tree
validation stays in RITK-SNAP-DIRECTORY-001.

Revision 2026-09-09: [RITK-SNAP-RESOURCES-001](../../backlog.md#RITK-SNAP-RESOURCES-001)
closes the parent-directory boundary through the owning Moirai PAL filesystem
layer. `moirai_pal::fs::open_file_within_root` validates normal relative
components, walks from directory handles with Unix `openat` or Windows
relative `NtCreateFile`, rejects traversal and link components, and returns the
handle that the caller reads. WebAssembly reports an explicit unsupported
error. `ritk-dicom::read_file_within_root_with_budget` composes that handle with
the existing Consus `ParseBudget`; DICOM parsing and member policy remain in
RITK. Native and Unix-target gates cover the consumer and the Moirai API covers
the handle walk with value-semantic link and traversal tests.

Revision 2026-09-10: [RITK-SNAP-METIS-001](../../backlog.md#RITK-SNAP-METIS-001)
completes the pinned presentation inventory. The current RITK shell is the
`eframe::App` implementation in `crates/ritk-snap/src/app/state.rs`, launched
by `crates/ritk-snap/src/launch.rs` for native and browser targets. Its
format-aware state, loaders, renderers and DICOM workflows remain in RITK.
The inspected source snapshots are RITK
`8e53327c53d14afcec64a92b563e6b48db85eb15` and Métis
`2711454493747d373e4b09ac4e42e6b828a1e18e`; both provider trees were clean at
inventory time. The inventory below records the exact replacement seams and
the host gaps that must close before the shell can be removed.

Revision 2026-09-10 (first presentation increment): RITK now exposes
`ritk_snap::presentation::PresentationFrame`, a validated row-major RGBA value
constructed from the existing RITK `SliceRenderer` output. The Windows adapter
`run_native_frame` converts that value to a Métis `Framebuffer`, presents it on
the real hidden native surface, and closes the host after one event batch.
Focused nextest coverage proves frame dimensions and storage validation,
RITK grayscale display semantics, channel transfer, and the native
present/close transition. No DICOM parser, metadata, path, or geometry state
crosses into the Métis dependency. The browser handoff, viewer action dispatch,
three-view presentation, and GPU path remain open gaps in this migration.

Revision 2026-09-10 (native event increment): RITK now translates the bounded
Moirai `WindowEvent` batch into the public format-neutral
`PresentationEvent` contract. Pointer buttons, coordinates, key repeat state,
Unicode text, IME phases, resize, DPI, focus and lifecycle values are preserved
with provider limits and allocation errors surfaced. The native presentation
probe observes that translation during the real hidden host run; the remaining
work is dispatching those events into `SnapApp` actions without reintroducing
egui carriers or moving DICOM state into Métis.

Revision 2026-09-10 (viewer action increment): RITK now reduces a bounded
`PresentationEvent` batch through `PresentationDispatcher` into the typed
`ViewerAction` contract. Press state is held in a fixed button array; movement
produces checked deltas and deterministic multi-button drag actions; release
classifies clicks versus drags; focus loss emits cancellation actions; and
malformed sequences, coordinate overflow, overlong composition and allocation
failures leave dispatcher state unchanged. Keyboard, text, resize and DPI
values remain format-neutral. The action contract closes the event-to-viewer
boundary.

Revision 2026-09-10 (viewer action adapter increment): RITK now maps the
validated viewport placement into `ViewerViewport` and applies reduced actions
to the existing `SnapApp` transitions. The eframe shell feeds its bounded
pointer response into the same presentation event dispatcher used by native
and browser hosts, so pan, zoom, window/level, labels, measurements, linked
cursor updates, keyboard tool selection, slice navigation and focus
cancellation share one RITK-owned transition path. `ImagePoint` and
`ViewportOffset` remove egui point values from in-progress tool state. Wheel
modifier semantics, browser handoff, three-view presentation, GPU
presentation and complete application-window capture evidence remain open
increments. DICOM parsing, decoded volume state and clinical semantics remain
in RITK; no DICOM value crosses the Métis seam.

Revision 2026-09-10 (viewer action correction): the adapter now preserves
multi-click measurement anchors across click releases, accumulates repaint
requests across one event batch, and clears both reducer and viewer gesture
state when a host loses its final pointer coordinate. Presentation coordinates
and checked deltas use finite display-pixel `f64` values so browser coordinates
and native `i32` positions reach the format-neutral boundary without rounding.
The RITK package gate passes 774/774 nextest tests, strict Clippy, formatting and
locked package checks; the correction remains inside RITK and does not move
DICOM parsing, volume state or clinical semantics into Métis.

Revision 2026-09-10 (viewport transform correction): the viewport now subtracts
its display origin and applies the source transform in `f64` before narrowing
to the viewer's image-space representation. This preserves one-pixel mapping
when a native surface coordinate is above `f32` integer precision. The RITK
package gate passes 775/775 nextest tests; DICOM parsing, volume state and
clinical semantics remain in RITK.

Revision 2026-09-10 (consumer mapping correction): label painting, linked
cursor updates, and pointer-intensity sampling now consume the mapped
format-neutral image point directly. They no longer reconstruct image
coordinates from a narrowed presentation point. The RITK package gate passes
776/776 nextest tests; DICOM parsing, volume state and clinical semantics
remain in RITK.

Revision 2026-09-10 (wheel and modifier ownership correction): RITK now carries
the provider-neutral `PresentationModifiers` and `PointerWheel` event values,
reduces finite wheel deltas into `ViewerAction::PointerWheel`, and applies the
existing Ctrl/Command zoom and plain vertical slice-step policies in the RITK
action adapter. The native producer translates Moirai's `ModifierState`; the
egui producer feeds the same event path. Métis only transports the modifier
snapshot through its native surface. RITK owns all wheel behavior, DICOM
opening, decoded volume state and clinical semantics; no DICOM value or parser
crosses the Métis seam. The focused action, native translation and adapter
tests are the acceptance oracle for this increment.

Revision 2026-09-10 (interactive native session): RITK now hosts one complete
Windows viewer session through Métis's `NativeApplication` loop. The session
opens a selected study with the existing RITK loader, renders the active
orthogonal slice, maps it into a bounded Métis `Framebuffer`, translates every
bounded native batch, and applies the resulting actions to RITK viewer state.
Resize and minimize preserve the framebuffer contract, nonzero DPI is retained,
focus loss cancels pointer gestures, and close or destruction records terminal
state. A hidden capture run exits after its first idle batch and writes the
validated RITK source frame, so capture is finite and does not wait for user
input. Session tests assert slice changes, resize/minimize behavior, DPI
rejection, focus cancellation, close, and capture completion. The command is
`ritk-snap PATH --metis-native`; the native host owns only window, event and
framebuffer lifecycle. Browser handoff, the three-view composition, GPU upload,
full application-window goldens and packaged installer artifacts remain open.
The public `AppLaunchOptions` struct gains `metis_native`, so downstream
struct literals must add the field; this increment is classified as a breaking
public change and follows the migration note in
`docs/migration_selected_dicom.md`.

Revision 2026-09-10 (native interaction rendering correction): the native
compositor now applies RITK's pan offset when placing each orthogonal frame.
Consequently a primary-button drag changes both the RITK interaction state and
the presented Métis framebuffer; the native-session regression test checks both
observable values. The offset remains viewer state and no DICOM value crosses
the host boundary.

The native session also applies keyboard slice navigation through the same
RITK action adapter. A page-down event advances the active slice and triggers a
new composed framebuffer; the regression test checks the state and pixel
results together.

Revision 2026-09-10 (browser handoff increment): the WASM launcher mounts
Métis's generic HTML5/CSS host before starting the eframe canvas runner. The
RITK `browser_input` adapter drains the host's bounded named-byte batch into the
existing `egui::DroppedFile` carrier, after which RITK's dropped-input policy
classifies and loads DICOM bytes through the canonical scanner and series
loader. The adapter has no DICOM detection, parser, path, or viewer state. A
locked native check and the Atlas-overlay wasm32 compile/Clippy run cover the
consumer boundary. The standalone locked wasm32 check and warning-denied
Clippy now pass after the merged [Mnemosyne #141](https://github.com/ryancinsight/Mnemosyne/pull/141),
[Coeus #393](https://github.com/ryancinsight/Coeus/pull/393), [Apollo
#386](https://github.com/ryancinsight/apollo/pull/386), and [Leto
#187](https://github.com/ryancinsight/leto/pull/187) portability fixes. A
browser runtime bundle and visual capture remain open because this checkout
does not include a wasm-bindgen packaging tool.

Revision 2026-09-10 (application packaging increment): RITK now carries the
root application manifest at `metis.json`. The manifest declares
one `ritk-snap` Cargo binary and no DICOM or patient-data resources. Métis's
manifest-driven `package` command therefore produces the RITK executable,
inventory, and Windows per-user MSI while RITK retains DICOM parsing, decoded
volume state, geometry, and clinical display semantics. The package workflow
is a release artifact path; it does not grant registry publication, signing, or
patient-data inclusion. The local Windows x64 smoke exited 0 with 793/793
RITK tests already green; `inventory.json` recorded a 24,283,136-byte
`ritk-snap.exe` with SHA-256
`b03b940927e3c2f86f4b4e773a0f0235c6e7e5ac6728c60753825f42a9ab0bb7` and a
9,273,344-byte `org.ritk.snap.msi` with SHA-256
`5f4a7d434d934be29986c8db92a0a9cbe99b3596e3b5e34da112ff9a0a93c0bc`;
`ritk-snap.exe --help` exited 0. The committed
`.github/workflows/metis-package.yml` reproduces this package path on a
manual Windows dispatch with the `2e21146c6a9de73666396705c25dfa7555eb172c`
revision recorded in `Cargo.lock`; hosted artifact collection is evidence only
and does not publish or sign the release.

Revision 2026-09-10 (viewer load-task increment): pending primary and
secondary `VolumeInput` requests now run through one bounded Moirai blocking
task per target. RITK assigns checked generations and a cooperative
cancellation token; superseded, closed, or cancelled tasks cannot publish a
`LoadedVolume`, status, or frame. Current-load failures update the status while
retaining the previous study. The task bridge owns no DICOM logic and Métis is
unchanged.

Revision 2026-09-11 (browser canvas adapter increment): `metis-web` now exposes
the format-neutral `CanvasFrame` and `CanvasSurface` seam over Moirai's bounded
HTML5 canvas provider. RITK's `WebCanvasPresenter` implements that seam for the
existing `PresentationFrame`, so a future browser viewer can present RITK-owned
RGBA pixels without importing DICOM state into Métis or copying the source
frame in the Rust host. The current `start_web` entrypoint still launches the
eframe canvas while the complete browser viewer migration and visual capture
remain open; this increment proves only the typed consumer boundary.

Revision 2026-09-11 (direct browser canvas workflow increment): RITK now
exports `start_web_canvas` and `stop_web_canvas` for a Métis-owned HTML5 canvas.
The workflow mounts the generic Métis host, drains its bounded named-byte batch
through the shared RITK dropped-input reducer, loads DICOM bytes with RITK's
existing series loader, renders the selected RITK slice, and presents the
borrowed `PresentationFrame` through Moirai's browser canvas provider. The
reducer test opens a synthetic Part 10 study from the same named-byte action;
WASM Clippy covers the browser task and presenter. `start_web` remains the
eframe shell, and browser pointer actions, orthogonal composition, GPU upload,
and runtime visual capture remain open. No DICOM parser, metadata, geometry, or
viewer state is implemented in Métis.

Revision 2026-09-11 (WASM packaging increment): `ritk-snap` now declares a
`cdylib` library target so the browser artifact is the actual RITK library,
not the native-only binary artifact. The lockfile advances the Mnemosyne
consumer to merged [PR #143](https://github.com/ryancinsight/Mnemosyne/pull/143)
(`be6efcb8`), whose host-environment FFI gate removes the final `getenv` link
failure on `wasm32-unknown-unknown`. The release build and
`wasm-bindgen 0.2.128 --target web` package pass against the live Atlas
overlay; the locked commands are the standalone-checkout and CI acceptance
path. They export `start_web` and `start_web_canvas`; browser pointer actions,
three-view composition, GPU upload, and runtime visual capture remain open.
DICOM ownership stays in RITK.

Revision 2026-09-08: [RITK-SNAP-DIRECTORY-001](../../backlog.md#RITK-SNAP-DIRECTORY-001)
now validates the Explicit VR Little Endian DICOMDIR record sequence before
membership is admitted. RecordInUseFlag, next/lower offsets, incoming-link
uniqueness, cycles, and first/last root-chain termination are checked; only
reachable active IMAGE records are followed. Each admitted reference is opened
through the bounded no-follow path and its SOP class, SOP instance, and transfer
syntax are compared with the record. Synthetic selection tests cover inactive,
unreachable, malformed-link, identity-mismatch, and final-symlink cases.

Revision 2026-09-09: [RITK-SNAP-DICOM-SUBSTRATE-001](../../backlog.md#RITK-SNAP-DICOM-SUBSTRATE-001)
reaffirms the ownership boundary after an attempted GUI-side opening slice was
reviewed. RITK's existing `ritk-io` named-byte scanner, bounded series loader,
image/metadata result, tests, and visual workflow remain the single DICOM
implementation. Métis receives validated RITK results through a presentation
adapter; it does not define a parallel DICOM volume or decoder.

Revision 2026-09-06: [RITK-SNAP-OPEN-001](../../backlog.md#RITK-SNAP-OPEN-001)
requires explicit acquisition selection. Inspection of `d3cbd8eb` finds that
the sidebar discards discovered file membership, the reader selects a majority
or merges tied series, and metadata accumulates before that selection. The
following contract replaces those behaviors before using the viewer as an oracle.

Verification revision: the local Skull index names 304 active IMAGE references
but contains 303 files; `DICOM/I303` is absent. Requiring successful loading
depended on dropping that reference. Complete synthetic indexed studies replace
those two success oracles; missing references remain rejection tests. The
directory record contract is defined by
[PS3.3 F.3.2.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_F.3.2.2.html).

## Intent and evidence

The user selects Métis as the future presentation framework for `ritk-snap`,
with correct DICOM opening and viewing as required acceptance. At source revision
`341228ee3861c5e9a091dcf58de500510f948505`, native launch uses eframe/egui and
browser launch uses eframe's canvas runner. No Tauri dependency was found in
the viewer manifest, workspace manifest or lockfile. This record establishes
the migration target; it does not claim that migration or runtime validation ran.

## Decision

Replace the viewer shell with Métis when its required real-host capabilities
and the viewer conformance inventory pass. RITK continues to own DICOM parsing,
codec selection, study/series/frame identity, physical geometry, medical display
transforms, viewer navigation and analysis. Métis owns presentation, input,
window/browser lifecycle and scoped host access. Reuse Iris for its existing
visualization contracts and Moirai for execution and transport.

Extract GUI-specific carriers from the existing viewer state/render boundary
where required by the migration; do not move DICOM logic into Métis or duplicate
RITK decoders. Framework gaps are implemented in Métis and reusable runtime gaps
in Moirai. The [Métis V09 contract](../../../metis/docs/VERIFICATION.md#V09)
owns the shared application demonstration requirements.

The first acceptance work is a reproducible opening/display baseline. The board
records observed file-dispatch, series-selection, multiframe, RGB, grayscale and
fixture-absence gaps. Fix them before using the current viewer as a differential
oracle. Working same-input comparison against egui is useful evidence, but
agreement with a defective baseline cannot establish correctness.

## Migration and failure boundaries

### Acquisition selection

RITK owns one discovery descriptor carrying series identity and exact member
paths. The viewer retains this descriptor through primary and secondary
selection, highlighting and pending loads. A folder is a discovery location;
it is not an acquisition identifier. A selected single file supplies its UID
before matching neighbouring instances. Directory and byte-batch loads without
selection require one unambiguous image series; population and dimensions do
not authorize choosing a different series. Validate image identity before
accumulating series metadata or assembling geometry.

An explicitly selected DICOMDIR, or a directory's present DICOMDIR, owns its
referenced file set. A malformed index, missing reference, absolute component,
parent traversal or reference escaping the file-set root fails the load.
Invalid index contents cannot trigger an unrelated flat-directory scan.
Discovery and decoding use the same resolved member set. Failure preserves
the previously loaded study and surfaces the reason to the user.

Session persistence records selection intent, not trusted decoded metadata.
Restoring a selected series reopens its exact recorded members and verifies
the recorded UID before replacing the current volume. Older path-based
sessions remain readable at the persistence boundary and follow the corrected
path-opening rules; ambiguous old directory sources require explicit selection.
Changes to public viewer selection carriers require downstream callers to use
the descriptor directly; no folder-only forwarding API remains.
The [selection migration guide](../migration_selected_dicom.md) enumerates
the changed caller surfaces and persisted-format transition.

Keeping folder plus UID while rediscovering on every selection loses the
already discovered DICOMDIR membership. Choosing the largest acquisition
cannot establish user intent. Both alternatives are rejected. Tests must
distinguish two same-folder series, selected minority data, reversed input
order and metadata from an unselected first file.

### Input and host boundaries

Inventory the actual application at a pinned revision: opening/series browser,
PACS workflows, orthogonal and projection views, window/level, pan/zoom/cursor,
overlays, measurements/annotations, filters/segmentation, persistence/export,
shortcuts, errors and recovery. Preserve required behavior in independently
verified vertical increments; a complete shell cutover removes superseded
framework dependencies and all affected call sites. Keep the current working
shell until the replacement meets its acceptance; no forwarding layer ships.

Use small synthetic studies with known voxel values, frame identities and
physical landmarks. Pin transfer syntax, photometric interpretation and frame
organization coverage per host; an unsupported required case remains an open
gap. Browser file handles/bytes do not imply native filesystem or PACS authority.
Cancellation, closing or failed replacement must not publish stale study data.
Bound decode tasks, buffers and pending requests; handle malformed input without
unbounded allocation. Public screenshots, fixtures and diagnostics contain no
patient data. No remote upload or service is introduced by this decision.

For acquisition selection, the assets are the selected study's identity,
pixels and geometry, and local files outside an imported file set. Untrusted
Part 10 bytes cross the parser boundary; index references cross the filesystem
boundary; persisted selectors cross the restore boundary. Required mitigations
are explicit UID/member validation, bounded parser behavior, reference rejection
and validation again on restore. A path check alone does not establish safety
against concurrent filesystem replacement; claims about that race require
handle-based access and independent verification at the filesystem boundary.

### Ingestion resource contract

[RITK-SNAP-RESOURCES-001](../../backlog.md#RITK-SNAP-RESOURCES-001) owns the
remaining ingestion limits. At the current lock, `ritk-dicom` delegates parsing
to dicom-rs 0.10. Its file-meta and value readers allocate from declared lengths
before reading those values, and nested objects recurse without a depth policy.
Limiting encoded file size alone therefore does not bound parser allocation.

Use the existing Consus `ParseBudget` dimensions for encoded bytes, element
count and nesting at the `ritk-dicom` parser boundary; `ritk-io` owns cumulative
retained-study accounting. Validate remaining input and allocation allowances
before materializing values, including meta headers, sequences, fragments and
deflated datasets. Decoded pixels require their own checked shape and allocation
budget. Consus's current bounded reader limits logical length, not all reserved
capacity, so its successful return is not a peak-memory oracle. Test allocation
requests at boundary values and malformed declared lengths.

Root-confined open operations belong in Moirai's filesystem layer. The merged
`moirai-pal::fs::open_file_within_root` contract resolves every component from
the selected root handle and returns that handle for reading, so RITK no longer
recreates platform traversal. Retaining the exact validated Part 10 bytes and
the RITK parser budgets remains a consumer responsibility; the filesystem race
is covered by the Moirai implementation and its platform tests.

### Presentation inventory and replacement seams

The inventory is a contract map, not a second implementation. Each row keeps
the RITK owner explicit and names the Métis capability required at the host
boundary.

| Current RITK surface | RITK responsibility retained | Métis/Moirai seam | Gap before cutover |
| --- | --- | --- | --- |
| `SnapApp::update` in `app/state.rs`; `run_app_with_options` and `start_web` in `launch.rs` | Frame ordering, load/recovery decisions, viewer state and DICOM policy | Windows `metis_platform::native::NativeSurface::{poll_events,wait_events,present,close}`; browser `metis-web::{metis_start,metis_stop,take_file_drop}` | The RITK session connects the Windows host loop for three views and the WASM launcher drains browser file batches. A reusable loop for arbitrary apps, GPU presentation, packaging, and browser runtime visual evidence remain; `NativeSurface` is Windows-only. |
| `SnapApp` fields and `app/*_ops.rs` transitions | Volume identity, series selection, navigation, measurements, overlays, PACS and persistence | `ritk_snap::presentation::PresentationDispatcher` and `ViewerAction`; Métis IPC/fragment actions are transport seams only | RITK's `ViewerViewport` adapter applies the actions on eframe and the first Windows Métis session. Browser host parity, stale-completion guards and multi-viewport dispatch remain before shell cutover; no DICOM state may cross into a Métis crate. |
| `egui::Context`, `RawInput`, `Event`, `DroppedFile`, and pointer handling in `ui/*` | Pointer/keyboard semantics are translated into RITK actions | `metis-platform::PlatformEvent`; native Moirai `WindowEvent`; browser `FileDropBatch`/`take_file_drop` | Browser file batches now reach the existing RITK classifier and loader. Native file-picker grants, browser runtime packaging, focus/text/IME parity and trusted native file ingress remain incomplete for the viewer. |
| `ToolState`'s `egui::Pos2` carriers in `tools/interaction/tool_state.rs` | In-progress pan, zoom, window/level and measurement coordinates | `ImagePoint` and `ViewportOffset` plus the format-neutral action contract | Closed in the adapter increment; transformed image coordinates and screen-space pan offsets retain their existing semantics. |
| `egui::ColorImage`, `TextureHandle`, `render::{slice_render,mip_vr,gpu_*}` | Scalar/RGB presentation, W/L, colormap, MPR, MIP/VR and GPU numerical behavior | `metis-ui-lang::RasterImage`, `DisplayList`, and `metis-platform::Framebuffer`; Iris remains the visualization contract | A bounded image upload/texture cache and GPU-capable presentation path must support three orthogonal views and projections without copying DICOM or replacing Iris render contracts. |
| `rfd::FileDialog` and `app/io_ops.rs` | User-selected paths, selected-study identity, export/session semantics | Moirai filesystem grants; Métis browser byte batches | Native dialog and browser byte-batch adapters must preserve exact selection intent and return typed failures to RITK. |
| `process_pending_loads`, `pacs_worker`, `tick_cine`, and repaint requests | Decode/PACS/cine scheduling, cancellation and result publication | Moirai bounded tasks and host pump; `metis-frontend::AsyncFrontendApp` is an IPC pattern | The RITK bridge uses per-target generations and cooperative cancellation; host close and supersession invalidate pending publication. |
| Menus, panels, overlays, annotations and accessibility behavior in `ui/*` | Medical labels, physical-coordinate overlays, measurements and actions | Métis UI language DOM/CSS layout plus format-neutral display commands | Widget, text, clipboard, context-menu, accessibility and overlay primitives need a conformance slice before porting the complete shell. |
| `CaptureApp` in `launch/capture.rs` (`ViewportCommand::Screenshot`, `Event::Screenshot`, completion and close) | Eframe application-window PNG capture, study-load requirement and failure reporting | Métis framebuffer readback plus host close/present result | The Windows Métis session now provides a finite source-frame capture; complete application-window readback and three-view capture remain before shell cutover. |
| `clap` binary options and eframe packaging in `main.rs` | RITK viewer arguments and capture workflow | `metis-cli` `init`, `dev`, `build`, `package`, `completions`; `package` produces the Windows MSI; `metis-platform` native surface | An application manifest and installer workflow must carry the RITK binary, assets and permissions as one distributable app; MSI installation is the packaged artifact workflow, not a separate `install` command. |

The initial implementation slices after this inventory are a format-neutral
RITK viewer host contract: accept a RITK-produced presentation frame, translate
typed host events, reduce them to viewer actions, and apply them to the
existing viewer transitions on a native Métis surface. The frame, event,
action-reducer, wheel policy, eframe adapter, three-view native composition,
first Windows interactive session, and browser byte handoff are complete.
GPU presentation, native packaging, browser runtime visual evidence, and
complete application-window capture remain before shell cutover. DICOM opening
remains exercised by RITK's existing file and byte loaders; Métis receives only
bounded format-neutral bytes and validated presentation data.

## Alternatives and validation

Retaining egui indefinitely contradicts the requested framework target. Removing
it before Métis can operate the viewer would lose behavior. Reimplementing DICOM
inside Métis would duplicate the existing format owner. These alternatives are
rejected; the complete host and DICOM acceptance gates determine cutover.

Run native and actual browser/desktop input traces under committed finite
budgets. Assert decoded values and geometry independently, then inspect the
rendered images and interaction outcomes. The user manual gains actual opening,
series selection, three-view display and rejection/recovery captures when those
Métis workflows run. Compare memory and latency only under matched fixtures and
host conditions; architecture choice alone establishes no improvement.
