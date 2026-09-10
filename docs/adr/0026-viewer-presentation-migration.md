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
The inventory below records the exact replacement seams and the host gaps that
must close before the shell can be removed.

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
| `SnapApp::update` in `app/state.rs`; `run_app_with_options` and `start_web` in `launch.rs` | Frame ordering, load/recovery decisions, viewer state and DICOM policy | `metis-platform::NativeSurface::{poll_events,wait_events,present,close}`; `metis-web::{metis_start,metis_stop}` | A reusable application loop must connect host events, bounded repaint/present and shutdown for an arbitrary RITK app. |
| `SnapApp` fields and `app/*_ops.rs` transitions | Volume identity, series selection, navigation, measurements, overlays, PACS and persistence | RITK typed commands remain the model; Métis IPC/fragment actions are transport seams only | The viewer needs a typed command/event adapter with cancellation and stale-completion guards; no DICOM state may cross into a Métis crate. |
| `egui::Context`, `RawInput`, `Event`, `DroppedFile`, and pointer handling in `ui/*` | Pointer/keyboard semantics are translated into RITK actions | `metis-platform::PlatformEvent`; native Moirai `WindowEvent`; browser `FileDropBatch`/`take_file_drop` | Event normalization, file-picker grants, focus/text/IME and trusted native file ingress are incomplete for the viewer. |
| `egui::ColorImage`, `TextureHandle`, `render::{slice_render,mip_vr,gpu_*}` | Scalar/RGB presentation, W/L, colormap, MPR, MIP/VR and GPU numerical behavior | `metis-ui-lang::RasterImage`, `DisplayList`, and `metis-platform::Framebuffer`; Iris remains the visualization contract | A bounded image upload/texture cache and GPU-capable presentation path must support three orthogonal views and projections without copying DICOM or replacing Iris render contracts. |
| `rfd::FileDialog` and `app/io_ops.rs` | User-selected paths, selected-study identity, export/session semantics | Moirai filesystem grants; Métis browser byte batches | Native dialog and browser byte-batch adapters must preserve exact selection intent and return typed failures to RITK. |
| `process_pending_loads`, `pacs_worker`, `tick_cine`, and repaint requests | Decode/PACS/cine scheduling, cancellation and result publication | Moirai bounded tasks and host pump; `metis-frontend::AsyncFrontendApp` is an IPC pattern | A viewer-specific bounded task bridge is absent; it must never publish a stale or cancelled study. |
| Menus, panels, overlays, annotations and accessibility behavior in `ui/*` | Medical labels, physical-coordinate overlays, measurements and actions | Métis UI language DOM/CSS layout plus format-neutral display commands | Widget, text, clipboard, context-menu, accessibility and overlay primitives need a conformance slice before porting the complete shell. |
| `clap` binary options and eframe packaging in `main.rs` | RITK viewer arguments and capture workflow | `metis-cli` build/init/dev/install packaging and `metis-platform` native surface | An application manifest and installer workflow must carry the RITK binary, assets and permissions as one distributable app. |

The first implementation slice after this inventory is a format-neutral RITK
viewer host contract: accept a RITK-produced presentation frame and typed host
events on a native Métis surface, then exercise the same contract through the
browser handoff. It must prove a single slice before three-view and GPU
cutover. DICOM opening remains exercised by RITK's existing file and byte
loaders; Métis receives only validated presentation data and user actions.

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
