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
bytes for later decode. The portable implementation detects replacement during
the resolution/inspection window; operating-system no-follow guarantees remain
a platform boundary, and complete DICOMDIR record-tree validation stays in
RITK-SNAP-DIRECTORY-001.

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

Root-confined open operations belong in Moirai's filesystem layer. Its current
path-based open/read APIs do not establish that contract. Retaining the exact
validated Part 10 bytes prevents scan/decode replacement but does not close the
earlier canonicalize/open race. These are required follow-on implementations,
not properties established by acquisition selection.

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
