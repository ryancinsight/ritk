# ADR 0026: Viewer presentation migration to Métis

Status: Accepted

Date: 2026-09-06

Driver: [RITK-SNAP-METIS-001](../../backlog.md#RITK-SNAP-METIS-001).

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
