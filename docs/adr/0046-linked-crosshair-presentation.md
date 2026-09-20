# ADR 0046: Linked crosshair presentation at the Métis host boundary

* Status: Accepted
* Item: [RITK-SNAP-METIS-CROSSHAIR-001](../../backlog.md#RITK-SNAP-METIS-CROSSHAIR-001)

## Context

RITK already keeps one linked MPR cursor in volume voxel order `[z, y, x]`.
The eframe shell draws its crosshair from that state, while the Métis browser
and native sessions previously exposed the cursor only through pointer
transitions. A consumer needs the same cursor to remain aligned with the
rendered pixels after flips and quarter-turns, including anisotropic studies.

## Decision

`PresentationSnapshot` carries the crosshair visibility flag, the linked voxel,
and the `ViewTransform` used to produce each presented slice. Browser canvases
publish these values as stable `data-ritk-*` attributes. The RITK gallery draws
two CSS lines over each canvas and maps the voxel through the published
orientation and frame dimensions. The Windows Métis session builds two
`DisplayCommand::DrawLine` commands per plane using the same RITK mapping and
clips them to the image rectangle.

The rendered RGBA frames remain unchanged. RITK owns the cursor, orientation,
and DICOM semantics; Métis supplies only the browser DOM or native display-list
surface. Browser button activation and native `X` key activation change only
crosshair visibility. Repeated native key-down events are ignored.

## Alternatives

* Recompute the cursor independently in each host. Rejected because the browser
  and native projections would drift from the RITK voxel mapping.
* Draw the lines into the RGBA frame. Rejected because it changes clinical
  pixels, prevents independent visibility, and makes the frame cache carry UI
  chrome.
* Move cursor state into Métis. Rejected because DICOM slice topology and
  linked voxel semantics belong to RITK.

## Evidence and limits

The native tests assert six crosshair display-list lines, orientation-aware
coordinate mapping, hidden-state emptiness, repaint behavior, and snapshot
state. Browser semantic tests and the trace validator assert typed visibility,
cursor, and orientation attributes. The gallery contract tests assert the
consumer control and CSS overlay surface. The actual MRI gallery and native
captures remain the visual evidence for non-black orthogonal planes; a future
hosted capture will add crosshair-on pixels to that gallery. OS accessibility
semantics for the visual lines remain outside this presentation increment.
