# ADR 0028: Transformed viewport coordinate fidelity

Status: Accepted

Date: 2026-09-09

Driver: [RITK-SNAP-COORDINATES-001](../../backlog.md#RITK-SNAP-COORDINATES-001).

## Context

`ritk-snap` renders each source slice through a discrete flip/rotation
transform before it reaches the viewport texture. Cursor hit testing,
crosshairs, measurements, orientation labels, and RT overlays previously used
the untransformed source rectangle. A rotated or mirrored image could therefore
show a valid pixel while reporting a cursor, annotation, or anatomical edge at
the wrong location.

Physical annotation values are stored in `f32` fields for the existing viewer
serialization contract, while volume spacing is admitted as `f64`. Narrowing a
finite spacing can produce zero or infinity, and derived lengths or ROI areas
can overflow. Displaying those results would silently change the measurement's
meaning.

## Decision

`ViewTransform` is the single source of truth for source/display mapping. It
exposes continuous edge-coordinate forward and inverse maps in addition to the
pixel transform. A source slice of size `[width, height]` occupies the edge
domain `[0,width] × [0,height]`; flips are applied before the clockwise
rotation, and quarter turns swap the output dimensions. Pixel centres therefore
map without a half-pixel offset.

Viewport interaction maps screen coordinates to output edge coordinates and
then through `output_to_source` before converting to source voxels. Crosshairs,
measurement annotations, label painting, RT-STRUCT contours, RT-DOSE cells,
and orientation labels project source coordinates through `source_to_output`.
The same transformed output dimensions and physical sampling distances drive
fit layout and hit testing, so the displayed image and its interaction
rectangle cannot diverge.

All physical measurement entry points validate positive finite spacing and its
`f32` representation. Checked length and ROI constructors reject non-finite
derived values with `MeasurementError`; pointer handlers surface a status and
do not append an invalid annotation. The older unchecked calculation methods
remain the arithmetic primitives used by callers that already own a validated
contract.

## Alternatives

Keeping separate transform formulas in each overlay is rejected because every
new orientation would need several synchronized changes and a missed path
would create a clinically misleading display. Storing screen coordinates in
annotations is rejected because screen layout, zoom, and host DPI are
presentation state; source voxel coordinates remain stable persisted data.
Clamping invalid physical values to zero or a finite fallback is rejected
because it hides a malformed or unrepresentable measurement.

## Verification

The transform suite checks continuous round trips and pixel-centre agreement
for all 16 flip/rotation combinations. The linked-cursor suite drives the
application pointer update path from projected crosshairs on all three axes
and every transform, then asserts the original voxel. Orientation-label tests
assert transformed edge slots. Measurement tests reject zero, non-finite,
unrepresentable spacing and non-finite derived values. Existing physical-aspect
and deterministic DICOM workflow captures remain the visual baseline; the
manual documents the transform controls and the source/display coordinate
contract.

## Revision history

- 2026-09-09: Initial decision and implementation for
  RITK-SNAP-COORDINATES-001.
