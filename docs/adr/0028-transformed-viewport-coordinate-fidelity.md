# ADR 0028: Source picking and display navigation

Status: Accepted

Revision: 2026-11-18 — separate source-image picking from host navigation
because inverse-image deltas change pan distance and drag sensitivity.

Delivery: [navigation correction](https://github.com/ryancinsight/ritk/pull/991).

## Context

RITK transforms a source slice through flips and quarter turns before display.
Picking and annotations must follow that transform. Navigation instead acts
on the displayed image: a pan must follow the pointer regardless of image
fit, physical sample spacing, zoom, or orientation.

The prior navigation path reused the inverse image mapping. For a 100-pixel
image fitted to 200 display pixels, a 24-pixel pointer displacement became
12 image pixels, then a 12-pixel screen pan. At a quarter turn, the same
inverse also exchanged the drag axes. These are unit mismatches, distinct
from event batching or image-space measurement accuracy.

## Decision

`ViewTransform` owns continuous source/output edge maps. A source image of
size `[width,height]` occupies `[0,width] × [0,height]`; flips precede the
clockwise rotation, and quarter turns exchange output dimensions. Picking
uses the orientation inverse; annotations, crosshairs, clinical directions,
and RT overlays use the forward map. Persisted measurements remain in source
or patient coordinates, never host coordinates.

Navigation has a separate explicit coordinate basis on `ViewerViewport`.
Native Windows coordinates are client pixels; eframe coordinates are logical
display points. Both use one navigation unit per host coordinate. Browser
events are normalized fractions and use captured backing-raster dimensions
per fraction, as specified by [ADR 0043](0043-browser-viewport-transform.md).
The basis is independent of texel size, spacing, orientation, zoom, and pan.

On a navigation press over the image, RITK constructs a Métis
`metis_core::input::PointerDrag` from the raw host position and the validated
basis. Its displacement is `(current - origin) * units_per_coordinate`.
The origin and scales remain fixed for the gesture. Native resize or browser
frame replacement therefore cannot reinterpret an active anchor. Image tools
retain bounded source-image mapping; captured navigation can leave the image.

Métis owns finite coordinate/scale admission and displacement mechanics.
RITK owns pan, zoom, window/level sensitivity, source ownership, calibration,
and measurement values. No DICOM type or policy moves into Métis.

Navigation variants of public `ToolState` contain the captured drag rather
than an `ImagePoint` start. The public `pan_from_drag_delta` helper is removed;
all callers migrate together. The [migration guide](../migration_selected_dicom.md)
records this unreleased breaking change without a forwarding API or a release.

## Numeric and failure contract

Coordinate arithmetic stays in the host protocol's `f64`; RITK validates its
`f32` representation before crossing into the existing pan/zoom state.
Pan and zoom arithmetic remains `f32`; window/level retains its existing
`f64` formula and checked final narrowing. A candidate update validates every
component and final value before assigning any field.

An unrepresentable navigation update produces a visible status and tracing
diagnostic. It preserves the last valid display values and original anchor,
then allows later safe movement, cancellation, and release. Earlier valid
updates in the same batch remain applied. The malformed-event and wheel
preflight guarantees in [ADR 0026](0026-viewer-presentation-migration.md)
remain unchanged; this decision does not introduce a UI transaction framework.

Physical measurement admission continues to reject nonpositive, nonfinite,
or unrepresentable spacing and nonfinite derived values through typed errors.

## Rejected alternatives

An inverse-image navigation anchor remains dependent on fit and orientation.
A live recomputed display basis changes gesture meaning during resize.
Storing host positions in measurements makes persisted geometry depend on
window layout and DPI. A sibling re-export of the Métis input value obscures
dependency ownership; RITK depends directly on `metis-core`.
Rolling back every earlier UI action exceeds the existing input contract and
would require ownership of irreversible dialog and operating-system effects.

## Regression oracles

Native rendered image origins follow raw `[24,-12]` pixel movement at fits
0.5, 1, and 2 under all flip/rotation combinations. Vertical 24-pixel movement
from zoom 1 yields 0.76; horizontal movement leaves zoom unchanged.
Window center/width 40/400 with displacement `[24,-12]` and sensitivity 4
yields 88/496. Assertions use these display values rather than inverse-image
helpers. Browser nonidentity tests follow ADR 0043.

Split and combined event batches agree. Invalid updates preserve both fields
of a pan/window pair; subsequent safe movement and release remain usable.
Source picking, calibration, slice ownership, focus/capture loss, modifier
overrides, and suspended measurements retain their independent regression
cases. The separate eframe painting and true-press-origin change additionally
checks image/annotation/crosshair translation and picking within a fixed clip.
