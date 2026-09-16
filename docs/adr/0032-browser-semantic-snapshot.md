# ADR 0032: RITK browser semantic snapshot

Status: Accepted

Date: 2026-09-11

Driver: [RITK-SNAP-METIS-001](../../backlog.md#RITK-SNAP-METIS-001).

## Context

The Métis browser runner can capture generic canvas pixels and trusted input,
but it must not interpret DICOM bytes or clinical display state. A RITK-owned
browser workflow therefore needs a stable semantic surface that a driver can
assert after a load or navigation action. The existing canvas pixels alone do
not identify the selected axis, slice bounds, or whether a frame was actually
presented.

## Decision

RITK publishes a bounded semantic snapshot on each RITK canvas through
`data-ritk-*` attributes:

| Attribute | Values |
| --- | --- |
| `data-ritk-load-state` | `empty` or `ready` |
| `data-ritk-frame-state` | `empty` or `presented` |
| `data-ritk-axis` | `0` axial, `1` coronal, `2` sagittal |
| `data-ritk-slice-index` | zero-based decimal index |
| `data-ritk-slice-count` | positive decimal count |
| `data-ritk-frame-width` / `data-ritk-frame-height` | presented pixels, or `0` when empty |
| `data-ritk-cine-fps` | finite decimal rate from `1` through `60` |

The snapshot is derived by a pure RITK value and is written by the RITK
browser viewer after presentation. The viewer caches the last value so an
idle frame does not rewrite the DOM. It contains no path, patient identifier,
DICOM tag, status message, or pixel data. Métis remains a format-neutral host;
the consumer owns the meaning and assertions of these attributes.

## Rejected alternative

Adding DICOM selectors or clinical assertions to Métis would move RITK domain
state into the GUI host and would make the browser runner unusable for other
consumers. Publishing the full human-readable status message would expose
unstable text and could leak path or patient data. Requiring a JavaScript
callback would add an unbounded application-owned bridge where a bounded DOM
snapshot is sufficient.

## Verification

Revision 2026-09-16: the browser canvas publishes
`data-ritk-display-aspect` and applies that width-to-height ratio to its CSS
rectangle. RITK derives it from the rendered voxel dimensions and the loaded
volume's axis-specific sample distances. Backing RGBA dimensions remain
unchanged. The saved MRI fixture has 0.5 mm in-plane sampling and approximately
2.5 mm slice spacing: its coronal and sagittal extents are approximately
256 by 235 mm, rather than the raw 512 by 94 pixel ratio. Native geometry
tests and the browser trace check this distinction; pixel hashes alone cannot
establish correct physical display proportions.

The gallery permits styles only from its own origin. Browser reproduction
rejected the first implementation's complete `style` attribute: the declared
ratio was correct, but the CSS rectangle retained the raw pixel aspect, and
the new trace validator failed. Publication therefore uses Moirai's individual
CSS property setter for width, height and aspect ratio. The gallery's content
security policy remains unchanged.

Input mapping uses the canvas's measured CSS dimensions at each nonempty
event batch. Dividing those dimensions by the backing frame dimensions gives
the displayed texel sizes used by `ViewerViewport`. This keeps pointer and
wheel bounds aligned with the physical display, including after resize. A
unit texel scale incorrectly rejected wheel positions below row 94 in the
corrected coronal and sagittal canvases.

This mapping covers the gallery's borderless, unpadded canvases without CSS
transforms on them or their ancestors. Custom embeddings with borders, padding
or transforms need local content-box geometry: `offsetX/Y` ignore transforms,
whereas the browser's bounding rectangle includes them. That broader embedding
contract is tracked as `RITK-BROWSER-LOCAL-BOX-001`; it is not established by
the saved-gallery capture.

Revision 2026-09-15: `data-ritk-frame-generation` counts newly rendered frames
successfully uploaded to each canvas. The counter advances after presentation,
including repaint-triggered cache invalidation, and excludes cached uploads on
ordinary animation frames. It is checked for overflow and contains no study
metadata. The focused rate trace uses this counter to distinguish an effective
rate action from a suppressed repeated keydown; it does not measure display
refresh frequency or prove that the compositor displayed a frame.

The RITK semantic value has unit tests for empty and presented states,
including axis, slice and frame-dimension preservation. The WASM build and
warning-denied Clippy cover the DOM publication path. The browser manual and
Metis consumer trace use the attributes only for RITK-owned viewer assertions;
pixel captures remain the visual oracle and the semantic snapshot does not
claim clinical correctness by itself.
