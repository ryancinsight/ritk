# ADR 0043: Browser viewport transform

Status: Accepted

Revision: 2026-11-18 — pan in captured backing-pixel coordinates rather than
through the source-image inverse.

Delivery: [pan correction](https://github.com/ryancinsight/ritk/pull/992).

## Context

RITK's browser presentation already rendered real DICOM slices through the
format-neutral Métis canvas boundary. Ctrl/Command-wheel zoom and the Pan tool
updated `SnapApp`, but the browser renderer still uploaded the untransformed
slice. A pointer therefore acted on pixels different from those shown to the
user. A second browser-specific raster sampling policy would create host drift.

## Decision

RITK applies one bounded pixel-space viewport transform after slice rendering
and before the browser canvas upload. The output keeps the source dimensions
and uses nearest-neighbour sampling. For an output pixel centre `q`, frame
centre `c`, zoom `z` and pan `p`, the source sample is

```text
source = ((q - c - p) / z) + c
```

Samples outside the source frame are opaque black. Identity state (`z = 1`,
`p = (0, 0)`) leaves the existing RGBA storage in place. Non-identity frames
write into the existing reusable scratch buffer and swap it into the frame.
The browser `ViewerViewport` uses the same inverse equation in image-edge
coordinates before applying the existing orientation transform; events over a
panned black edge are ignored. The projection canvas remains display-only and
is not coupled to the orthogonal viewport state.

The transform stays in RITK. Métis receives only the resulting bounded RGBA
frame and continues to own canvas, event and host lifecycle concerns. CSS or
JavaScript canvas transforms are not used because they would separate the
displayed pixels from the RITK pointer coordinate contract.

## Pan coordinates

Normalized events keep their existing fractional coordinates. An explicit
navigation basis maps display extent `d` to backing extent `b` using `b / d`
componentwise. The production display extent is `[1,1]`. Métis `PointerDrag`
captures this scale and the raw origin at press; later resize, zoom, and
orientation changes do not reinterpret the anchor.

This intentionally corrects inverse-zoom/rotation coupling: in a
100-pixel-wide backing frame, fraction movement `0.5 → 0.6` pans ten backing
pixels at zoom 0.5, 1, or 2, under every orientation. The former inverse-image
path produced five pixels at zoom 2. Tests check both state and the rendered
landmark shift. A mid-gesture resize retains the press basis. These are
backing-pixel units, without a CSS-pixel sensitivity claim.

Picking still uses the raster and orientation inverse. DICOM measurements
remain in source/patient coordinates. Checked per-update rejection follows
[ADR 0028](0028-transformed-viewport-coordinate-fidelity.md), preserving the
last valid pan and allowing later safe movement and release. Zoom and
window/level drag sensitivity remain separate correction work.

## Alternatives

1. Keep the transform in a browser CSS transform. Rejected because RITK's
   pointer mapping would need a second browser-only coordinate model and pixel
   capture would not observe the transformed raster.
2. Apply zoom and pan only in the input adapter. Rejected because state would
   change without changing the presented image.
3. Duplicate the native compositor in the browser module. Rejected because
   the viewport equation is a shared RITK policy and duplicated sampling would
   drift across hosts.

## Invariants and failure modes

- Frame dimensions, physical display spacing and RGBA channel order are
  unchanged by the transform.
- Zoom and pan are finite; zoom must be positive. Invalid values return a
  typed presentation error before storage is swapped.
- Identity output is byte-preserving and non-identity output has bounded
  storage proportional to one frame.
- Black out-of-bounds pixels cannot create pointer annotations or measurements.
- DICOM decoding, window/level, colormap, geometry and clinical semantics
  remain RITK-owned; Métis remains format-neutral.

## Verification

The viewport pixel tests cover identity byte preservation, centred zoom,
positive pan with black exposed edges, invalid inputs and the storage reuse
path. Action-adapter tests prove that a transformed browser click resolves to
the source pixel shown by the transform and that a click over a panned black
edge is ignored. The locked `ritk-snap` library suite, strict Clippy and the
WASM checks are run against the delivery revision; the existing real 94-file
MRI replay remains the visual DICOM oracle.
