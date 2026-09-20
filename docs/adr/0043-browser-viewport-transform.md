# ADR 0043: Browser viewport transform

- Status: Accepted
- Date: 2026-09-20
- Driver: [RITK-BROWSER-VIEWPORT-001](../../backlog.md#RITK-BROWSER-VIEWPORT-001)

## Context

RITK's browser presentation already rendered real DICOM slices through the
format-neutral Métis canvas boundary. Ctrl/Command-wheel zoom and the Pan tool
updated `SnapApp`, but the browser renderer still uploaded the untransformed
slice. A pointer therefore acted on pixels different from those shown to the
user. The native session already had a validated zoom/pan policy, so a second
browser-specific policy would create host drift.

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
The browser `ViewerViewport` uses the same inverse equation before applying
the existing orientation transform; events over a panned black edge are
ignored. The projection canvas remains display-only and is not coupled to the
orthogonal viewport state.

The transform stays in RITK. Métis receives only the resulting bounded RGBA
frame and continues to own canvas, event and host lifecycle concerns. CSS or
JavaScript canvas transforms are not used because they would separate the
displayed pixels from the RITK pointer coordinate contract.

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
