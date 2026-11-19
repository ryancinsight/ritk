# ADR 0043: Browser viewport transform

Status: Accepted

Revision: 2026-11-18 — retain normalized browser events while correcting
inverse-zoom and orientation coupling in navigation displacement.

Delivery: [navigation correction](https://github.com/ryancinsight/ritk/pull/991).

## Context

The browser host presents a bounded RGBA frame and reports pointer positions
as fractions of its displayed canvas. RITK owns raster zoom/pan and source
picking. Applying navigation through the picking inverse incorrectly divided
drag displacement by zoom and rotated its axes.

## Decision

RITK applies a pixel-space viewport transform after slice rendering and
before canvas upload. Output dimensions remain those of the source frame.
For output pixel center `q`, frame center `c`, zoom `z`, and pan `p`, the
nearest-neighbor source sample is:

```text
source = ((q - c - p) / z) + c
```

Outside samples are opaque black. Identity zoom/pan preserves the RGBA
storage; other transforms use the existing reusable scratch buffer and swap
it into the frame. Physical display spacing and channel order remain intact.
Zoom must be positive and all transform values finite; invalid values return
a typed error before storage is swapped.

Picking uses the inverse transform in image-edge coordinates, followed by
the orientation inverse. Black exposed edges cannot create measurements or
annotations. The projection canvas remains display-only.

Navigation uses an independent explicit basis from displayed host extent
`d` to backing-frame extent `b`: `units_per_coordinate = b / d` componentwise.
The production browser extent is `[1,1]`, so each normalized fraction spans
the corresponding backing dimension. The basis and raw press origin are
captured in Métis `PointerDrag` at press and remain fixed for that gesture.
They do not depend on zoom, pan, orientation, sample spacing, or a later
frame resize. This is a backing-pixel convention, not a CSS-pixel sensitivity
claim. Native and eframe units are defined in
[ADR 0028](0028-transformed-viewport-coordinate-fidelity.md).

This intentionally changes the defective drag response: in a 100-pixel-wide
backing frame, fraction movement `0.5 → 0.6` pans ten backing pixels at zoom
0.5, 1, or 2. At zoom 2 the former inverse-image path produced five pixels.
Zoom and window/level likewise use backing-pixel displacement independent
of orientation. Event normalization and source measurement coordinates keep
their existing meanings.

Métis owns canvas lifecycle, event delivery, and reusable displacement
mechanics. RITK owns raster sampling, zoom/pan/window policy, DICOM decoding,
clinical units, calibration, and annotations. Browser CSS transforms do not
replace the RITK raster operation.

## Rejected alternatives

CSS-only transforms separate captured raster pixels from RITK picking.
Updating only input state leaves the displayed image unchanged.
An inverse-image navigation delta couples display response to zoom and image
orientation. Recomputing a drag's scale from a live frame changes its meaning
during resize. A browser-specific duplicated raster compositor would drift
from the shared RITK sampling policy.

## Regression oracles and limits

Raster tests cover byte-preserving identity, centered zoom, positive pan and
black edges, invalid transforms, and scratch reuse. Picking tests resolve a
transformed source landmark and ignore exposed black edges.

Navigation tests use a 100×100 backing frame across zoom 0.5/1/2 and every
orientation. Horizontal fraction movement 0.1 moves a rendered landmark ten
backing pixels; vertical movement 0.1 scales zoom by 0.9. With window center
40, width 400, and sensitivity 4, horizontal movement 0.1 yields width 440
and unchanged center. A resize during the gesture retains the press basis.
Recoverable arithmetic rejection follows ADR 0028 and permits safe later
updates and release. Browser integration tests additionally establish the
actual browser-to-session event boundary for their tested host configuration.
