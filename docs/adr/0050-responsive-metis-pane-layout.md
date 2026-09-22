# ADR 0050: Responsive Métis pane layout

- Status: Accepted
- Date: 2026-09-21
- Item: [RITK-SNAP-METIS-RESPONSIVE-LAYOUT-001](../../backlog.md#RITK-SNAP-METIS-RESPONSIVE-LAYOUT-001)

## Context

The RITK viewer already carries validated physical spacing with every
`PresentationFrame`, but the Métis native surface currently has only a fixed
three-panel orthogonal composition or a fixed four-panel projection
composition. The browser entrypoints likewise require a fixed canvas count and
do not expose a responsive single/dual/four-pane policy. The host must not
interpret DICOM metadata or derive voxel geometry.

## Decision

RITK owns a small, host-neutral `PaneLayout` value. It maps a bounded surface
extent to one of three pane counts: one pane for narrow surfaces, two panes
for intermediate surfaces, and four panes for large surfaces. Pane roles are
stable: axial, coronal, sagittal, then an optional scalar projection. The
native compositor and browser surface consume the same role order. Each pane
uses the existing spacing-aware placement contract, which letterboxes the
frame inside its rectangle and never stretches physical dimensions.

The browser responsive entrypoint receives four named canvases. It hides
inactive roles through trusted DOM style properties and does not register
input listeners for the projection or inactive panes. The native entrypoint
selects the layout on each bounded resize/repaint and keeps inactive native
viewports non-interactive. Existing fixed entrypoints and the default
three-panel capture remain stable for compatibility fixtures.

## Alternatives

Reusing eframe's `LayoutMode` would couple the Métis host to the legacy shell
state and would not describe browser canvases. A second host-specific layout
implementation would allow native and browser geometry to diverge. Both are
rejected in favor of one RITK presentation contract.

## Failure modes and verification

Invalid or zero surface extents return a typed error before allocation.
Layout tests assert exact pane counts, disjoint rectangles and full coverage
of the usable surface after separators. Spacing tests use anisotropic values
and assert the rendered image aspect remains the physical ratio. The browser
listener lifecycle releases inactive panes' input guards. The
public 94-file MRI replay supplies the visual oracle; DICOM decoding and
clinical semantics remain in RITK.

## Verification record

The implementation is in `ritk-snap` and keeps DICOM decoding and geometry in
RITK. Locked native nextest runs 495 tests with 495 passes. Native and
`wasm32-unknown-unknown` library checks, strict Clippy, formatting, doctests
(4 passed, 1 intentionally ignored), and warning-clean Rustdoc pass. The
native command in the manual decoded the public 94-file MRI-DIR study and
produced a 1280×800 RGBA responsive capture with SHA-256
`056bf2cd8828df63972af2fe785e436ef0256e501a3ac7386535db0db169a0c9` and
534,414 non-black pixels; visual inspection shows axial, coronal, sagittal
and axial-MIP anatomy in the four-pane layout. The browser responsive surface
compiles for WASM and releases listeners by rebuilding hidden canvases without
input guards. The existing hosted gallery remains the visual oracle for the
fixed raster/WebGPU entrypoints. The consumer-owned saved-study gallery now
moves its existing canvas elements into the direct trusted container before
mounting this entrypoint. The consumer keeps the linked crosshair overlay in a
positioned layer over each adaptive canvas and reanchors it on browser resize.
Its responsive browser capture is tracked by
`RITK-SNAP-METIS-RESPONSIVE-BROWSER-001` and retains the same real-study pixel
and listener oracles as the fixed entrypoints.

### Revision 2026-09-21

Accepted after the native/WASM and visual evidence above. The responsive
browser contract is intentionally consumer-owned: RITK supplies the trusted
container and named canvases, while Métis remains a format-neutral canvas and
event provider. The native responsive workflow is an additive entrypoint so
the existing exhaustive `NativePresentationMode` enum remains stable.
Existing fixed browser and native entrypoints remain stable. The responsive
gallery integration is a consumer presentation change; it does not move DICOM
decoding or geometry into Métis.

The responsive consumer reapplies definite `width` and `height` values after
publishing each canvas's physical aspect. The generic fixed-gallery publisher
uses an auto height, which would let an anisotropic canvas expand a CSS grid
track beyond the trusted container; responsive tracks instead size the canvas
to their grid cell and preserve the physical ratio with `object-fit: contain`.
