# ADR 0050: Responsive Métis pane layout

- Status: Proposed
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
and assert the rendered image aspect remains the physical ratio. Browser
listener-count tests prove inactive panes do not retain input guards. The
public 94-file MRI replay supplies the visual oracle; DICOM decoding and
clinical semantics remain in RITK.
