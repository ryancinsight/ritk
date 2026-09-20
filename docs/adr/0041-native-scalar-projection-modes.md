# ADR 0041: Native scalar projection modes

- Status: Accepted
- Date: 2026-09-20
- Driver: [RITK-SNAP-SLAB-PRESENTATION-001](../../backlog.md#RITK-SNAP-SLAB-PRESENTATION-001)

## Context

The typed slab contract now provides exact maximum, minimum, and arithmetic-mean
planes for scalar `LoadedVolume` values. The native Métis session currently
exposes only the maximum projection, so the two other validated statistics
cannot be demonstrated through the same real DICOM workflow. A presentation
choice must remain a RITK concern: DICOM decoding, scalar reduction,
window/level, colormap and physical spacing stay upstream of the host boundary.

The existing maximum renderer is already the reviewed pixel contract for the
native four-panel layout. Replacing it with a second implementation would risk
changing captures and would make the native and eframe paths disagree.

## Decision

Extend `NativePresentationMode` with `OrthogonalWithMinip` and
`OrthogonalWithAverage`. Each mode maps to one `ProjectionStatistic`; the
existing `OrthogonalWithMip` maps to `Maximum` and keeps the existing MIP
renderer byte-for-byte. Minimum and average use `SlabProjection` over the
validated full axis-0 extent, then apply the existing RITK grayscale and
colormap mapping before constructing a `PresentationFrame`.

All three projection modes use the bounded 2×2 native layout. The lower-right
panel is display-only and its overlay labels the selected statistic and frame
dimensions. The default `Orthogonal` mode remains unchanged. Color volumes
return the existing scalar-projection error, and a mismatch between a mode and
its projection state is a typed session error rather than an implicit fallback.

The CLI names are `orthogonal-with-mip`, `orthogonal-with-minip`, and
`orthogonal-with-average`. The browser, oblique physical-plane resampling,
GPU slab dispatch, and installer are outside this item and retain their own
contracts.

## Alternatives

1. Add a boolean or string statistic field beside `NativePresentationMode`.
   Rejected because it permits an invalid combination and duplicates the
   `ProjectionStatistic` mapping at the session boundary.
2. Reimplement minimum and average beside the existing MIP loop. Rejected
   because the typed slab contract already owns range validation and traversal;
   a second indexing path would drift.
3. Move statistic selection into Métis. Rejected because the host is
   format-neutral and must not own DICOM or voxel semantics.

## Invariants and failure modes

- Every projection mode maps to exactly one `ProjectionStatistic`.
- The native projection accepts only scalar volumes and uses the validated
  full axis-0 extent; no index is clamped or inferred from host state.
- Maximum output remains byte-equivalent to the existing RITK MIP renderer.
- Minimum and average output is window/level and colormap mapped by the same
  RITK policy as orthogonal slices.
- The projection overlay identifies the statistic and frame dimensions.

## Verification

Unit tests cover all three mode mappings, four-panel rendering and overlay
labels, scalar pixel output, maximum-pixel equivalence, malformed color input,
and typed minimum/average slab values. The native locked nextest suite, strict
native and WASM checks/Clippy, rustdoc, formatting, lock validation and the
real public CT capture commands are run against the delivery revision.
