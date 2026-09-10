# ADR 0027: Patient-coordinate fusion for viewer comparison

Status: Accepted

Date: 2026-09-09

Driver: [RITK-SNAP-FUSION-001](../../backlog.md#RITK-SNAP-FUSION-001).

## Context

`ritk-snap` previously blended a primary slice with a secondary slice by
scaling output row and column indices between the two raster sizes. That is
only correct when both arrays have identical origin, direction, spacing and
slice selection. A translated, rotated or anisotropic acquisition could
therefore produce a plausible image while placing anatomy at the wrong
location.

The viewer stores spatial dimensions in `[depth, row, column]` order. Each
`LoadedVolume` carries an origin, a row-major direction matrix and spacing in
the same order. DICOM's FrameOfReferenceUID is the identity contract for
combining separate acquisitions. A missing identity cannot be inferred from
similar dimensions or modality labels.

## Decision

Add one validated `AffineTransform` under the viewer geometry module. Its
forward map is

```text
patient = origin + direction · diag(spacing) · voxel
```

and its inverse is computed once with a scale-aware singularity check. RT
STRUCT projection uses this same transform, so patient-to-voxel arithmetic is
not duplicated in the overlay path.

`render_fused_slice` now returns a typed `FusionError` on malformed volume
layout, invalid geometry, invalid selections, incompatible or missing frame
identities, non-parallel planes, or a selected plane outside the secondary
normal extent. For two volumes without FrameOfReferenceUID values, fusion is
allowed only when origin, direction and spacing are exactly identical. When
both identifiers exist they must be equal.

The primary output grid remains authoritative. Every primary voxel centre is
mapped through patient space into the secondary grid, where the in-plane
coordinates use nearest-neighbour sampling. A mapped secondary point outside
the secondary field of view leaves the primary display value unchanged. The
application derives the secondary slice from the primary slice centre in
patient space; side-by-side comparison retains its independent slice
selection. Non-parallel planes are rejected because one fixed secondary slice
cannot represent an oblique resampling plane.

## Alternatives

Keeping normalized raster coordinates is rejected because it has no physical
alignment invariant. Resampling the secondary volume into a new temporary
buffer is deferred: it would allocate and cache a second volume for every
viewport update, while nearest-neighbour point sampling provides the required
comparison contract without changing source data. Silently accepting missing
or unequal frame identities is rejected because it turns an unknown
registration into an apparent clinical overlay.

## Failure and verification contract

The renderer validates shape, channel count, data length, affine invertibility,
slice bounds and blend-weight finiteness before allocating output. Its
manufactured tests cover translated and rotated anisotropic grids, frame
identity mismatch, missing identity on differing grids, normal and in-plane
out-of-field behavior, non-parallel planes, and patient-coordinate slice
selection. The affine tests cover a rotated anisotropic round trip and
singularity rejection. The DICOM manual records the physical equation and
the explicit failure behavior; a visual compare capture is required before
the item closes.

## Revision history

- 2026-09-09: Initial decision and implementation for
  RITK-SNAP-FUSION-001.
