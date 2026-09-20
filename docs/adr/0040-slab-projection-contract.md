# ADR 0040: Typed bounded slab projection contract

- Status: Accepted
- Date: 2026-09-20
- Driver: [RITK-SNAP-SLAB-PROJECTION-001](../../backlog.md#RITK-SNAP-SLAB-PROJECTION-001)

## Context

RITK already owns decoded DICOM voxel storage and exposes scalar axial MIP and
volume rendering to its native shell. The merged physical-volume contract
preserves dimensions, spacing, origin and direction at the VTK boundary, but a
consumer still cannot request a bounded slab statistic without reaching into
the viewer's storage layout. That would duplicate indexing rules in each host
and make a future oblique or GPU implementation appear equivalent to an
axis-aligned operation when it is not.

The first contract must be host-neutral and exact. It therefore operates on
the existing row-major `[depth, rows, columns, channels]` layout, accepts only
scalar volumes, and names the statistic explicitly. It reports a compact
`[rows, columns]`, `[depth, columns]` or `[depth, rows]` plane with the source
axis and sample count retained for a host to label or render.

## Decision

Add `ProjectionStatistic::{Maximum, Minimum, Average}` and a validated
`SlabProjection` request. The request contains an axis, centre index and
half-width in voxels. Construction rejects axes outside `0..=2`, non-scalar
volumes, empty dimensions and a range that cannot be represented inside the
selected extent. The inclusive range is `centre - half_width ..= centre +
half_width`; the constructor clamps no input and performs no implicit
resampling.

`SlabProjection::compute` traverses the requested source samples once per
output pixel and returns an owned `ProjectionPlane`. `compute_into` performs
the same traversal into caller-owned `Vec<f32>` storage so repeated consumers
can retain capacity. Output dimensions follow `LoadedVolume::extract_slice`:
axis 0 is `[columns, rows]`, axis 1 is `[columns, depth]`, and axis 2 is
`[rows, depth]`. Average uses the exact arithmetic mean of the selected `f32`
samples. No GUI image type, DICOM identifier or VTK object crosses this
boundary.

The current increment does not add oblique physical-plane resampling, GPU
dispatch, RGB statistics, or a new Metis API. Those operations need their own
contracts and independent differential evidence. A host may apply the existing
DICOM window/level and colormap after projection.

## Alternatives

1. Add a `slab_width` field to `ProjectionMode`. Rejected because it couples a
   host UI state enum to a numerical request and cannot represent statistic or
   axis validation without partial states.
2. Reuse the MIP loop with sentinel values and a mode boolean. Rejected because
   minimum and average have different initialization and accumulation laws;
   a typed statistic keeps those invariants explicit.
3. Put slab sampling in `ritk-vtk` or Metis. Rejected because voxel indexing
   and clinical scalar semantics belong to RITK; VTK is the physical carrier
   and Metis is the format-neutral presentation host.

## Invariants and failure modes

- Only scalar (`channels == 1`) volumes are accepted.
- Every selected source index is in the validated spatial extent.
- Output length equals the product of the returned dimensions.
- `Maximum` and `Minimum` preserve source values exactly; `Average` performs
  one `f32` sum and division by the validated sample count.
- Empty or malformed requests return typed errors and never panic or allocate
  from an unbounded input.

## Verification

Manufactured volumes exercise all three axes and statistics with closed-form
values, invalid axis/channel/extent partitions, one-sample equivalence with
slice extraction, and scratch-capacity reuse. Package nextest, strict Clippy,
rustdoc, formatting, lock validation and diff checks run against the exact
delivery revision. Existing real-MRI captures remain the presentation oracle;
this contract adds no fabricated clinical image.
