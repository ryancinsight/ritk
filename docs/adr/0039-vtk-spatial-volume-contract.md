# ADR 0039: Physical volume contract at the VTK boundary

- Status: Accepted
- Date: 2026-09-20
- Driver: [RITK-VTK-SPATIAL-VOLUME-001](../../backlog.md#RITK-VTK-SPATIAL-VOLUME-001)

## Context

`ritk-snap::LoadedVolume` already carries direction-aware physical geometry and
channel-fastest scalar samples. `ritk-vtk::VtkImageData` models the VTK
regular-grid payload, but its legacy representation has no direction matrix
and owns `Vec<f32>` attribute arrays. Converting a clinical volume directly to
that type either drops orientation or copies the complete payload before a VTK
consumer can use it. Either outcome prevents a trustworthy physical-volume
pipeline for Metis and later resampling/projection work.

The boundary must preserve the RITK axis contract: source dimensions are
`[depth, row, column]`, spacing is `[dz, dy, dx]`, and the direction matrix's
columns describe those source axes. VTK dimensions and spacing are ordered
`[x, y, z] = [column, row, depth]`; its direction columns therefore reorder to
`[column, row, depth]` at the conversion boundary.

## Decision

Add `ritk_vtk::VtkImageVolume`, a validated, zero-copy spatial volume carrier.
It owns an `Arc<Vec<f32>>` scalar payload plus VTK-order extent, origin,
spacing, direction and channel count. The type exposes borrowed scalar data and
geometry accessors. `to_vtk_image_data` is an explicit copy boundary for
existing VTK serializers and filters that require the legacy `Vec<f32>`
attribute representation; it never runs during construction or conversion from
`LoadedVolume`.

`ritk-snap` implements `TryFrom<&LoadedVolume> for VtkImageVolume`. The adapter
performs only axis/order conversion and delegates validation to the VTK
carrier. It does not parse DICOM, compute clinical semantics, or move viewer
state into `ritk-vtk`. Metis continues to receive format-neutral rendered
frames; a later increment may consume this carrier for oblique resampling and
slab projection.

## Alternatives

1. Add a direction field to `VtkImageData`. Rejected for this increment:
   direction is not part of the legacy VTK ImageData serialization contract,
   and adding a public field breaks every external struct literal while still
   leaving the payload-copy problem.
2. Copy `LoadedVolume` into `VtkImageData` and retain direction in a side map.
   Rejected: side metadata can be separated from the payload and silently
   lost; the copy also violates the hot-path zero-copy requirement.
3. Put DICOM or `LoadedVolume` types in `ritk-vtk`. Rejected: it reverses the
   dependency direction and makes a published VTK crate depend on the viewer
   domain.

## Failure modes and invariants

- Empty dimensions, zero channels, overflowed sample counts and mismatched
  payload lengths are rejected before a carrier is constructed.
- Origin and direction entries must be finite; spacing must be finite and
  strictly positive; the direction matrix must be numerically invertible.
- The conversion shares the source allocation; pointer identity is asserted
  by a regression test.
- VTK materialization retains VTK's x-fastest point order and channel count,
  and its result passes `VtkImageData::validate`.

## Verification

The increment uses analytical geometry tests for an anisotropic rotated volume,
negative tests for each validity partition, a pointer-identity test for the
zero-copy handoff, and a materialization round trip over scalar and RGB
channels. Package nextest, strict Clippy, rustdoc, formatting and diff checks
run against the exact commit.
