# ADR 0039: Physical volume contract at the VTK boundary

- Status: Accepted
- Date: 2026-09-20
- Delivery: [RITK PR #523](https://github.com/ryancinsight/ritk/pull/523), [RITK PR #715](https://github.com/ryancinsight/ritk/pull/715).
- Revision: 2026-10-02 — Move axis mapping into `ritk-vtk` and reject invalid dimensions or spacing before payload I/O or output truncation.

## Context

`ritk-snap::LoadedVolume` carries direction-aware physical geometry and
channel-fastest samples. `ritk-vtk::VtkImageVolume` preserves that geometry in a
zero-copy VTK handoff, while legacy structured-points files encode dimensions,
origin and spacing in XYZ order and omit the direction matrix.

RITK image tensors use `[depth, row, column]` order. Their direction columns
describe those tensor axes. VTK uses `[x, y, z] = [column, row, depth]`, with X
varying fastest. The provider must convert metadata at its boundary while
leaving the x-fastest sample sequence unchanged. VTK structured-points spacing
must be strictly positive, and dimensions must be at least one, as required by
[VTK's legacy file specification, Dataset Format](https://docs.vtk.org/en/v9.6.1/vtk_file_formats/vtk_legacy_file_format.html#dataset-format).

## Decision

`ritk-vtk::VtkImageVolume::from_tensor_parts` accepts tensor-order dimensions,
spacing and direction, maps them to VTK order, validates the geometry and
shares the scalar allocation. `ritk-snap` passes its metadata as stored and
does not contain VTK axis-conversion logic. Format-specific transformations
remain in the RITK format provider.

The legacy reader maps VTK dimensions and spacing into RITK tensor order and
constructs the corresponding tensor-axis direction. It rejects zero dimensions,
overflowed dimension products and invalid spacing before decoding scalar data.
The writer accepts the same VTK-aligned direction, reverses tensor spacing into
file order, and rejects other directions or non-Cartesian coordinate maps. It
validates dimensions, sample count and finite positive spacing before creating
or truncating the destination. Both paths preserve the sample sequence.

`VtkImageVolume` owns shared samples and direction-aware geometry.
`to_vtk_image_data` is the explicit copy boundary for legacy serializers and
filters that require owned `Vec<f32>` attributes; conversion and construction
do not copy the payload.

## Alternatives

1. Convert axes in `ritk-snap`. Rejected because format-specific semantics
   belong in the RITK provider and would otherwise be duplicated by every
   consumer.
2. Treat identity direction in tensor order as VTK-aligned. Rejected because
   tensor axes and VTK coordinates have different order; that drops the
   physical mapping for non-cubic, anisotropic volumes.
3. Add direction to the legacy file or infer it from spacing. Rejected because
   the legacy structured-points format has no direction field; spacing cannot
   encode rotation or obliquity.

## Consequences

- Empty dimensions, zero channels, overflowed sample counts, mismatched sample
  counts, non-finite origins or directions, non-positive spacing, and singular
  directions are rejected before a volume carrier is constructed.
- Legacy writer geometry and sample count are checked before the output path is
  opened, so invalid images cannot truncate an existing file.
- The VTK provider owns the tensor/XYZ permutation and shares the sample
  allocation. Tests assert geometry values, file round trips, and pointer
  identity.
- Legacy structured-points cannot preserve an arbitrary direction matrix;
  callers requiring oblique output use a direction-aware VTK format.

## Verification

Axis-order properties, anisotropic rotated geometry, malformed spacing and
dimensions, unsupported direction, destination preservation, payload validation
and pointer identity form the behavioral oracle. The source change is checked with
focused `ritk-vtk`, `ritk-io` and `ritk-snap` nextest suites, strict Clippy,
Rustdoc, formatting and the repository's configured gate.
