# ADR 0038: Carry physical display geometry with presentation frames

Status: Accepted

Date: 2026-09-19

Delivery: [RITK PR #518](https://github.com/ryancinsight/ritk/pull/518), merge `694904718d7ec922883ab2a6b572e9aa29edab99`.

## Decision

`ritk_snap::presentation::PresentationFrame` stores a validated
`PresentationSpacing` value containing the row and column sample distances used
to display its pixels. Slice construction derives the pair from
`LoadedVolume::spacing` and axis order; transformed native frames retain the
pair in their output orientation. Pixel-only fixtures use unit spacing through
the existing constructor.

Native placement and browser semantics read the frame value. They do not
recompute spacing from a second copy of the volume state. The browser adapter
converts the validated pair to Metis's format-neutral `DisplaySpacing` at the
canvas boundary. DICOM identifiers, paths and decoded volume storage remain
inside RITK.

## Alternatives

- Keep spacing in `RenderedView` and derive browser attributes separately.
  Rejected because each host could drift while producing identical pixels.
- Add voxel or DICOM metadata to Metis. Rejected because the host boundary is
  format-neutral and must not own clinical semantics.
- Infer physical geometry from pixel dimensions. Rejected because anisotropic
  volumes have identical pixel shapes with different physical extents.

## Invariants and verification

- Every stored distance is finite and strictly positive.
- Axis order is `[row, column]` for the rendered frame and follows the RITK
  slice convention.
- Native transformed frames use the same display order that their pixels use.
- Browser `data-ritk-display-aspect` and native placement derive from the same
  frame metadata.

Axis-specific anisotropic fixtures, transformed native layouts, browser aspect
validation, malformed spacing rejection, and the real 94-file MRI replay are
the acceptance evidence. The VTK adapter and oblique/slab projection remain
separate RITK work items.
