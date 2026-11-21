# ADR 0064: Type DICOM geometry and PET activity

- Status: Proposed
- Class: [major]
- Date: 2026-11-19

Store reconstructed geometry and per-slice distances as Aequitas lengths, and
store gantry tilt as an angle. DICOM uses millimetres and degrees. Existing
image geometry and textual tag adapters request those units explicitly. The
conversion preserves axis order and carries ordinary floating-point rounding.

Store patient weight and physical half-life as mass and time. Wrap reciprocal
time in a radionuclide activity type. Classic PET encodes total activity in Bq.
Enhanced PET and Nuclear Medicine encode it in MBq. The reader dispatches by
SOP class. Multiple isotope items leave the series scalar unspecified and
remain available in the preserved object. The viewer accepts classic PET SUV
conversion when all instances declare Bq/mL pixel units. Enhanced PET activity
is parsed, with viewer conversion reserved for a validated pixel mapping.

A scalar or modality-only dispatch can produce a million-fold dose error.
DICOM Part 10 fixtures cover all three storage classes, a nonrepresentable
dose, and series spacing. Viewer tests cover normalization and pixel units.

## Series assembly

Preserve the parsed origin and in-plane lengths during series assembly. Depth
is a length returned by the projection-spacing analysis, where the established
absolute thresholds use millimetres. Compute patients' origin in XYZ and keep
tensor spacing in depth, row, and column order. The tensor loader requests
millimetres through the metadata accessors. An oblique Part 10 fixture checks
the axis order, patient origin, measured depth, and single-slice fallback.

## Series writer image planes

Native and Coeus image adapters interpret the image's origin and spacing
in millimetres. The metadata-preserving writer takes the reader's existing
lengths. All series writers compute `origin + z · Δdepth · dir_depth` in
lengths and use the same Image Plane Module encoder. DICOM receives
millimetres, pixel spacing in row/column order, and orientation cosines
in column/row order. A spacing that underflows to zero metres or a
patient position that exceeds the scalar range is rejected before output.

## Multiframe image geometry

Represent top-level and functional-group patient positions, pixel spacing,
and thickness as lengths. A malformed dimensional tuple supplies no value
and leaves the established metadata fallback in effect. The scalar and color
decoders form signed plane projections from length components and
dimensionless normals. The existing spacing analysis and interpolation
receive millimetres.

The generic scalar loader uses the first decoded frame's geometry when
functional groups supply it. The top-level header remains its fallback.
For an oblique Part 10 stack at patient x coordinates 17, 21, 29, and 33 mm,
the 4 mm median gap places a reconstructed plane at 25 mm. The regression
checks patient XYZ origin, anisotropic spacing, pixel values, the
interpolated-plane label, and a malformed top-level tuple.
