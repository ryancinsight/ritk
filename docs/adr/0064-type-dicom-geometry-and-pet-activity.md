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
