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

## Modality value units

Window center, width, and typed rescale offset use the post-rescale unit.
CT HU is a semantic dimensionless value. An explicit Rescale Type takes
precedence, and classic CT supplies an implicit HU unit for original
non-localizer single-energy images. Ordinary MR or other unspecified
values keep an uncalibrated numeric value. Mass per volume in mg/mL
equals SI kg/m³. Classic PET BQML maps to activity concentration, and
the viewer requires that code on every instance before calculating SUV.
Classic GML carries mass-per-volume SUV and the source normalization.
The standard supplies body weight when SUV Type is absent. The viewer
reports a body-weight SUV directly only when every instance uses it.
Enhanced PET keeps its existing frame-qualified Real World Value Mapping.

Keep the established scalar codec parameters and image buffers for pixel
decode and GPU presentation. Convert the window and SUV to DICOM/display
units before the existing rendering boundary. Normalization divides
activity concentration by activity per unit body mass.

[DICOM CT Image Module](https://dicom.nema.org/medical/Dicom/current/output/chtml/part03/sect_C.8.2.html)
and [PET Information Modules](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.html)
define the interpretation used here.

## Reconcile frame units before spatial resampling

The [Pixel Value Transformation Macro](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html#sect_C.7.6.16.2.9)
defines the unit of each frame after its affine rescale. Enhanced frame
metadata resolves per-frame, shared functional group, then instance-level
units. The classic loader uses the unit resolved for each slice.

Resampling forms `(1 − w)a + wb` after each frame's own affine calibration.
That expression has a physical meaning only on a common value scale. The
reader accepts equal resolved units, including arrays with consistently
unspecified units. It reports the conflicting source frames for different
units or a declared and unspecified pair. The
[Modality LUT and Rescale Type](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.html#sect_C.11.1.1.2)
permits vendor declarations. Shared string ownership preserves their full
identity through frame metadata and samples. It does not infer units from
adjacent planes. Uniform stacks do not enter this interpolation gate.
Quantitative Enhanced PET SUV remains a separate channel with its existing
validated real-world mappings. Part 10 fixtures exercise differing slopes
on one scale, incompatible scales, and missing declarations.

## Classic PET activity reference

Calculate body-weight SUV only when pixel concentration and administered
activity refer to the same time. For a concentration `C`, body mass `M`,
administered activity `A₀`, and elapsed time `t`, the normalization is
`C × M / (A₀ × 2^(−t/half_life))`. `ADMIN` pixels already share the
administration reference. For `START`, subtract
`half_life × log₂(Decay Factor)` from the declared frame reference to
recover the scanner's acquisition-start reference. For `NONE`, use the
declared frame reference when it falls within the acquisition interval.
Series Date and Time locate Frame Reference Time on the timeline.

An explicit date distinguishes midnight rollover from reversed timing.
Interpret timestamps on a common time base and require a known decay
mode, a unique isotope, and nonnegative elapsed time for `START` and
`NONE`. Withhold SUV for time-only administration, malformed or
contradictory clocks, and series that require distinct normalization
factors. A series scalar can move across spatial interpolation because
`k((1−w)a + wb) = (1−w)ka + wkb` for constant `k`. Series with
different factors would require frame-specific normalization before
interpolation. Keep their decoded activity concentration available for
display. Pre-normalized GML body-weight SUV and the Enhanced PET
quantitative channel continue to use their source mappings.

Using zero for missing time conflates unavailable data with an actual
zero interval. Using Series Time as the acquisition event or inferring
the administration date from times of day can manufacture an interval.
The reader instead resolves calibration while it has each source dataset.
Viewer pointer and fusion code consume the same optional calibration.
Part 10 regressions check the two-voxel scaling against a decimal
reference, mode semantics, dated rollover, UTC offsets, malformed
timing, and cross-slice agreement.

[DICOM PET Isotope Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.2.html)
and [PET Image Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.9.4.html)
define the timing attributes.
