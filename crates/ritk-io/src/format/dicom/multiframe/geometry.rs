//! DICOM length decoding and patient-plane projection.

use aequitas::systems::si::{quantities::Length, units::Millimeter};

pub(in crate::format::dicom) const DEFAULT_SPACING: Length = Length::from_base(0.001);

/// Parse an entire DS tuple. Dropping one token would shift the patient axes.
pub(super) fn parse_lengths_mm<const N: usize>(text: &str) -> Option<[Length; N]> {
    let mut values = text.split('\\');
    let mut lengths = [Length::from_base(0.0); N];
    for length in &mut lengths {
        *length = length_from_mm(values.next()?.trim().parse().ok()?)?;
    }
    values.next().is_none().then_some(lengths)
}

pub(super) fn length_from_mm(value: f64) -> Option<Length> {
    let length = Length::from_unit::<Millimeter>(value);
    (value.is_finite() && (value == 0.0 || length.into_base() != 0.0)).then_some(length)
}

/// The normal is dimensionless. Its dot product with a patient point is a
/// signed distance along the depth axis.
pub(in crate::format::dicom) fn project_position(
    [x, y, z]: [Length; 3],
    [nx, ny, nz]: [f64; 3],
) -> Length {
    x * nx + y * ny + z * nz
}

pub(in crate::format::dicom) fn millimetres<const N: usize>(lengths: [Length; N]) -> [f64; N] {
    lengths.map(|length| length.in_unit::<Millimeter>())
}
