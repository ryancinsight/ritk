//! Physical deviation measurements for one image-space segment.

use super::{Annotation, MeasurementError};

impl Annotation {
    /// Return the baseline, physical offset, and acute angle for a segment.
    pub(crate) fn compute_deviation_with_angle(
        start: [f32; 2],
        end: [f32; 2],
        spacing: [f32; 2],
    ) -> Result<([f32; 2], f32, f32), MeasurementError> {
        let dx = ((end[1] - start[1]) * spacing[1]).abs();
        let dy = ((end[0] - start[0]) * spacing[0]).abs();
        if !dx.is_finite() || !dy.is_finite() {
            return Err(MeasurementError::NonFiniteResult { kind: "deviation" });
        }
        let (baseline, offset, run) = if dx >= dy {
            ([start[0], end[1]], dy, dx)
        } else {
            ([end[0], start[1]], dx, dy)
        };
        Ok((baseline, offset, offset.atan2(run).to_degrees()))
    }
}

#[cfg(test)]
mod tests {
    use super::Annotation;

    #[test]
    fn deviation_uses_physical_row_and_column_spacing() {
        let (baseline, distance, angle) =
            Annotation::compute_deviation_with_angle([0.0, 0.0], [2.0, 4.0], [2.0, 1.0])
                .expect("finite physical deviation");
        assert_eq!(baseline, [0.0, 4.0]);
        assert_eq!(distance, 4.0);
        assert!((angle - 45.0).abs() < 1e-6);
    }
}

#[cfg(test)]
#[test]
fn deviation_rejects_unrepresentable_physical_components() {
    assert_eq!(
        Annotation::compute_deviation_with_angle([0.0; 2], [2.0; 2], [f32::MAX; 2]),
        Err(MeasurementError::NonFiniteResult { kind: "deviation" })
    );
}
