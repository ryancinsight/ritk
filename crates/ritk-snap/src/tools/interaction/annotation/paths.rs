//! Physical deviation measurements for one image-space segment.

use super::Annotation;

impl Annotation {
    /// Return the baseline, physical offset, and acute angle for a segment.
    pub(crate) fn compute_deviation_with_angle(
        start: [f32; 2],
        end: [f32; 2],
        spacing: [f32; 2],
    ) -> ([f32; 2], f32, f32) {
        let dx = (end[1] - start[1]) * spacing[1];
        let dy = (end[0] - start[0]) * spacing[0];
        if dx.abs() >= dy.abs() {
            (
                [start[0], end[1]],
                dy.abs(),
                dy.abs().atan2(dx.abs()).to_degrees(),
            )
        } else {
            (
                [end[0], start[1]],
                dx.abs(),
                dx.abs().atan2(dy.abs()).to_degrees(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Annotation;

    #[test]
    fn deviation_uses_physical_row_and_column_spacing() {
        let (baseline, distance, angle) =
            Annotation::compute_deviation_with_angle([0.0, 0.0], [3.0, 4.0], [2.0, 1.0]);
        assert_eq!(baseline, [0.0, 4.0]);
        assert_eq!(distance, 6.0);
        assert!((angle - 36.869_9).abs() < 0.001);
    }
}
