//! Physical path and segment measurements.

use super::Annotation;

impl Annotation {
    /// Compute shoelace area in mm² for a closed polygon.
    pub fn compute_polygon_area(points: &[[f32; 2]], spacing: [f32; 2]) -> f32 {
        let [origin, vertices @ ..] = points else {
            return 0.0;
        };
        let sum = vertices
            .windows(2)
            .map(|pair| {
                let [a, b] = pair else {
                    unreachable!("invariant: windows(2) contains two vertices")
                };
                let [ar, ac] = [a[0] - origin[0], a[1] - origin[1]];
                let [br, bc] = [b[0] - origin[0], b[1] - origin[1]];
                ar * bc - br * ac
            })
            .sum::<f32>();
        0.5 * sum.abs() * spacing[0] * spacing[1]
    }

    /// Compute physical polyline length.
    pub fn compute_polyline_length(points: &[[f32; 2]], spacing: [f32; 2]) -> f32 {
        points
            .windows(2)
            .map(|pair| Self::compute_length(pair[0], pair[1], spacing))
            .sum()
    }

    /// Compute the acute angle between two lines in degrees.
    pub fn compute_cobb_angle(
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
        p4: [f32; 2],
        spacing: [f32; 2],
    ) -> f32 {
        let v1 = [p2[0] - p1[0], p2[1] - p1[1]];
        let v2 = [p4[0] - p3[0], p4[1] - p3[1]];
        let q1 = [v1[0] * spacing[0], v1[1] * spacing[1]];
        let q2 = [v2[0] * spacing[0], v2[1] * spacing[1]];
        let dot = q1[0] * q2[0] + q1[1] * q2[1];
        let determinant = q1[0] * q2[1] - q1[1] * q2[0];
        determinant.abs().atan2(dot.abs()).to_degrees()
    }

    /// Compute RadiAnt's deviation distance and acute angle for one segment.
    ///
    /// The reference is the nearer horizontal or vertical axis in physical
    /// space. The distance is therefore independent of endpoint direction,
    /// while the angle is always acute.
    pub(crate) fn compute_deviation_with_angle(
        start: [f32; 2],
        end: [f32; 2],
        spacing: [f32; 2],
    ) -> ([f32; 2], f32, f32) {
        let dx = (end[1] - start[1]) * spacing[1];
        let dy = (end[0] - start[0]) * spacing[0];
        let horizontal = dx.abs() >= dy.abs();
        if horizontal {
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
    fn polygon_area_and_polyline_length_use_spacing() {
        let square = [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]];
        assert_eq!(Annotation::compute_polygon_area(&square, [1.0, 1.0]), 4.0);
        assert_eq!(
            Annotation::compute_polyline_length(&square[..2], [2.0, 1.0]),
            2.0
        );
    }

    #[test]
    fn cobb_angle_is_line_angle() {
        let angle = Annotation::compute_cobb_angle(
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        );
        assert_eq!(angle, 90.0);
    }

    #[test]
    fn deviation_segment_reports_axis_distance_and_acute_angle() {
        let (reference, distance, angle) =
            Annotation::compute_deviation_with_angle([0.0, 0.0], [3.0, 4.0], [1.0, 1.0]);
        assert_eq!(reference, [0.0, 4.0]);
        assert_eq!(distance, 3.0);
        assert!((angle - 36.8699).abs() < 1e-3);
        let (reverse_reference, reverse_distance, reverse_angle) =
            Annotation::compute_deviation_with_angle([3.0, 4.0], [0.0, 0.0], [1.0, 1.0]);
        assert_eq!(reverse_reference, [3.0, 0.0]);
        assert_eq!(reverse_distance, distance);
        assert_eq!(reverse_angle, angle);
    }
}
