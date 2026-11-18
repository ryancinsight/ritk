use super::{Annotation, MeasurementError};

#[test]
fn cobb_angle_measures_physical_directions() {
    let origin = [0.0, 0.0];
    assert_eq!(
        Annotation::compute_cobb_angle(origin, [0.0, 4.0], [3.0, 2.0], [-1.0, 2.0], [1.0; 2]),
        Ok(90.0)
    );
    // Physical vectors [6, 4] and [8, -3] have determinant magnitude 50 and dot 36.
    let expected = 50.0_f32.atan2(36.0).to_degrees();
    // Normalization and products introduce only a few ulps at this well-conditioned input.
    let rounding_bound = 16.0 * f32::EPSILON * expected;
    for (start, end) in [(origin, [4.0, -3.0]), ([4.0, -3.0], origin)] {
        let actual = Annotation::compute_cobb_angle(origin, [3.0, 4.0], start, end, [2.0, 1.0])
            .expect("invariant: fixture has two finite nonzero physical directions");
        assert!((actual - expected).abs() <= rounding_bound);
    }
    assert!(matches!(
        Annotation::compute_cobb_angle(origin, origin, origin, [2.0, 0.0], [1.0; 2]),
        Err(MeasurementError::DegenerateSegment { kind: "Cobb angle" })
    ));
    assert!(matches!(
        Annotation::compute_cobb_angle(
            origin,
            [0.0, 2.0],
            origin,
            [2.0, 0.0],
            [f64::from(f32::MAX); 2]
        ),
        Err(MeasurementError::NonFiniteResult { kind: "Cobb angle" })
    ));
}
