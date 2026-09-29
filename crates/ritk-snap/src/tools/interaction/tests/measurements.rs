use super::*;

// ── compute_length ────────────────────────────────────────────────────────

fn patient_point(coordinates: [f64; 3]) -> PatientPointMm {
    PatientPointMm::try_new(coordinates).expect("finite patient point")
}

#[test]
fn patient_length_uses_three_dimensional_euclidean_distance() {
    let measurement = PatientLength::try_new(
        patient_point([1.0, 2.0, 3.0]),
        patient_point([4.0, 6.0, 3.0]),
    )
    .expect("3-4-5 segment is representable");

    assert_eq!(measurement.start_mm().coordinates(), [1.0, 2.0, 3.0]);
    assert_eq!(measurement.end_mm().coordinates(), [4.0, 6.0, 3.0]);
    assert_eq!(measurement.length_mm(), 5.0);
}

#[test]
fn patient_length_accepts_coincident_endpoints() {
    let point = patient_point([-12.5, 0.0, 42.25]);
    let measurement = PatientLength::try_new(point, point).expect("zero length is finite");

    assert_eq!(measurement.length_mm(), 0.0);
}

#[test]
fn patient_length_rejects_overflowing_displacement() {
    let result = PatientLength::try_new(
        patient_point([-f64::MAX, 0.0, 0.0]),
        patient_point([f64::MAX, 0.0, 0.0]),
    );

    assert!(matches!(
        result,
        Err(MeasurementError::NonFiniteResult {
            kind: "patient length"
        })
    ));
}

#[test]
fn patient_length_serde_validates_and_round_trips_endpoints() {
    let measurement = PatientLength::try_new(
        patient_point([0.0, 0.0, 0.0]),
        patient_point([3.0, 4.0, 0.0]),
    )
    .expect("3-4-5 segment is representable");
    let json = serde_json::to_string(&measurement).expect("serialize patient length");
    let recovered: PatientLength = serde_json::from_str(&json).expect("deserialize patient length");
    assert_eq!(recovered, measurement);
    assert_eq!(recovered.length_mm(), 5.0);

    let overflow = r#"{"start_mm":[-1.7976931348623157e308,0.0,0.0],"end_mm":[1.7976931348623157e308,0.0,0.0]}"#;
    let error = serde_json::from_str::<PatientLength>(overflow)
        .expect_err("overflowing patient length must be rejected");
    assert!(error.to_string().contains("patient length"));
}

/// Axis-aligned horizontal displacement with unit spacing must yield the
/// exact integer pixel distance.
///
/// Analytical: p1=[0,0], p2=[0,3], spacing=[1,1]
/// length = √( (0·1)² + (3·1)² ) = √9 = 3.0
#[test]
fn test_compute_length_axis_aligned() {
    let p1 = [0.0_f32, 0.0];
    let p2 = [0.0_f32, 3.0];
    let spacing = [1.0_f32, 1.0];
    let length = Annotation::compute_length(p1, p2, spacing);
    assert_eq!(
        length, 3.0_f32,
        "axis-aligned horizontal length of 3 pixels with unit spacing must equal 3.0 mm"
    );
}

/// Axis-aligned vertical displacement with unit spacing.
///
/// Analytical: p1=[0,0], p2=[4,0], spacing=[1,1]
/// length = √( (4·1)² + (0·1)² ) = √16 = 4.0
#[test]
fn test_compute_length_axis_aligned_vertical() {
    let p1 = [0.0_f32, 0.0];
    let p2 = [4.0_f32, 0.0];
    let spacing = [1.0_f32, 1.0];
    let length = Annotation::compute_length(p1, p2, spacing);
    assert_eq!(
        length, 4.0_f32,
        "axis-aligned vertical length of 4 pixels with unit spacing must equal 4.0 mm"
    );
}

/// Non-unit spacing scales the physical length independently per axis.
///
/// Analytical: p1=[0,0], p2=[2,0], spacing=[0.5, 1.0]
/// length = √( (2·0.5)² + (0·1.0)² ) = √1 = 1.0
#[test]
fn test_compute_length_anisotropic_spacing() {
    let p1 = [0.0_f32, 0.0];
    let p2 = [2.0_f32, 0.0];
    let spacing = [0.5_f32, 1.0];
    let length = Annotation::compute_length(p1, p2, spacing);
    let expected = 1.0_f32;
    assert!(
        (length - expected).abs() < 1e-5,
        "anisotropic length: expected {expected}, got {length}"
    );
}

#[test]
fn checked_measurements_reject_unrepresentable_spacing() {
    assert!(matches!(
        Annotation::validate_spacing([0.0, 1.0]),
        Err(MeasurementError::InvalidSpacing { index: 0, .. })
    ));
    assert!(matches!(
        Annotation::validate_spacing([f64::MAX, 1.0]),
        Err(MeasurementError::UnrepresentableSpacing { index: 0, .. })
    ));
}

#[test]
fn checked_measurements_reject_nonfinite_derived_values() {
    assert!(matches!(
        Annotation::compute_length_checked([0.0, 0.0], [f32::MAX, 0.0], [1.0, 1.0]),
        Err(MeasurementError::NonFiniteResult { kind: "length" })
    ));
    assert!(matches!(
        Annotation::compute_roi_rect_stats_checked(
            [0.0, 0.0],
            [0.0, 0.0],
            &[f32::NAN],
            1,
            1,
            [1.0, 1.0],
        ),
        Err(MeasurementError::NonFiniteResult {
            kind: "rectangle ROI"
        })
    ));
}

// ── compute_angle ─────────────────────────────────────────────────────────

/// Three points forming a right angle (90°) at the vertex.
///
/// Analytical:
/// p1 = (0, 1), p2 = (0, 0) [vertex], p3 = (1, 0)
/// v₁ = (0,1)−(0,0) = (0,1), v₂ = (1,0)−(0,0) = (1,0)
/// dot = 0·1 + 1·0 = 0 ⟹ cos θ = 0 ⟹ θ = 90°
#[test]
fn test_compute_angle_right_angle() {
    let p1 = [0.0_f32, 1.0];
    let p2 = [0.0_f32, 0.0]; // vertex
    let p3 = [1.0_f32, 0.0];
    let angle = Annotation::compute_angle(p1, p2, p3);
    assert!(
        (angle - 90.0_f32).abs() < 0.001,
        "90° angle must be computed to within 0.001°, got {angle}°"
    );
}

/// Three collinear points (same direction) must yield 0°.
///
/// Analytical:
/// p1 = (0, 0), p2 = (0, 1) [vertex], p3 = (0, 2)
/// v₁ = (0,−1), v₂ = (0,1)
/// cos θ = −1 ⟹ θ = 180°
///
/// Note: the two rays point in exactly opposite directions, so the angle
/// at the vertex is 180°.
#[test]
fn test_compute_angle_straight_line() {
    let p1 = [0.0_f32, 0.0];
    let p2 = [0.0_f32, 1.0]; // vertex on the line
    let p3 = [0.0_f32, 2.0];
    let angle = Annotation::compute_angle(p1, p2, p3);
    assert!(
        (angle - 180.0_f32).abs() < 0.001,
        "straight-line angle must be 180°, got {angle}°"
    );
}

/// Degenerate input where p1 == p2 must return 0.0 rather than NaN or panic.
#[test]
fn test_compute_angle_degenerate_zero_length_ray() {
    let p1 = [1.0_f32, 1.0];
    let p2 = [1.0_f32, 1.0]; // p1 == p2 → zero-length ray
    let p3 = [2.0_f32, 3.0];
    let angle = Annotation::compute_angle(p1, p2, p3);
    assert_eq!(
        angle, 0.0_f32,
        "degenerate (zero-length ray) angle must return 0.0, got {angle}"
    );
}
