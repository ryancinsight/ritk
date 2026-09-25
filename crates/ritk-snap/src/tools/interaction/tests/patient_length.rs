use crate::geometry::PatientPointMm;
use crate::tools::interaction::{MeasurementError, PatientLength};

fn patient_point(coordinates: [f64; 3]) -> PatientPointMm {
    PatientPointMm::try_new(coordinates).expect("finite patient point")
}

#[test]
fn patient_length_uses_all_three_patient_axes() {
    let measurement = PatientLength::try_new(
        patient_point([1.0, 2.0, 3.0]),
        patient_point([3.0, 5.0, 9.0]),
    )
    .expect("three-axis displacement is representable");

    assert_eq!(measurement.start_mm().coordinates(), [1.0, 2.0, 3.0]);
    assert_eq!(measurement.end_mm().coordinates(), [3.0, 5.0, 9.0]);
    // The displacement [2, 3, 6] has Euclidean norm √(4 + 9 + 36) = 7 mm.
    assert_eq!(measurement.length_mm(), 7.0);
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
