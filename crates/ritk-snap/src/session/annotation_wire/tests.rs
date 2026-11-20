use crate::session::{
    load_from_file, save_to_file, tests::all_annotation_variants, ViewerSessionSnapshot,
};
use crate::tools::interaction::Annotation;
use serde_json::{error::Category, Value};

fn number_paths(value: &Value, prefix: &str, paths: &mut Vec<String>) {
    match value {
        Value::Number(_) => paths.push(prefix.to_owned()),
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                number_paths(value, &format!("{prefix}/{index}"), paths);
            }
        }
        Value::Object(values) => {
            for (key, value) in values {
                number_paths(value, &format!("{prefix}/{key}"), paths);
            }
        }
        _ => {}
    }
}

#[test]
fn image_annotation_numbers_reject_overflow_without_recomputing_finite_values() {
    for annotation in all_annotation_variants() {
        let value = serde_json::to_value(&annotation).expect("finite fixture serializes");
        let restored: Annotation =
            serde_json::from_value(value.clone()).expect("finite fixture restores");
        assert_eq!(restored, annotation);
        match annotation {
            Annotation::PatientLength { .. } => continue,
            Annotation::Length { .. }
            | Annotation::Angle { .. }
            | Annotation::RoiRect { .. }
            | Annotation::RoiEllipse { .. }
            | Annotation::Deviation { .. }
            | Annotation::HuPoint { .. }
            | Annotation::ClosedPolygon { .. }
            | Annotation::OpenPolygon { .. }
            | Annotation::Arrow { .. }
            | Annotation::Pencil { .. }
            | Annotation::CobbAngle { .. } => {}
        }
        let mut paths = Vec::new();
        number_paths(&value, "", &mut paths);
        for path in paths {
            let mut oversized = value.clone();
            *oversized
                .pointer_mut(&path)
                .expect("path belongs to this fixture") =
                serde_json::from_str("1e100").expect("finite JSON number exceeds binary32 range");
            let error = serde_json::from_value::<Annotation>(oversized)
                .expect_err("overflowing stored annotation component must be rejected");
            assert_eq!(error.classify(), Category::Data, "{path}");
        }
    }
}

#[test]
fn session_files_reject_overflow_and_retain_finite_cached_results() {
    let root = tempfile::tempdir().expect("create private session directory");
    let path = root.path().join("session.json");
    let mut snapshot = ViewerSessionSnapshot::empty();
    snapshot.annotations = all_annotation_variants();
    save_to_file(&snapshot, &path).expect("save finite session");
    assert_eq!(
        load_from_file(&path).expect("load finite session"),
        snapshot
    );
    let mut value = serde_json::to_value(&snapshot).expect("finite snapshot serializes");
    value["annotations"][0]["Length"]["length_mm"] =
        serde_json::from_str("1e100").expect("finite JSON number exceeds binary32 range");
    std::fs::write(
        &path,
        serde_json::to_vec(&value).expect("serialize finite JSON"),
    )
    .expect("write synthetic invalid session");
    let error = load_from_file(&path).expect_err("non-finite cached Length must not load");
    let json_error = error
        .downcast_ref::<serde_json::Error>()
        .expect("preserve JSON source error");
    assert_eq!(json_error.classify(), Category::Data);
}

#[test]
fn non_finite_components_cannot_serialize_as_null() {
    for component in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let annotations = [
            Annotation::Length {
                plane: 0,
                location: None,
                p1: [component, 0.0],
                p2: [3.0, 4.0],
                length_mm: 5.0,
            },
            Annotation::Length {
                plane: 0,
                location: None,
                p1: [0.0; 2],
                p2: [3.0, 4.0],
                length_mm: component,
            },
            Annotation::OpenPolygon {
                location: None,
                points: vec![[0.0; 2], [component, 1.0]],
                length_mm: 2.0,
            },
            Annotation::CobbAngle {
                location: None,
                points: [[component, 0.0]; 4],
                angle_deg: 90.0,
            },
        ];
        for annotation in annotations {
            let error = serde_json::to_value(annotation)
                .expect_err("non-finite input must not become null");
            assert_eq!(error.classify(), Category::Data);
        }
    }
}

#[test]
fn finite_cached_extremes_and_signed_zero_retain_their_bits() {
    for length_mm in [
        -f32::MAX,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
    ] {
        let annotation = Annotation::Length {
            plane: 0,
            location: None,
            p1: [0.0; 2],
            p2: [3.0, 4.0],
            length_mm,
        };
        let json = serde_json::to_string(&annotation).expect("finite Length serializes");
        let restored: Annotation = serde_json::from_str(&json).expect("finite Length restores");
        let Annotation::Length {
            length_mm: restored,
            ..
        } = restored
        else {
            panic!("Length variant must not change");
        };
        assert_eq!(restored.to_bits(), length_mm.to_bits());
    }
}

#[test]
fn finite_json_exponents_cannot_overflow_image_annotation_storage() {
    // 2^128 < 2^130 = 8192^10 < 10^40, whereas 10^300 < 2^1000.
    // Thus these magnitudes exceed every finite binary32 value while still
    // fitting binary64 JSON decoding. Both signs must fail at storage entry.
    for exponent in 40..=300 {
        for sign in ["", "-"] {
            let json = format!(
                "{{\"Length\":{{\"p1\":[0,0],\"p2\":[3,4],\"length_mm\":{sign}1e{exponent}}}}}"
            );
            let error = serde_json::from_str::<Annotation>(&json)
                .expect_err("finite JSON input cannot become an infinite cached length");
            assert_eq!(error.classify(), Category::Data);
        }
    }
}

#[test]
fn patient_lengths_retain_their_patient_coordinate_range() {
    use crate::geometry::PatientPointMm;
    use crate::tools::interaction::PatientLength;

    let start = PatientPointMm::try_new([0.0; 3]).expect("finite origin");
    let end = PatientPointMm::try_new([1e100, 0.0, 0.0]).expect("finite patient coordinates");
    let annotation = Annotation::PatientLength {
        plane: 0,
        location: None,
        length: PatientLength::try_new(start, end).expect("finite patient displacement"),
    };
    let json = serde_json::to_string(&annotation).expect("finite patient length serializes");
    let restored: Annotation =
        serde_json::from_str(&json).expect("patient range remains unchanged");
    assert_eq!(restored, annotation);
}
