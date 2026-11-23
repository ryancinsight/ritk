//! Independently calculated DICOM patient coordinates and nearest-plane boundaries.
use super::*;
use crate::dicom::loader::{load_volume_from_path, tests::fixtures};

fn traced() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

fn read(positions: [i32; 3], frame: Option<&str>) -> LoadedVolume {
    let dir = tempfile::tempdir().unwrap();
    fixtures::write_positioned_study(
        dir.path(),
        "CT",
        fixtures::SERIES_UID,
        &fixtures::SAMPLES,
        (Some([1.5, 0.5]), true),
        positions,
        frame,
    )
    .unwrap();
    load_volume_from_path(dir.path()).unwrap()
}

#[test]
fn decoded_planes_use_patient_centers_and_measured_irregular_intervals() {
    let _trace = traced();
    let v = tracing::info_span!("arrange").in_scope(|| read([10, 12, 16], Some("2.25.259")));
    let stack = tracing::info_span!("act").in_scope(|| PatientStack::from_volume(&v).unwrap());
    let _check = tracing::info_span!("assert").entered();
    assert_eq!(stack.centers[1].as_slice(), &[12.0, 20.75, 30.75]);
    assert_eq!(stack.distances, [0., 2., 6.]);
    // At each bisector, and one representable coordinate to either side.
    for (boundary, left, equal, right) in [
        (-1.0_f64, None, Some(0), Some(0)),
        (1.0, Some(0), Some(0), Some(1)),
        (4.0, Some(1), Some(1), Some(2)),
        (8.0, Some(2), Some(2), None),
    ] {
        assert_eq!(
            [boundary.next_down(), boundary, boundary.next_up()].map(|x| stack.nearest(x)),
            [left, equal, right]
        );
    }
    assert_eq!(stack.nearest(f64::NAN), None);
    assert!(PatientStack::from_volume(&read([10, 12, 14], None)).is_none());
}

#[test]
fn synchronization_qualifies_frame_and_unoriented_dihedral_angle() {
    let _trace = traced();
    let (volume, mut tilted) = tracing::info_span!("arrange").in_scope(|| {
        let v = read([10, 12, 14], Some("2.25.259"));
        let target = PatientStack::from_volume(&v).unwrap();
        (v, target)
    });
    let source = PatientStack::from_volume(&volume).unwrap();
    for (degrees, expected) in [
        (0.0_f64, Some(1)),
        (5., Some(1)),
        (5.0001, None),
        (90., None),
        (180., Some(1)),
    ] {
        let result = tracing::info_span!("act", degrees).in_scope(|| {
            let (sin, cos) = degrees.to_radians().sin_cos();
            tilted.normal = Vector3::new([cos, sin, 0.]);
            if degrees == 180. {
                tilted.centers.reverse();
            }
            source.corresponding_slice(1, &tilted)
        });
        tracing::info_span!("assert", degrees).in_scope(|| assert_eq!(result, expected));
    }
    tilted.frame = "2.25.260".into();
    assert_eq!(source.corresponding_slice(1, &tilted), None);
}
