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
    let dir = tempfile::tempdir().expect("create the DICOM directory");
    fixtures::write_positioned_study(
        dir.path(),
        "CT",
        fixtures::SERIES_UID,
        &fixtures::SAMPLES,
        (Some([1.5, 0.5]), true),
        positions,
        frame,
    )
    .expect("write the three positioned slices");
    load_volume_from_path(dir.path()).expect("decode the real Part 10 series")
}

#[test]
fn decoded_planes_use_patient_centers_and_measured_irregular_intervals() {
    let _trace = traced();
    let v = tracing::info_span!("arrange").in_scope(|| read([10, 12, 16], Some("2.25.259")));
    let stack = tracing::info_span!("act")
        .in_scope(|| PatientStack::from_volume(&v).expect("qualify the irregular stack"));
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
        let target = PatientStack::from_volume(&v).expect("qualify the target stack");
        (v, target)
    });
    let source = PatientStack::from_volume(&volume).expect("qualify the source stack");
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

#[test]
fn anchored_travel_crosses_frames_and_respects_angle_and_extent() {
    let _trace = traced();
    let (mut source, mut target) = tracing::info_span!("arrange").in_scope(|| {
        let qualify = |positions, frame| {
            PatientStack::from_volume(&read(positions, Some(frame))).expect("positioned stack")
        };
        (
            qualify([10, 12, 14], "2.25.259"),
            qualify([100, 101, 103], "2.25.260"),
        )
    });
    assert_eq!(source.anchored_slice(0, &target), None);
    source.anchor = Some(1);
    target.anchor = Some(1);
    for (degrees, index, expected) in [
        (30.0_f64, 2, Some(2)),
        (30.001, 2, None),
        (0.0, 0, None),
        (180., 0, Some(2)),
    ] {
        let next = tracing::info_span!("act", degrees).in_scope(|| {
            let (sin, cos) = degrees.to_radians().sin_cos();
            target.normal = Vector3::new([cos, sin, 0.0]);
            source.anchored_slice(index, &target)
        });
        tracing::info_span!("assert").in_scope(|| assert_eq!(next, expected));
    }
}

#[test]
fn plane_trace_clips_oblique_edge_corner_and_disjoint_cases() {
    let _trace = traced();
    let corners =
        tracing::info_span!("arrange").in_scope(|| [[0., 0.], [8., 0.], [8., 4.], [0., 4.]]);
    for (distances, expected) in [
        ([-6., 2., 6., -2.], Some([[6., 0.], [2., 4.]])), // x+y=6
        ([0., 8., 4., -4.], Some([[0., 0.], [4., 4.]])),  // x-y=0
        ([0., 0., 4., 4.], Some([[0., 0.], [8., 0.]])),
        ([0., 8., 12., 4.], None), // only one corner touches
        ([1., 9., 13., 5.], None),
        ([0.; 4], None),
    ] {
        let actual = tracing::info_span!("act").in_scope(|| PlaneTrace::clip(corners, distances));
        tracing::info_span!("assert").in_scope(|| assert_eq!(actual, expected));
    }
}

#[test]
fn single_slice_localizer_shares_a_frame_but_cannot_supply_scroll_travel() {
    let _trace = traced();
    let root = tempfile::tempdir().expect("localizer directory");
    let files = fixtures::write_oriented_study(
        root.path(),
        "CT",
        fixtures::SERIES_UID,
        &fixtures::SAMPLES,
        (Some([1.5, 2.0]), true),
        fixtures::PositionedSlices {
            origins: [[10, 20, 29], [10, 20, 30], [10, 20, 31]],
            orientation: "1\\0\\0\\0\\1\\0",
            frame: Some("2.25.259"),
        },
    )
    .expect("encode axial images");
    let single = crate::dicom::loader::load_dicom_series_from_named_bytes(&[(
        files[1].0.clone(),
        &files[1].1,
    )])
    .expect("decode a single image");
    let target = PatientStack::from_volume(&single).expect("localizer plane");
    let source =
        PatientStack::from_volume(&read([10, 12, 14], Some("2.25.259"))).expect("sagittal source");
    let trace = tracing::info_span!("act")
        .in_scope(|| source.trace_on(1, &target, 0).expect("same patient space"));
    let _check = tracing::info_span!("assert").entered();
    // Target center x=13, increasing by 2 mm per column. Sagittal source x=12.
    assert_eq!(trace.distance([-0.5, 0.]), 0.);
    assert_eq!(trace.distance([0., 0.]), 1.);
    assert!(source.trace_on(1, &source, 2).is_none());
    assert!(!source.can_anchor(&target));
    let unrelated =
        PatientStack::from_volume(&read([10, 12, 14], Some("2.25.999"))).expect("other frame");
    assert!(unrelated.trace_on(1, &target, 0).is_none());
}

#[test]
fn cursor_uses_pixel_centers_measured_slabs_and_shared_frames() {
    let _trace = traced();
    let source = tracing::info_span!("arrange").in_scope(|| {
        PatientStack::from_volume(&read([10, 12, 16], Some("2.25.259"))).expect("source stack")
    });
    let mut target =
        PatientStack::from_volume(&read([10, 12, 16], Some("2.25.259"))).expect("target stack");
    let point = tracing::info_span!("act")
        .in_scope(|| source.patient_point(1, [0.5, 1.5]).expect("selected texel"));
    let _check = tracing::info_span!("assert").entered();
    assert_eq!(point.as_slice(), &[12., 20., 31.5]);
    let continuous = source
        .patient_point(1, [1.25, 0.75])
        .expect("fractional point");
    assert_eq!(
        source
            .project_cursor(continuous, &target)
            .expect("round trip")
            .pixel,
        [1.25, 0.75]
    );
    target.half_thickness.fill(Some(0.5));
    for (x, index, distant) in [
        (9.0, 0, true),
        (10.5, 0, false),
        (10.5_f64.next_up(), 0, true),
        (11.0, 0, true),
        (11.0_f64.next_up(), 1, true),
        (12.0, 1, false),
        (19.0, 2, true),
    ] {
        let projected = source
            .project_cursor(Vector3::new([x, 20., 31.5]), &target)
            .expect("in field of view");
        assert_eq!(
            (projected.index, projected.pixel, projected.distant),
            (index, [0.5, 1.5], distant)
        );
    }
    assert!(source
        .project_cursor(Vector3::new([12., 100., 31.5]), &target)
        .is_none());
    target.frame = "2.25.260".into();
    assert!(source.project_cursor(point, &target).is_none());
}
