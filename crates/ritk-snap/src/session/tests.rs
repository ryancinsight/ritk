#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use crate::geometry::PatientPointMm;
use crate::tools::interaction::{Annotation, AnnotationLocation, PatientLength};
use ritk_core::rejection::assert_rejects;
use std::path::PathBuf;

fn canonical_snapshot_no_annotations() -> ViewerSessionSnapshot {
    ViewerSessionSnapshot {
        format: SessionFormat,
        source: Some(StudySource::Path(PathBuf::from("C:/studies/DICOMDIR"))),
        viewer_state: ViewerState {
            slice_index: 12,
            window_center: Some(40.0),
            window_width: Some(400.0),
        },
        colormap: NamedColorMap::Bone,
        axis: 2,
        active_tool: ToolKind::MeasureAngle,
        multi_planar: true,
        show_overlay: false,
        show_crosshair: true,
        show_rt_struct_overlay: false,
        show_rt_dose_overlay: false,
        rt_dose_opacity: 0.5,
        show_series_browser: false,
        sidebar_tab: SidebarTab::Metadata,
        coronal_slice: 7,
        sagittal_slice: 9,
        pan_offset: [11.5, -3.25],
        zoom: 2.5,
        cine_enabled: true,
        cine_fps: 18.0,
        annotations: Vec::new(),
    }
}

/// Build all Annotation variants with analytically exact values.
///
/// The stored length deliberately differs from the 3-4-5 endpoint geometry;
/// persistence must retain historical results instead of recomputing them.
pub(super) fn all_annotation_variants() -> Vec<Annotation> {
    vec![
        Annotation::Length {
            plane: 0,
            location: None,
            p1: [0.0, 0.0],
            p2: [3.0, 4.0],
            length_mm: 123.25,
        },
        Annotation::PatientLength {
            plane: 2,
            location: None,
            length: PatientLength::try_new(
                PatientPointMm::try_new([0.0, 0.0, 0.0]).expect("finite patient point"),
                PatientPointMm::try_new([3.0, 4.0, 0.0]).expect("finite patient point"),
            )
            .expect("3-4-5 segment is representable"),
        },
        Annotation::Angle {
            location: None,
            p1: [0.0, 1.0],
            p2: [0.0, 0.0],
            p3: [1.0, 0.0],
            angle_deg: 90.0,
        },
        Annotation::RoiRect {
            location: None,
            top_left: [10.0, 10.0],
            bottom_right: [20.0, 30.0],
            mean: 42.5,
            std_dev: 1.25,
            min: 40.0,
            max: 45.0,
            area_mm2: 200.0,
        },
        Annotation::RoiEllipse {
            location: None,
            center: [15.0, 20.0],
            radii: [5.0, 10.0],
            mean: 37.0,
            std_dev: 2.0,
            min: 33.0,
            max: 41.0,
            area_mm2: 314.159_27,
        },
        Annotation::HuPoint {
            location: None,
            pos: [5.0, 8.0],
            value: -150.0,
        },
        Annotation::Deviation {
            location: Some(AnnotationLocation { plane: 1, slice: 4 }),
            start: [2.0, 3.0],
            end: [5.0, 7.0],
            reference: [5.0, 3.0],
            distance_mm: 12.5,
            angle_deg: 53.0,
        },
        Annotation::ClosedPolygon {
            location: Some(AnnotationLocation { plane: 2, slice: 6 }),
            points: vec![[1.0, 1.0], [1.0, 6.0], [4.0, 1.0]],
            area_mm2: 77.25,
            perimeter_mm: 44.5,
        },
        Annotation::OpenPolygon {
            location: None,
            points: vec![[2.0, 2.0], [4.0, 7.0], [9.0, 7.0]],
            length_mm: 31.75,
        },
        Annotation::Arrow {
            location: Some(AnnotationLocation { plane: 0, slice: 8 }),
            tail: [3.0, 4.0],
            head: [8.0, 12.0],
        },
        Annotation::Pencil {
            location: None,
            points: vec![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]],
        },
        Annotation::CobbAngle {
            location: Some(AnnotationLocation {
                plane: 1,
                slice: 10,
            }),
            points: [[0.0, 0.0], [3.0, 1.0], [5.0, 8.0], [9.0, 9.0]],
            angle_deg: 42.25,
        },
    ]
}

#[test]
fn session_snapshot_default_matches_viewer_defaults() {
    let snapshot = ViewerSessionSnapshot::default();

    assert_eq!(snapshot.source, None);
    assert_eq!(snapshot.viewer_state.slice_index, 0);
    assert_eq!(snapshot.viewer_state.window_center, None);
    assert_eq!(snapshot.viewer_state.window_width, None);
    assert_eq!(snapshot.colormap, NamedColorMap::Grayscale);
    assert_eq!(snapshot.axis, 0);
    assert_eq!(snapshot.active_tool, ToolKind::WindowLevel);
    assert!(!snapshot.multi_planar);
    assert!(snapshot.show_overlay);
    assert!(!snapshot.show_crosshair);
    assert!(snapshot.show_rt_struct_overlay);
    assert!(snapshot.show_series_browser);
    assert_eq!(snapshot.sidebar_tab, SidebarTab::Series);
    assert_eq!(snapshot.coronal_slice, 0);
    assert_eq!(snapshot.sagittal_slice, 0);
    assert_eq!(snapshot.pan_offset, [0.0, 0.0]);
    assert_eq!(snapshot.zoom, 1.0);
    assert!(!snapshot.cine_enabled);
    assert_eq!(snapshot.cine_fps, 12.0);
    assert!(
        snapshot.annotations.is_empty(),
        "default annotations must be empty"
    );
}

#[test]
fn session_snapshot_json_round_trip_preserves_values_no_annotations() {
    let snapshot = canonical_snapshot_no_annotations();
    let json = serde_json::to_string_pretty(&snapshot).expect("serialize snapshot");
    let serialized: serde_json::Value = serde_json::from_str(&json).expect("parse session JSON");
    assert_eq!(serialized["format"], 3);
    let recovered: ViewerSessionSnapshot =
        serde_json::from_str(&json).expect("deserialize snapshot");
    assert_eq!(recovered, snapshot);
}

#[test]
fn session_snapshot_json_round_trip_preserves_all_annotation_variants() {
    let mut snapshot = canonical_snapshot_no_annotations();
    snapshot.annotations = all_annotation_variants();
    assert_eq!(
        snapshot.annotations.len(),
        12,
        "every annotation variant must be represented"
    );

    let json = serde_json::to_string_pretty(&snapshot).expect("serialize snapshot");
    let serialized: serde_json::Value = serde_json::from_str(&json).expect("parse session JSON");
    assert_eq!(serialized["format"], 3);
    let recovered: ViewerSessionSnapshot =
        serde_json::from_str(&json).expect("deserialize snapshot");

    assert_eq!(recovered, snapshot);

    // Verify Length annotation values round-trip exactly.
    match &recovered.annotations[0] {
        Annotation::Length {
            plane,
            p1,
            p2,
            length_mm,
            ..
        } => {
            assert_eq!(*plane, 0);
            assert_eq!(*p1, [0.0f32, 0.0f32]);
            assert_eq!(*p2, [3.0f32, 4.0f32]);
            assert_eq!(
                *length_mm, 123.25,
                "stored historical result is authoritative"
            );
        }
        other => panic!("expected Length, got {:?}", other),
    }

    // Verify patient-space endpoints and derived distance round-trip exactly.
    match &recovered.annotations[1] {
        Annotation::PatientLength { plane, length, .. } => {
            assert_eq!(*plane, 2, "non-default plane round-trips");
            assert_eq!(length.start_mm().coordinates(), [0.0, 0.0, 0.0]);
            assert_eq!(length.end_mm().coordinates(), [3.0, 4.0, 0.0]);
            assert_eq!(length.length_mm(), 5.0);
        }
        other => panic!("expected PatientLength, got {:?}", other),
    }

    // Verify Angle annotation values round-trip exactly.
    match &recovered.annotations[2] {
        Annotation::Angle {
            p1,
            p2,
            p3,
            angle_deg,
            ..
        } => {
            assert_eq!(*p1, [0.0f32, 1.0f32]);
            assert_eq!(*p2, [0.0f32, 0.0f32]);
            assert_eq!(*p3, [1.0f32, 0.0f32]);
            assert_eq!(*angle_deg, 90.0f32, "orthogonal rays form 90°");
        }
        other => panic!("expected Angle, got {:?}", other),
    }

    // Verify HU point annotation round-trips negative value exactly.
    match &recovered.annotations[5] {
        Annotation::HuPoint { pos, value, .. } => {
            assert_eq!(*pos, [5.0f32, 8.0f32]);
            assert_eq!(*value, -150.0f32);
        }
        other => panic!("expected HuPoint, got {:?}", other),
    }
}

#[test]
fn session_snapshot_json_omitted_fields_retain_defaults() {
    // Validate backward compatibility for omitted optional and defaulted fields.
    let json_without_annotations = r#"{
        "viewer_state": {"slice_index": 0, "window_center": null, "window_width": null},
        "colormap": "Grayscale",
        "axis": 0,
        "active_tool": "WindowLevel",
        "multi_planar": false,
        "show_overlay": true,
        "show_crosshair": false,
        "show_rt_struct_overlay": true,
        "show_series_browser": true,
        "sidebar_tab": "Series",
        "coronal_slice": 0,
        "sagittal_slice": 0,
        "pan_offset": [0.0, 0.0],
        "zoom": 1.0,
        "cine_enabled": false,
        "cine_fps": 12.0
    }"#;

    let recovered: ViewerSessionSnapshot = serde_json::from_str(json_without_annotations)
        .expect("must deserialize legacy JSON without annotations field");

    assert_eq!(recovered.format, SessionFormat);
    assert_eq!(recovered.source, None);
    assert!(!recovered.show_rt_dose_overlay);
    assert_eq!(recovered.rt_dose_opacity, 0.5);
    assert!(
        recovered.annotations.is_empty(),
        "annotations must default to empty for legacy JSON; got {:?}",
        recovered.annotations
    );
}

#[test]
fn save_to_file_and_load_from_file_round_trip_with_annotations() {
    let mut snapshot = canonical_snapshot_no_annotations();
    snapshot.annotations = all_annotation_variants();

    let path = unique_temp_path("ritk_snap_session_test_round_trip", "json");

    save_to_file(&snapshot, &path).expect("save_to_file must succeed");
    let recovered = load_from_file(&path).expect("load_from_file must succeed");

    // Confirm annotations survive the file round-trip.
    assert_eq!(
        recovered.annotations, snapshot.annotations,
        "annotation values must survive the session file round-trip"
    );

    // Confirm navigation state survives.
    assert_eq!(recovered.viewer_state.slice_index, 12);
    assert_eq!(recovered.colormap, NamedColorMap::Bone);
    assert_eq!(recovered.axis, 2);
    assert_eq!(recovered.zoom, 2.5f32);

    // Clean up temp file.
    let _ = std::fs::remove_file(&path);
}

#[test]
fn save_to_file_produces_valid_json_with_annotations_key() {
    let mut snapshot = ViewerSessionSnapshot::default();
    snapshot.annotations.push(Annotation::HuPoint {
        location: None,
        pos: [1.0, 2.0],
        value: 100.0,
    });

    let path = unique_temp_path("ritk_snap_session_test_annotations_key", "json");

    save_to_file(&snapshot, &path).expect("save_to_file must succeed");
    let json = std::fs::read_to_string(&path).expect("read saved file");

    assert!(
        json.contains("\"annotations\""),
        "saved JSON must contain 'annotations' key; got: {}",
        &json[..json.len().min(200)]
    );
    assert!(
        json.contains("HuPoint"),
        "saved JSON must contain 'HuPoint' variant; got: {}",
        &json[..json.len().min(400)]
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_from_file_returns_error_for_nonexistent_path() {
    let path = PathBuf::from("/nonexistent/path/session.json");
    let result = load_from_file(&path);
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("session") || msg.contains("nonexistent") || msg.contains("No such"),
        "error must reference the path; got: {}",
        msg
    );
}

#[test]
fn load_from_file_returns_error_for_invalid_json() {
    let path = unique_temp_path("ritk_snap_session_test_invalid", "json");
    std::fs::write(&path, b"not valid json {{{{").expect("write invalid file");

    let result = load_from_file(&path);
    assert_rejects(result, "expected ident at line 1 column 2");

    let _ = std::fs::remove_file(&path);
}

/// Temp path unique across concurrent test processes; see ritk-mif's copy for
/// why a timestamp alone races under nextest's parallelism.
fn unique_temp_path(stem: &str, extension: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{pid}_{nanos:016x}_{seq}.{extension}"))
}

#[test]
fn session_formats_preserve_values_and_reject_unknown_versions() {
    let mut snapshot = canonical_snapshot_no_annotations();
    let legacy_annotations = all_annotation_variants()
        .into_iter()
        .filter(|annotation| {
            matches!(
                annotation,
                Annotation::Length { .. }
                    | Annotation::Angle { .. }
                    | Annotation::RoiRect { .. }
                    | Annotation::RoiEllipse { .. }
                    | Annotation::HuPoint { .. }
            )
        })
        .collect::<Vec<_>>();
    snapshot.annotations = legacy_annotations;
    let unversioned: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/session-v1.json"))
            .expect("parse format 1 fixture");
    let expected_one = snapshot.clone();
    let recovered_unversioned =
        serde_json::from_value::<ViewerSessionSnapshot>(unversioned.clone())
            .expect("unversioned format 1 session loads");
    assert_eq!(recovered_unversioned, expected_one);
    assert_eq!(
        serde_json::to_value(&recovered_unversioned).expect("serialize migrated session")["format"],
        3
    );

    let mut version_one = unversioned;
    version_one["format"] = serde_json::json!(1);
    let recovered_one = serde_json::from_value::<ViewerSessionSnapshot>(version_one)
        .expect("supported path-based session format");
    assert_eq!(recovered_one, expected_one);
    assert_eq!(
        serde_json::to_value(&recovered_one).expect("serialize migrated format 1 session")
            ["format"],
        3
    );

    let mut version_two: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/session-v2.json"))
            .expect("parse format 2 fixture");
    let mut expected_two = snapshot;
    expected_two.source = Some(StudySource::Dicom {
        series_uid: "1.2.826.0.1.3680043.8.498.1".to_owned(),
        files: vec![
            PathBuf::from("C:/studies/series/0001.dcm"),
            PathBuf::from("C:/studies/series/0002.dcm"),
        ],
    });
    let recovered_two = serde_json::from_value::<ViewerSessionSnapshot>(version_two.clone())
        .expect("supported exact-acquisition session format");
    assert_eq!(recovered_two, expected_two);
    assert_eq!(
        serde_json::to_value(&recovered_two).expect("serialize migrated format 2 session")
            ["format"],
        3
    );

    version_two["format"] = serde_json::json!(4);
    let error = serde_json::from_value::<ViewerSessionSnapshot>(version_two)
        .expect_err("unknown version rejects");
    assert!(error
        .to_string()
        .contains("unsupported viewer session format version"));
}

#[test]
fn current_session_round_trips_pencil_samples_and_image() {
    let mut snapshot = canonical_snapshot_no_annotations();
    snapshot.annotations.push(Annotation::Pencil {
        location: Some(crate::tools::interaction::AnnotationLocation { plane: 0, slice: 3 }),
        points: vec![[1.0, 2.0], [2.0, 3.0], [2.0, 4.0]],
    });
    let json = serde_json::to_vec(&snapshot).expect("serialize Pencil mark");
    let restored: ViewerSessionSnapshot =
        serde_json::from_slice(&json).expect("restore Pencil mark");
    assert_eq!(restored, snapshot);
}

#[path = "tests/representation.rs"]
mod representation;

#[path = "tests/annotation_wire.rs"]
mod annotation_wire;
