use super::*;
use crate::tools::interaction::Annotation;

// ─── Helper: build a canonical snapshot with known values ─────────────────────

fn canonical_snapshot_no_annotations() -> ViewerSessionSnapshot {
    ViewerSessionSnapshot {
        source: Some(PathBuf::from("C:/studies/DICOMDIR")),
        viewer_state: ViewerState {
            slice_index: 12,
            window_center: Some(40.0),
            window_width: Some(400.0),
        },
        colormap: Colormap::Bone,
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
/// Values are chosen so that equality is exact under f32 bit representation:
/// - Length: |p2 - p1| in each axis is 3.0 and 4.0 px → Pythagoras gives
///   length_mm = √((3×1.0)² + (4×1.0)²) = 5.0 mm exactly.
/// - Angle: 90° vertex — rays (0,1)→(0,0) and (0,0)→(1,0) are orthogonal.
/// - HU point: exact integer-valued position and intensity.
fn all_annotation_variants() -> Vec<Annotation> {
    vec![
        Annotation::Length {
            p1: [0.0, 0.0],
            p2: [3.0, 4.0],
            length_mm: 5.0,
        },
        Annotation::Angle {
            p1: [0.0, 1.0],
            p2: [0.0, 0.0],
            p3: [1.0, 0.0],
            angle_deg: 90.0,
        },
        Annotation::RoiRect {
            top_left: [10.0, 10.0],
            bottom_right: [20.0, 30.0],
            mean: 42.5,
            std_dev: 1.25,
            min: 40.0,
            max: 45.0,
            area_mm2: 200.0,
        },
        Annotation::RoiEllipse {
            center: [15.0, 20.0],
            radii: [5.0, 10.0],
            mean: 37.0,
            std_dev: 2.0,
            min: 33.0,
            max: 41.0,
            area_mm2: 314.159_27,
        },
        Annotation::HuPoint {
            pos: [5.0, 8.0],
            value: -150.0,
        },
    ]
}

// ─── Snapshot defaults ────────────────────────────────────────────────────────

#[test]
fn session_snapshot_default_matches_viewer_defaults() {
    let snapshot = ViewerSessionSnapshot::default();

    assert_eq!(snapshot.source, None);
    assert_eq!(snapshot.viewer_state.slice_index, 0);
    assert_eq!(snapshot.viewer_state.window_center, None);
    assert_eq!(snapshot.viewer_state.window_width, None);
    assert_eq!(snapshot.colormap, Colormap::Grayscale);
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
    assert!(snapshot.annotations.is_empty(), "default annotations must be empty");
}

// ─── JSON in-memory round-trips ───────────────────────────────────────────────

#[test]
fn session_snapshot_json_round_trip_preserves_values_no_annotations() {
    let snapshot = canonical_snapshot_no_annotations();
    let json = serde_json::to_string_pretty(&snapshot).expect("serialize snapshot");
    let recovered: ViewerSessionSnapshot =
        serde_json::from_str(&json).expect("deserialize snapshot");
    assert_eq!(recovered, snapshot);
}

#[test]
fn session_snapshot_json_round_trip_preserves_all_annotation_variants() {
    let mut snapshot = canonical_snapshot_no_annotations();
    snapshot.annotations = all_annotation_variants();
    assert_eq!(snapshot.annotations.len(), 5, "expected 5 annotation variants");

    let json = serde_json::to_string_pretty(&snapshot).expect("serialize snapshot");
    let recovered: ViewerSessionSnapshot =
        serde_json::from_str(&json).expect("deserialize snapshot");

    assert_eq!(recovered.annotations.len(), 5);

    // Verify Length annotation values round-trip exactly.
    match &recovered.annotations[0] {
        Annotation::Length { p1, p2, length_mm } => {
            assert_eq!(*p1, [0.0f32, 0.0f32]);
            assert_eq!(*p2, [3.0f32, 4.0f32]);
            assert_eq!(*length_mm, 5.0f32, "5-4-3 right triangle: length = 5 mm");
        }
        other => panic!("expected Length, got {:?}", other),
    }

    // Verify Angle annotation values round-trip exactly.
    match &recovered.annotations[1] {
        Annotation::Angle { p1, p2, p3, angle_deg } => {
            assert_eq!(*p1, [0.0f32, 1.0f32]);
            assert_eq!(*p2, [0.0f32, 0.0f32]);
            assert_eq!(*p3, [1.0f32, 0.0f32]);
            assert_eq!(*angle_deg, 90.0f32, "orthogonal rays form 90°");
        }
        other => panic!("expected Angle, got {:?}", other),
    }

    // Verify HU point annotation round-trips negative value exactly.
    match &recovered.annotations[4] {
        Annotation::HuPoint { pos, value } => {
            assert_eq!(*pos, [5.0f32, 8.0f32]);
            assert_eq!(*value, -150.0f32);
        }
        other => panic!("expected HuPoint, got {:?}", other),
    }
}

#[test]
fn session_snapshot_json_annotations_field_defaults_to_empty_when_absent() {
    // Validate backward compatibility: a JSON object lacking the "annotations"
    // field deserializes to an empty Vec (enforced by #[serde(default)]).
    let json_without_annotations = r#"{
        "source": null,
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

    let recovered: ViewerSessionSnapshot =
        serde_json::from_str(json_without_annotations)
            .expect("must deserialize legacy JSON without annotations field");

    assert!(
        recovered.annotations.is_empty(),
        "annotations must default to empty for legacy JSON; got {:?}",
        recovered.annotations
    );
}

// ─── File I/O SSOT ────────────────────────────────────────────────────────────

#[test]
fn save_to_file_and_load_from_file_round_trip_with_annotations() {
    let mut snapshot = canonical_snapshot_no_annotations();
    snapshot.annotations = all_annotation_variants();

    let dir = std::env::temp_dir();
    let path = dir.join("ritk_snap_session_test_round_trip.json");

    save_to_file(&snapshot, &path).expect("save_to_file must succeed");
    let recovered = load_from_file(&path).expect("load_from_file must succeed");

    // Confirm annotations survive the file round-trip.
    assert_eq!(
        recovered.annotations.len(),
        snapshot.annotations.len(),
        "annotation count must be preserved through file round-trip"
    );

    // Confirm navigation state survives.
    assert_eq!(recovered.viewer_state.slice_index, 12);
    assert_eq!(recovered.colormap, Colormap::Bone);
    assert_eq!(recovered.axis, 2);
    assert_eq!(recovered.zoom, 2.5f32);

    // Clean up temp file.
    let _ = std::fs::remove_file(&path);
}

#[test]
fn save_to_file_produces_valid_json_with_annotations_key() {
    let mut snapshot = ViewerSessionSnapshot::default();
    snapshot.annotations.push(Annotation::HuPoint {
        pos: [1.0, 2.0],
        value: 100.0,
    });

    let dir = std::env::temp_dir();
    let path = dir.join("ritk_snap_session_test_annotations_key.json");

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
    assert!(result.is_err(), "load_from_file must fail for nonexistent path");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("session") || msg.contains("nonexistent") || msg.contains("No such"),
        "error must reference the path; got: {}",
        msg
    );
}

#[test]
fn load_from_file_returns_error_for_invalid_json() {
    let dir = std::env::temp_dir();
    let path = dir.join("ritk_snap_session_test_invalid.json");
    std::fs::write(&path, b"not valid json {{{{").expect("write invalid file");

    let result = load_from_file(&path);
    assert!(result.is_err(), "load_from_file must fail for invalid JSON");

    let _ = std::fs::remove_file(&path);
}

