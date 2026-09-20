use super::test_volume;
use crate::app::state::SnapApp;
use crate::render::WindowLevel;
use crate::tools::interaction::Annotation;
use crate::tools::interaction::ViewportOffset;
use crate::tools::kind::ToolKind;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

#[test]
fn empty_snapshot_contains_stable_defaults() {
    let app = SnapApp::default();
    let snapshot = app.presentation_snapshot();

    assert!(!snapshot.loaded());
    assert_eq!(snapshot.revision(), 0);
    assert_eq!(snapshot.axis(), 0);
    assert_eq!(snapshot.slice_indices(), [0, 0, 0]);
    assert_eq!(snapshot.slice_counts(), [1, 1, 1]);
    assert_eq!(
        snapshot.window_level(),
        WindowLevel::new(
            f64::from(DEFAULT_WINDOW_CENTER),
            f64::from(DEFAULT_WINDOW_WIDTH)
        )
    );
    assert!(!snapshot.cine_enabled());
    assert_eq!(snapshot.cine_fps(), 12.0);
    assert_eq!(snapshot.zoom(), 1.0);
    assert_eq!(snapshot.pan(), ViewportOffset::new(0.0, 0.0));
    assert_eq!(snapshot.window_preset_index(), None);
    assert_eq!(snapshot.active_tool_index(), 2);
    assert_eq!(snapshot.active_tool_name(), "W/L");
    assert_eq!(snapshot.annotation_count(), 0);
    assert_eq!(snapshot.last_annotation(), None);
}

#[test]
fn loaded_snapshot_projects_all_host_visible_state() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 3, 2]));
    app.axis = 2;
    app.viewer_state.slice_index = 3;
    app.coronal_slice = 2;
    app.sagittal_slice = 1;
    app.viewer_state.window_center = Some(500.0);
    app.viewer_state.window_width = Some(800.0);
    app.cine.set_enabled(true, 0.0);
    app.cine.set_fps(18.0);
    app.zoom = 1.75;
    app.pan_offset = ViewportOffset::new(12.0, -8.0);
    app.active_tool = ToolKind::Pan;
    app.bump_visual_revision();

    let snapshot = app
        .presentation_snapshot()
        .with_window_preset_index(Some(4));

    assert!(snapshot.loaded());
    assert_eq!(snapshot.revision(), 1);
    assert_eq!(snapshot.axis(), 2);
    assert_eq!(snapshot.slice_indices(), [3, 2, 1]);
    assert_eq!(snapshot.slice_counts(), [4, 3, 2]);
    assert_eq!(snapshot.window_level(), WindowLevel::new(500.0, 800.0));
    assert!(snapshot.cine_enabled());
    assert_eq!(snapshot.cine_fps(), 18.0);
    assert_eq!(snapshot.zoom(), 1.75);
    assert_eq!(snapshot.pan(), ViewportOffset::new(12.0, -8.0));
    assert_eq!(snapshot.window_preset_index(), Some(4));
    assert_eq!(snapshot.active_tool_index(), 0);
    assert_eq!(snapshot.active_tool_name(), "Pan");
    assert_eq!(snapshot.annotation_count(), 0);
    assert_eq!(snapshot.last_annotation(), None);
}

#[test]
fn snapshot_projects_latest_annotation_kind_and_primary_value() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 3, 2]));
    app.annotations.push(Annotation::HuPoint {
        pos: [1.0, 2.0],
        value: -42.5,
    });

    let snapshot = app.presentation_snapshot();

    assert_eq!(snapshot.annotation_count(), 1);
    let summary = snapshot
        .last_annotation()
        .expect("invariant: inserted annotation is projected");
    assert_eq!(summary.kind(), crate::presentation::AnnotationKind::HuPoint);
    assert_eq!(summary.primary_value(), -42.5);
}
