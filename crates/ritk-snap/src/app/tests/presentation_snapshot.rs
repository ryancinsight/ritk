use super::test_volume;
use crate::app::state::SnapApp;
use crate::render::WindowLevel;
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
}
