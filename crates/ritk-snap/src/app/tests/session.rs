//! Session snapshot, view reset, study close, and slice-mapping tests.

use super::*;
use crate::app::state::{ProjectionMode, SeriesLoadTarget};
use crate::render::histogram::compute_histogram;
use crate::ui::LinkedCursor;
use crate::AppLaunchOptions;

#[test]
fn session_snapshot_round_trip_preserves_cine_state() {
    let mut app = SnapApp::default();
    app.cine.restore(true, 18.0);
    let snapshot = app.session_snapshot();
    assert!(snapshot.cine_enabled);
    assert_eq!(snapshot.cine_fps, 18.0);

    let mut recovered = SnapApp::default();
    recovered.apply_session_snapshot(snapshot);
    assert!(recovered.cine.enabled);
    assert_eq!(recovered.cine.fps, 18.0);
}

#[test]
fn reset_view_to_fit_restores_canonical_transform() {
    let mut app = SnapApp::default();
    app.zoom = 3.25;
    app.pan_offset = egui::vec2(24.0, -8.0);
    app.texture_dirty = false;
    app.coronal_dirty = false;
    app.sagittal_dirty = false;
    app.mip_dirty = false;

    app.reset_view_to_fit();

    assert_eq!(app.zoom, 1.0);
    assert_eq!(app.pan_offset, egui::Vec2::ZERO);
    assert!(app.texture_dirty);
    assert!(app.coronal_dirty);
    assert!(app.sagittal_dirty);
    assert!(app.mip_dirty);
    assert_eq!(app.status_message, "Zoom reset to fit.");
}

#[test]
fn close_study_clears_loaded_and_cached_state() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([2, 2, 2]));
    app.loaded_secondary = Some(test_volume([2, 2, 2]));
    app.multi_planar = true;
    app.dual_plane = true;
    app.compare_side_by_side = true;
    app.series_load_target = SeriesLoadTarget::Secondary;
    app.linked_cursor = Some(LinkedCursor::from_slices([2, 2, 2], 1, 1, 1));
    app.pointer_intensity = 123.0;
    app.pan_offset = egui::vec2(8.0, -4.0);
    app.zoom = 3.0;
    app.cached_histogram = Some(compute_histogram(&[0.0, 1.0, 1.0, 2.0], 0.0, 2.0, 4));
    app.selected_series = Some(std::path::PathBuf::from("series"));
    app.projection_mode = ProjectionMode::Vr;

    app.close_study();

    assert!(app.loaded.is_none(), "loaded volume must be cleared");
    assert!(
        app.loaded_secondary.is_none(),
        "secondary volume must be cleared"
    );
    assert!(!app.multi_planar, "multi-planar mode must reset to false");
    assert!(!app.dual_plane, "dual-plane mode must reset to false");
    assert!(
        !app.compare_side_by_side,
        "compare mode must reset to false"
    );
    assert_eq!(
        app.series_load_target,
        SeriesLoadTarget::Primary,
        "series target must reset to primary"
    );
    assert!(app.linked_cursor.is_none(), "linked cursor must be cleared");
    assert!(
        app.cached_histogram.is_none(),
        "histogram cache must be cleared"
    );
    assert!(
        app.selected_series.is_none(),
        "selected series must be cleared"
    );
    assert_eq!(app.pointer_intensity, 0.0, "pointer intensity must reset");
    assert_eq!(app.pan_offset, egui::Vec2::ZERO, "pan must reset");
    assert_eq!(app.zoom, 1.0, "zoom must reset");
    assert_eq!(
        app.projection_mode,
        ProjectionMode::Mip,
        "projection mode must reset to MIP"
    );
    assert_eq!(app.status_message, "Study closed.");
}

#[test]
fn test_viewer_state_default() {
    let state = ViewerState::default();
    assert_eq!(state.slice_index, 0);
    assert_eq!(state.window_center, None);
    assert_eq!(state.window_width, None);
}

#[test]
fn test_app_launch_options_default_has_no_initial_path() {
    let options = AppLaunchOptions::default();
    assert_eq!(
        options.initial_path, None,
        "default launch options must not queue a startup load"
    );
}

#[test]
fn map_slice_index_between_volumes_maps_bounds_and_midpoint() {
    assert_eq!(SnapApp::map_slice_index_between_volumes(0, 300, 90), 0);
    assert_eq!(SnapApp::map_slice_index_between_volumes(299, 300, 90), 89);
    let mapped = SnapApp::map_slice_index_between_volumes(150, 300, 90);
    assert!(
        mapped >= 44 && mapped <= 45,
        "midpoint mapping should stay near the secondary midpoint"
    );
}
