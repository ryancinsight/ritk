//! Slice navigation and cine-loop playback tests.

use super::*;
use crate::app::slice_ops::CineTick;
use crate::ui::LinkedCursor;

#[test]
fn cine_loop_advances_and_wraps_active_axis() {
    let mut app = SnapApp::default();
    let shape = [3, 4, 5];
    app.loaded = Some(test_volume(shape));
    app.viewer_state.slice_index = 2;
    app.linked_cursor = Some(LinkedCursor::from_slices(shape, 2, 0, 0));

    app.advance_slice_for_axis_loop(0, 1);

    assert_eq!(app.viewer_state.slice_index, 0);
    assert_eq!(app.linked_cursor.expect("cursor").voxel(), [0, 0, 0]);
}

#[test]
fn cine_tick_keeps_timing_and_state_transition_host_neutral() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.cine.set_fps(10.0);
    app.cine.set_enabled(true, 0.0);

    assert_eq!(app.tick_cine_at(0.09), CineTick::Waiting);
    assert_eq!(app.tick_cine_at(0.10), CineTick::Advanced(1));
    assert_eq!(app.viewer_state.slice_index, 1);
    assert_eq!(app.tick_cine_at(0.31), CineTick::Advanced(2));
    assert_eq!(app.viewer_state.slice_index, 0);
}

#[test]
fn cine_tick_stops_when_the_study_closes() {
    let mut app = SnapApp::default();
    app.cine.set_enabled(true, 0.0);

    assert_eq!(app.tick_cine_at(1.0), CineTick::Inactive);
    assert!(!app.cine.enabled);
}

#[test]
fn cine_toggle_requires_a_study_and_resets_the_host_anchor() {
    let mut app = SnapApp::default();
    assert!(!app.toggle_cine());
    assert!(!app.cine.enabled);

    app.loaded = Some(test_volume([3, 4, 5]));
    assert!(app.toggle_cine());
    assert!(app.cine.enabled);
    assert_eq!(app.cine.consume_steps(100.0), 0);
    assert!(!app.toggle_cine());
    assert!(!app.cine.enabled);
}

/// advance_slice_for_axis_loop wraps correctly and routes through set_slice_for_axis.
///
/// Axis 0 has 3 slices; advance from index 2 by 1 step wraps to 0.
#[test]
fn advance_slice_for_axis_loop_wraps_and_advances_visual_revision() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.viewer_state.slice_index = 2; // last slice
    let revision = app.visual_revision;

    app.advance_slice_for_axis_loop(0, 1);

    assert_eq!(app.viewer_state.slice_index, 0, "wrap-around failed");
    assert!(app.visual_revision > revision);
}

#[test]
fn slice_navigation_shortcuts_advance_or_rewind_active_axis() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.axis = 0;
    app.viewer_state.slice_index = 1;

    app.apply_slice_navigation_shortcuts(true, false, false, false, false, false);
    assert_eq!(app.viewer_state.slice_index, 0);

    app.apply_slice_navigation_shortcuts(false, false, false, true, false, false);
    assert_eq!(app.viewer_state.slice_index, 1);
}

#[test]
fn slice_navigation_shortcuts_use_priority_when_multiple_keys_pressed() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.axis = 0;
    app.viewer_state.slice_index = 1;

    app.apply_slice_navigation_shortcuts(true, true, false, false, false, false);
    assert_eq!(app.viewer_state.slice_index, 0);
}

#[test]
fn slice_navigation_shortcuts_home_end_jump_to_axis_boundaries() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.axis = 2;
    app.sagittal_slice = 2;

    app.apply_slice_navigation_shortcuts(false, false, false, false, true, false);
    assert_eq!(app.sagittal_slice, 0);

    app.apply_slice_navigation_shortcuts(false, false, false, false, false, true);
    assert_eq!(app.sagittal_slice, 4);
}

#[test]
fn slice_navigation_shortcuts_home_takes_priority_over_end() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.axis = 0;
    app.viewer_state.slice_index = 1;

    app.apply_slice_navigation_shortcuts(false, false, false, false, true, true);
    assert_eq!(app.viewer_state.slice_index, 0);
}
