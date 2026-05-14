//! Slice navigation and cine-loop playback tests.

use super::*;
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

/// advance_slice_for_axis_loop wraps correctly and routes through set_slice_for_axis.
///
/// Axis 0 has 3 slices; advance from index 2 by 1 step wraps to 0.
#[test]
fn advance_slice_for_axis_loop_wraps_and_marks_dirty() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 4, 5]));
    app.viewer_state.slice_index = 2; // last slice

    app.advance_slice_for_axis_loop(0, 1);

    assert_eq!(app.viewer_state.slice_index, 0, "wrap-around failed");
    assert!(app.texture_dirty, "texture dirty not set after advance");
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
