//! Cursor and voxel-value interaction tests.

use super::*;
use crate::ui::{LinkedCursor, RotationSteps, ViewTransform};

#[test]
fn linked_cursor_click_updates_all_slices() {
    let mut app = SnapApp::default();
    let shape = [8, 10, 20];
    app.loaded = Some(test_volume(shape));
    app.viewer_state.slice_index = 3;
    app.coronal_slice = 5;
    app.sagittal_slice = 9;
    app.linked_cursor = Some(LinkedCursor::from_slices(shape, 3, 5, 9));

    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 100.0));
    app.update_linked_cursor_from_pointer(0, Some(egui::pos2(150.0, 20.0)), rect);

    assert_eq!(app.viewer_state.slice_index, 3);
    assert_eq!(app.coronal_slice, 2);
    assert_eq!(app.sagittal_slice, 15);
    assert_eq!(app.axis, 0);
    assert_eq!(app.linked_cursor.expect("cursor").voxel(), [3, 2, 15]);
}

#[test]
fn linked_cursor_pointer_events_follow_all_display_transforms() {
    let shape = [8, 10, 20];
    let target = [3, 6, 11];
    let rect = egui::Rect::from_min_size(egui::pos2(20.0, 30.0), egui::vec2(400.0, 300.0));
    let rotations = [
        RotationSteps::Zero,
        RotationSteps::Ninety,
        RotationSteps::OneEighty,
        RotationSteps::TwoSeventy,
    ];

    let mut app = SnapApp::default();
    app.loaded = Some(test_volume(shape));
    for rotation in rotations {
        for (flip_h, flip_v) in [(false, false), (true, false), (false, true), (true, true)] {
            let transform = ViewTransform {
                flip_h,
                flip_v,
                rotation,
            };
            app.view_transform = transform;
            app.viewer_state.slice_index = target[0];
            app.coronal_slice = target[1];
            app.sagittal_slice = target[2];
            app.linked_cursor = Some(LinkedCursor::from_slices(
                shape, target[0], target[1], target[2],
            ));

            for axis in 0..3 {
                let point = app
                    .linked_cursor
                    .as_ref()
                    .and_then(|cursor| cursor.viewport_crosshair(shape, axis, rect, transform))
                    .expect("target voxel must project into the viewport");
                app.update_linked_cursor_from_pointer(axis, Some(point), rect);
                assert_eq!(
                    app.linked_cursor.expect("cursor").voxel(),
                    target,
                    "pointer projection must invert transform {transform:?} on axis {axis}"
                );
            }
        }
    }
}

#[test]
fn stepping_slice_updates_linked_cursor_axis_coordinate() {
    let mut app = SnapApp::default();
    let shape = [8, 10, 20];
    app.loaded = Some(test_volume(shape));
    app.viewer_state.slice_index = 3;
    app.coronal_slice = 5;
    app.sagittal_slice = 9;
    app.linked_cursor = Some(LinkedCursor::from_slices(shape, 3, 5, 9));

    app.step_slice_for_axis(1, 2);

    assert_eq!(app.coronal_slice, 7);
    assert_eq!(app.linked_cursor.expect("cursor").voxel(), [3, 7, 9]);
}

#[test]
fn current_cursor_value_reads_loaded_voxel_at_linked_position() {
    let mut app = SnapApp::default();
    let mut volume = test_volume([2, 3, 4]);
    volume.data = Arc::new((0..24).map(|v| v as f32).collect());
    app.loaded = Some(volume);
    app.linked_cursor = Some(LinkedCursor::from_slices([2, 3, 4], 1, 2, 3));

    assert_eq!(app.current_cursor_value(), Some(23.0));
}
