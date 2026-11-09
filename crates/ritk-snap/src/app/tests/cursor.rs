//! Cursor and voxel-value interaction tests.

use super::*;
use crate::app::viewer_viewport::ViewerViewport;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::ImagePoint;
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

    app.update_linked_cursor_from_pointer(0, Some(ImagePoint::new(15.0, 2.0)));

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
    // Centres of the target voxel in each axis's `[x, y]` image coordinates.
    let centres: [[f32; 2]; 3] = [[11.5, 6.5], [11.5, 3.5], [6.5, 3.5]];
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
            for (axis, centre) in centres.into_iter().enumerate() {
                let (width, height) =
                    crate::ui::axis_slice_dimensions(shape, axis).expect("axis dimensions");
                let output =
                    transform.source_to_output_coordinates(centre.map(f64::from), [width, height]);
                let viewport =
                    ViewerViewport::new(axis, [20.0, 30.0], [2.0, 2.0], [width, height], transform)
                        .expect("valid viewport geometry");
                let pointer = viewport
                    .map(ViewportPoint::new(
                        2.0_f64.mul_add(output[0], 20.0),
                        2.0_f64.mul_add(output[1], 30.0),
                    ))
                    .expect("the target voxel is inside the displayed image");

                app.view_transform = transform;
                app.viewer_state.slice_index = target[0];
                app.coronal_slice = target[1];
                app.sagittal_slice = target[2];
                app.linked_cursor = Some(LinkedCursor::from_slices(shape, 0, 0, 0));
                app.update_linked_cursor_from_pointer(axis, Some(pointer));
                assert_eq!(
                    app.linked_cursor.expect("cursor").voxel(),
                    target,
                    "pointer must invert {transform:?} on axis {axis}"
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
