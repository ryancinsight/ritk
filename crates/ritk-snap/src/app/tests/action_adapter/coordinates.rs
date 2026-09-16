//! Client-to-image coordinate mapping across the seam.

use super::super::test_volume;
use crate::app::action_adapter::ViewerViewport;
use crate::app::browser_geometry::viewport_for_display;
use crate::app::SnapApp;
use crate::label::LabelEditor;
use crate::presentation::browser_coordinates::content_fraction;
use crate::presentation::{PointerButton, PresentationEvent};
use crate::tools::kind::ToolKind;
use crate::ui::{LinkedCursor, ViewTransform};
use ritk_annotation::LabelId;
use std::sync::Arc;

use super::apply_app_event;

fn hu_point(display_size: [f64; 2], point: [f64; 2]) -> [f32; 2] {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 4, 8]));
    app.active_tool = ToolKind::PointHu;
    let point = content_fraction(point, display_size).expect("measured content point is valid");
    let viewport =
        viewport_for_display(0, [1.0; 2], [8, 4]).expect("browser display geometry is valid");
    for event in [
        PresentationEvent::PointerDown {
            x: point[0],
            y: point[1],
            button: PointerButton::Left,
        },
        PresentationEvent::PointerUp {
            x: point[0],
            y: point[1],
            button: PointerButton::Left,
        },
    ] {
        apply_app_event(&mut app, &viewport, event);
    }
    match app.annotations.as_slice() {
        // HuPoint persists positions as [row, column]; expose [x, y] here to
        // compare the ViewerViewport mapping in ImagePoint coordinates.
        [crate::tools::interaction::Annotation::HuPoint { pos, .. }] => [pos[1], pos[0]],
        annotations => panic!("expected one HU point annotation, found {annotations:?}"),
    }
}

#[test]
fn browser_css_geometry_maps_corners_center_and_resize() {
    let cases = [
        ([5.0, 5.0], [0.5, 0.5]),
        ([75.0, 5.0], [7.5, 0.5]),
        ([5.0, 35.0], [0.5, 3.5]),
        ([75.0, 35.0], [7.5, 3.5]),
        ([40.0, 20.0], [4.0, 2.0]),
    ];
    for (point, expected) in cases {
        assert_eq!(hu_point([80.0, 40.0], point), expected);
        assert_eq!(
            hu_point([160.0, 80.0], point.map(|coordinate| coordinate * 2.0)),
            expected
        );
    }
    assert_eq!(hu_point([160.0, 80.0], [40.0, 20.0]), [2.0, 1.0]);
}

#[test]
fn fractional_content_pixels_preserve_voxel_positions() {
    // The browser provider removes borders, padding and ancestor transforms.
    // These are its content-local dimensions and positions, not visual bounds.
    // Binary fractions make the expected voxel positions exactly representable.
    for size in [[80.5, 40.25], [161.0, 120.75], [40.25, 80.5]] {
        for (fraction, voxel) in [
            ([0.0625, 0.125], [0.5, 0.5]),
            ([0.9375, 0.875], [7.5, 3.5]),
            ([0.5, 0.5], [4.0, 2.0]),
        ] {
            assert_eq!(
                hu_point(size, [size[0] * fraction[0], size[1] * fraction[1]]),
                voxel
            );
        }
    }
}

#[test]
fn content_local_click_selects_same_voxel_on_every_plane() {
    let shape = [3, 4, 8];
    for (axis, frame, image_point) in [
        (0, [8, 4], [6.5, 2.5]),
        (1, [8, 3], [6.5, 1.5]),
        (2, [4, 3], [2.5, 1.5]),
    ] {
        for texel_size in [[1.0, 1.0], [10.0625, 20.125], [20.125, 10.0625]] {
            let mut app = SnapApp::default();
            app.loaded = Some(test_volume(shape));
            app.viewer_state.slice_index = usize::from(axis == 0);
            app.coronal_slice = if axis == 1 { 2 } else { 0 };
            app.sagittal_slice = if axis == 2 { 6 } else { 0 };
            app.linked_cursor = Some(LinkedCursor::from_slices(shape, 0, 0, 0));
            let viewport = viewport_for_display(axis, [1.0; 2], frame)
                .expect("content-local viewport is finite");
            let [x, y] = content_fraction(
                [
                    image_point[0] * texel_size[0],
                    image_point[1] * texel_size[1],
                ],
                [
                    f64::from(frame[0]) * texel_size[0],
                    f64::from(frame[1]) * texel_size[1],
                ],
            )
            .expect("measured content point is valid");
            for event in [
                PresentationEvent::PointerDown {
                    x,
                    y,
                    button: PointerButton::Left,
                },
                PresentationEvent::PointerUp {
                    x,
                    y,
                    button: PointerButton::Left,
                },
            ] {
                apply_app_event(&mut app, &viewport, event);
            }
            assert_eq!(app.linked_cursor.expect("loaded cursor").voxel(), [1, 2, 6]);
            assert_eq!(
                [
                    app.viewer_state.slice_index,
                    app.coronal_slice,
                    app.sagittal_slice
                ],
                [1, 2, 6]
            );
        }
    }
}

#[test]
fn fractional_client_coordinates_reach_image_mapping_unchanged() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([8, 8, 1]));
    app.active_tool = ToolKind::PointHu;
    let viewport = ViewerViewport::new(
        0,
        [10.25, 20.5],
        [0.25, 0.5],
        [8, 8],
        ViewTransform::default(),
    )
    .expect("fractional viewport geometry is valid");

    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 10.625,
            y: 21.25,
            button: PointerButton::Left,
        },
    );
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerUp {
            x: 10.625,
            y: 21.25,
            button: PointerButton::Left,
        },
    );

    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::HuPoint { pos, .. }]
            if *pos == [1.5, 1.5]
    ));
}

#[test]
fn large_client_coordinates_subtract_before_image_narrowing() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 1]));
    app.active_tool = ToolKind::PointHu;
    let viewport = ViewerViewport::new(
        0,
        [16_777_216.0, 0.0],
        [1.0, 1.0],
        [4, 4],
        ViewTransform::default(),
    )
    .expect("large viewport geometry is valid");

    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 16_777_217.0,
            y: 1.0,
            button: PointerButton::Left,
        },
    );
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerUp {
            x: 16_777_217.0,
            y: 1.0,
            button: PointerButton::Left,
        },
    );

    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::HuPoint { pos, .. }]
            if *pos == [1.0, 1.0]
    ));
}

#[test]
fn large_client_coordinates_feed_image_space_consumers() {
    let viewport = ViewerViewport::new(
        0,
        [16_777_216.0, 0.0],
        [1.0, 1.0],
        [5, 3],
        ViewTransform::default(),
    )
    .expect("large viewport geometry is valid");

    let mut intensity_app = SnapApp::default();
    let mut intensity_volume = test_volume([2, 3, 5]);
    intensity_volume.data = Arc::new((0..30).map(|value| value as f32).collect());
    intensity_app.loaded = Some(intensity_volume);
    apply_app_event(
        &mut intensity_app,
        &viewport,
        PresentationEvent::PointerMove {
            x: 16_777_217.0,
            y: 2.0,
        },
    );
    assert_eq!(intensity_app.pointer_intensity, 11.0);

    let mut cursor_app = SnapApp::default();
    cursor_app.loaded = Some(test_volume([2, 3, 5]));
    cursor_app.linked_cursor = Some(LinkedCursor::from_slices([2, 3, 5], 0, 0, 0));
    for event in [
        PresentationEvent::PointerDown {
            x: 16_777_217.0,
            y: 2.0,
            button: PointerButton::Left,
        },
        PresentationEvent::PointerUp {
            x: 16_777_217.0,
            y: 2.0,
            button: PointerButton::Left,
        },
    ] {
        apply_app_event(&mut cursor_app, &viewport, event);
    }
    assert_eq!(
        cursor_app.linked_cursor.expect("linked cursor").voxel(),
        [0, 2, 1]
    );

    let mut label_app = SnapApp::default();
    label_app.loaded = Some(test_volume([2, 3, 5]));
    label_app.label_editor = Some(LabelEditor::new([2, 3, 5]));
    label_app.label_brush_radius = 0;
    label_app.active_tool = ToolKind::LabelPaint;
    apply_app_event(
        &mut label_app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 16_777_217.0,
            y: 2.0,
            button: PointerButton::Left,
        },
    );
    let labels = label_app.label_editor.expect("label editor");
    assert_eq!(labels.current_map().label_at([0, 2, 1]), LabelId(1));
    assert_eq!(labels.current_map().label_at([0, 2, 0]), LabelId(0));
}
