//! Client-to-image coordinate mapping across the seam.

use super::super::test_volume;
use crate::app::action_adapter::ViewerViewport;
use crate::app::SnapApp;
use crate::label::LabelEditor;
use crate::presentation::{PointerButton, PresentationEvent};
use crate::tools::kind::ToolKind;
use crate::ui::{LinkedCursor, ViewTransform};
use ritk_annotation::LabelId;
use std::sync::Arc;

use super::apply_app_event;

#[test]
fn fractional_client_coordinates_reach_image_mapping_unchanged() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([8, 8, 1]));
    app.active_tool = ToolKind::PointHu;
    let viewport = ViewerViewport::new(
        0,
        egui::Pos2::new(10.25, 20.5),
        egui::vec2(0.25, 0.5),
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
        egui::Pos2::new(16_777_216.0, 0.0),
        egui::vec2(1.0, 1.0),
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
        egui::Pos2::new(16_777_216.0, 0.0),
        egui::vec2(1.0, 1.0),
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
