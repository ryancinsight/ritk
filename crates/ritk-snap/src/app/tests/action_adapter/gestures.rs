//! Pan, window/level, annotation, and measurement gestures.

use super::super::test_volume;
use crate::app::action_adapter::ViewerActionDisposition;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationDispatcher, PresentationEvent};
use crate::tools::kind::ToolKind;

use super::{apply_app_event, apply_events, viewport};

#[test]
fn pan_actions_update_viewer_state_without_gui_coordinates() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan;
    let viewport = viewport([64, 64]);
    let mut dispatcher = PresentationDispatcher::new();

    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 10.0,
            y: 10.0,
            button: PointerButton::Left,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 30.0, y: 0.0 },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 30.0,
            y: 0.0,
            button: PointerButton::Left,
        },
    );

    assert_eq!(app.pan_offset, egui::vec2(20.0, -10.0));
    assert!(app.tool_state.is_idle());
}

#[test]
fn window_level_actions_use_the_existing_sensitivity_mapping() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::WindowLevel;
    app.viewer_state.window_center = Some(40.0);
    app.viewer_state.window_width = Some(400.0);
    let viewport = viewport([64, 64]);
    let mut dispatcher = PresentationDispatcher::new();

    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 20.0,
            y: 20.0,
            button: PointerButton::Left,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 30.0, y: 15.0 },
    );

    assert_eq!(app.viewer_state.window_center, Some(60.0));
    assert_eq!(app.viewer_state.window_width, Some(440.0));
    assert!(app.texture_dirty);
    assert!(app.coronal_dirty);
    assert!(app.sagittal_dirty);
    assert!(app.mip_dirty);
}

#[test]
fn point_click_actions_record_a_ritk_annotation() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 1]));
    app.active_tool = ToolKind::PointHu;
    let viewport = viewport([4, 4]);
    let mut dispatcher = PresentationDispatcher::new();

    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 1.0,
            y: 2.0,
            button: PointerButton::Left,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 1.0,
            y: 2.0,
            button: PointerButton::Left,
        },
    );

    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::HuPoint { pos, value }]
            if *pos == [2.0, 1.0] && *value == 0.0
    ));
}

#[test]
fn length_measurement_clicks_advance_and_complete_through_the_adapter() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 1]));
    app.active_tool = ToolKind::MeasureLength;
    let viewport = viewport([4, 4]);

    for (x, y) in [(0.5, 0.5), (3.25, 0.5)] {
        apply_app_event(
            &mut app,
            &viewport,
            PresentationEvent::PointerDown {
                x,
                y,
                button: PointerButton::Left,
            },
        );
        apply_app_event(
            &mut app,
            &viewport,
            PresentationEvent::PointerUp {
                x,
                y,
                button: PointerButton::Left,
            },
        );
    }

    assert!(app.tool_state.is_idle());
    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::Length {
            p1,
            p2,
            length_mm,
        }] if *p1 == [0.5, 0.5]
            && *p2 == [0.5, 3.25]
            && (*length_mm - 2.75).abs() < 1e-6
    ));
}

#[test]
fn angle_measurement_clicks_advance_and_complete_through_the_adapter() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 1]));
    app.active_tool = ToolKind::MeasureAngle;
    let viewport = viewport([4, 4]);

    for (x, y) in [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5)] {
        apply_app_event(
            &mut app,
            &viewport,
            PresentationEvent::PointerDown {
                x,
                y,
                button: PointerButton::Left,
            },
        );
        apply_app_event(
            &mut app,
            &viewport,
            PresentationEvent::PointerUp {
                x,
                y,
                button: PointerButton::Left,
            },
        );
    }

    assert!(app.tool_state.is_idle());
    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::Angle { angle_deg, .. }]
            if (*angle_deg - 90.0).abs() < 1e-5
    ));
}

#[test]
fn adapter_accumulates_repaint_across_actions_in_one_batch() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    app.texture_dirty = false;
    let viewport = viewport([4, 4]);
    let disposition = app
        .apply_presentation_events(
            &[
                PresentationEvent::KeyDown {
                    virtual_key: 0x22,
                    repeated: false,
                },
                PresentationEvent::KeyUp { virtual_key: 0x22 },
            ],
            Some(&viewport),
        )
        .expect("navigation batch is supported");

    assert_eq!(app.viewer_state.slice_index, 1);
    assert_eq!(
        disposition,
        ViewerActionDisposition::Continue { repaint: true }
    );
}
