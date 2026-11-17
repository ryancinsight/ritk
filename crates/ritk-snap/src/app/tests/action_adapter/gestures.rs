//! Pan, window/level, annotation, and measurement gestures.

use super::super::test_volume;
use crate::app::action_adapter::ViewerActionDisposition;
use crate::app::SnapApp;
use crate::presentation::{
    PointerButton, PresentationDispatcher, PresentationEvent, PresentationModifiers,
};
use crate::tools::{interaction::ViewportOffset, kind::ToolKind};

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
            modifiers: crate::presentation::PresentationModifiers::NONE,
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

    assert_eq!(app.pan_offset, ViewportOffset::new(20.0, -10.0));
    assert!(app.tool_state.is_idle());
}

#[test]
fn window_level_actions_use_the_existing_sensitivity_mapping() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::WindowLevel;
    app.viewer_state.window_center = Some(40.0);
    app.viewer_state.window_width = Some(400.0);
    let revision = app.visual_revision;
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
            modifiers: crate::presentation::PresentationModifiers::NONE,
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
    assert!(app.visual_revision > revision);
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
            modifiers: crate::presentation::PresentationModifiers::NONE,
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
        [crate::tools::interaction::Annotation::HuPoint { pos, value, .. }]
            if *pos == [2.0, 1.0] && *value == 0.0
    ));
}

#[test]
fn arrow_drag_records_image_geometry_and_capture_loss_cancels_it() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 1]));
    app.active_tool = ToolKind::Arrow;
    let viewport = viewport([4, 4]);
    let mut dispatcher = PresentationDispatcher::new();

    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 1.0,
            y: 1.0,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 3.0, y: 2.0 },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 3.0,
            y: 2.0,
            button: PointerButton::Left,
        },
    );
    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::Arrow { tail, head, .. }]
            if *tail == [1.0, 1.0] && *head == [2.0, 3.0]
    ));

    app.annotations.clear();
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 1.0,
            y: 1.0,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 3.0, y: 2.0 },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerCaptureLost,
    );
    assert!(app.annotations.is_empty());
    assert!(app.tool_state.is_idle());
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
                modifiers: crate::presentation::PresentationModifiers::NONE,
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
            plane,
            location: Some(crate::tools::interaction::AnnotationLocation { plane: 0, slice: 0 }),
            p1,
            p2,
            length_mm,
        }] if *plane == 0
            && *p1 == [0.5, 0.5]
            && *p2 == [0.5, 3.25]
            && (*length_mm - 2.75).abs() < 1e-6
    ));
}

#[test]
fn deviation_measurement_completes_from_one_pointer_drag() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 1]));
    app.active_tool = ToolKind::MeasureDeviation;
    let viewport = viewport([4, 4]);
    let mut dispatcher = PresentationDispatcher::new();

    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 0.5,
            y: 0.5,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 2.5, y: 2.5 },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 2.5,
            y: 2.5,
            button: PointerButton::Left,
        },
    );

    assert!(app.tool_state.is_idle());
    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::Deviation {
            start,
            end,
            reference,
            distance_mm,
            angle_deg,
            ..
        }] if *start == [0.5, 0.5]
            && *end == [2.5, 2.5]
            && *reference == [0.5, 2.5]
            && (*distance_mm - 2.0).abs() < 1e-6
            && (*angle_deg - 45.0).abs() < 1e-6
    ));
}

#[test]
fn deviation_drag_rejects_release_after_slice_switch() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 2]));
    app.active_tool = ToolKind::MeasureDeviation;
    let viewport = viewport([4, 4]);
    let mut dispatcher = PresentationDispatcher::new();

    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 0.5,
            y: 0.5,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
    );
    app.set_slice_for_axis(0, 1);
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 2.5, y: 3.25 },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 2.5,
            y: 3.25,
            button: PointerButton::Left,
        },
    );

    assert!(app.tool_state.is_idle());
    assert!(app.annotations.is_empty());
}

#[test]
fn escape_discards_a_live_deviation_and_ignores_its_release() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 4, 4]));
    app.select_tool(ToolKind::MeasureDeviation);
    let viewport = viewport([4, 4]);
    for event in [
        PresentationEvent::PointerDown {
            x: 0.5,
            y: 0.5,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
        PresentationEvent::PointerMove { x: 2.5, y: 3.0 },
    ] {
        apply_app_event(&mut app, &viewport, event);
    }
    assert_eq!(app.tool_state.tool_kind(), Some(ToolKind::MeasureDeviation));
    for event in [
        PresentationEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
        PresentationEvent::PointerUp {
            x: 2.5,
            y: 3.0,
            button: PointerButton::Left,
        },
    ] {
        apply_app_event(&mut app, &viewport, event);
    }
    assert_eq!(app.annotations.as_slice(), &[]);
    assert_eq!(app.tool_state.tool_kind(), None);
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
                modifiers: crate::presentation::PresentationModifiers::NONE,
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
fn angle_anchors_on_different_slices_do_not_form_one_measurement() {
    use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint};

    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    app.active_tool = ToolKind::MeasureAngle;
    app.on_click(Some(ImagePoint::new(0.5, 0.5)));
    app.on_click(Some(ImagePoint::new(0.5, 1.5)));

    app.set_slice_for_axis(0, 1);
    app.on_click(Some(ImagePoint::new(0.5, 0.5)));
    assert_eq!(app.annotations.as_slice(), &[]);
    app.on_click(Some(ImagePoint::new(1.5, 0.5)));
    app.on_click(Some(ImagePoint::new(1.5, 1.5)));
    assert_eq!(
        app.annotations.as_slice(),
        &[Annotation::Angle {
            location: Some(AnnotationLocation { plane: 0, slice: 1 }),
            p1: [0.5, 0.5],
            p2: [0.5, 1.5],
            p3: [1.5, 1.5],
            angle_deg: 90.0,
        }]
    );
}

#[test]
fn adapter_accumulates_repaint_across_actions_in_one_batch() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);
    let disposition = app
        .apply_presentation_events(
            &[
                PresentationEvent::KeyDown {
                    virtual_key: 0x28,
                    repeated: false,
                    modifiers: PresentationModifiers::NONE,
                },
                PresentationEvent::KeyUp {
                    virtual_key: 0x28,
                    modifiers: PresentationModifiers::NONE,
                },
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

#[test]
fn length_measurement_first_click_then_drag_completes_through_the_adapter() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([8, 8, 1]));
    app.active_tool = ToolKind::MeasureLength;
    let viewport = viewport([8, 8]);

    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 0.5,
            y: 0.5,
            button: PointerButton::Left,
            modifiers: crate::presentation::PresentationModifiers::NONE,
        },
    );
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerUp {
            x: 0.5,
            y: 0.5,
            button: PointerButton::Left,
        },
    );

    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 0.5,
            y: 0.5,
            button: PointerButton::Left,
            modifiers: crate::presentation::PresentationModifiers::NONE,
        },
    );
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerMove { x: 4.5, y: 3.5 },
    );
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerUp {
            x: 4.5,
            y: 3.5,
            button: PointerButton::Left,
        },
    );

    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::Length { length_mm, .. }]
            if *length_mm == 5.0
    ));
    assert!(app.tool_state.is_idle());
}
