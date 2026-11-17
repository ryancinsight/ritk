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
fn cobb_drag_actions_record_two_directional_segments() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.active_tool = ToolKind::MeasureCobb;
    let viewport = viewport([64, 64]);
    let mut dispatcher = PresentationDispatcher::new();
    let modifiers = PresentationModifiers::NONE;
    for (start, end) in [((5.0, 10.0), (25.0, 10.0)), ((10.0, 5.0), (10.0, 25.0))] {
        apply_events(
            &mut app,
            &mut dispatcher,
            &viewport,
            PresentationEvent::PointerDown {
                x: start.0,
                y: start.1,
                button: PointerButton::Left,
                modifiers,
            },
        );
        apply_events(
            &mut app,
            &mut dispatcher,
            &viewport,
            PresentationEvent::PointerUp {
                x: end.0,
                y: end.1,
                button: PointerButton::Left,
            },
        );
    }
    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::CobbAngle {
            p1, p2, p3, p4, angle_deg, ..
        }] if *p1 == [10.0, 5.0]
            && *p2 == [10.0, 25.0]
            && *p3 == [5.0, 10.0]
            && *p4 == [25.0, 10.0]
            && *angle_deg == 90.0
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
