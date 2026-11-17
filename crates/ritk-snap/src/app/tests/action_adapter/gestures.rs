//! Pan, window/level, annotation, and measurement gestures.

use super::super::test_volume;
use crate::app::action_adapter::ViewerActionDisposition;
use crate::app::SnapApp;
use crate::presentation::{
    PointerButton, PresentationDispatcher, PresentationEvent, PresentationModifiers,
};
use crate::tools::{
    interaction::{ImagePoint, ViewportOffset},
    kind::ToolKind,
};
use metis_platform::rasterizer::MAX_STROKE_POINTS;

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
fn arrow_and_pencil_drags_preserve_exact_image_endpoints() {
    let mut app = SnapApp::default();
    let viewport = viewport([64, 64]);
    let mut dispatcher = PresentationDispatcher::new();
    let modifiers = PresentationModifiers::NONE;

    app.active_tool = ToolKind::Arrow;
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 10.0,
            y: 20.0,
            button: PointerButton::Left,
            modifiers,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 30.0, y: 40.0 },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 30.0,
            y: 40.0,
            button: PointerButton::Left,
        },
    );
    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::Arrow { tail, head, .. }]
            if *tail == [20.0, 10.0] && *head == [40.0, 30.0]
    ));

    app.active_tool = ToolKind::Pencil;
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 5.0,
            y: 6.0,
            button: PointerButton::Left,
            modifiers,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 8.0, y: 9.0 },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 11.0,
            y: 12.0,
            button: PointerButton::Left,
        },
    );
    assert!(matches!(
        app.annotations.as_slice(),
        [
            crate::tools::interaction::Annotation::Arrow { .. },
            crate::tools::interaction::Annotation::Pencil { points, .. }
        ] if points.as_slice() == [[6.0, 5.0], [9.0, 8.0], [12.0, 11.0]]
    ));
}

#[test]
fn pencil_drag_cancels_and_reports_the_path_capacity() {
    let mut app = SnapApp::default();
    app.on_drag_start(ToolKind::Pencil, Some(ImagePoint::new(0.0, 0.0)));

    for index in 1..MAX_STROKE_POINTS {
        let coordinate = f32::from(u16::try_from(index).expect("invariant: stroke limit fits u16"));
        app.on_drag(Some(ImagePoint::new(coordinate, coordinate)));
    }

    app.on_drag(Some(ImagePoint::new(
        f32::from(u16::try_from(MAX_STROKE_POINTS).expect("invariant: stroke limit fits u16")),
        f32::from(u16::try_from(MAX_STROKE_POINTS).expect("invariant: stroke limit fits u16")),
    )));

    assert!(app.tool_state.is_idle());
    assert!(app.annotations.is_empty());
    assert_eq!(app.status_message, "Annotation path sample limit reached");
}

#[test]
fn arrow_drag_keeps_sampling_free_after_many_moves() {
    let mut app = SnapApp::default();
    app.on_drag_start(ToolKind::Arrow, Some(ImagePoint::new(1.0, 2.0)));

    for index in 3..=8192_u16 {
        let coordinate = f32::from(index);
        app.on_drag(Some(ImagePoint::new(coordinate, coordinate)));
    }
    app.on_drag_end(Some(ImagePoint::new(8192.0, 8192.0)));

    assert!(matches!(
        app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::Arrow { tail, head, .. }]
            if *tail == [2.0, 1.0] && *head == [8192.0, 8192.0]
    ));
}

#[test]
fn escape_discards_a_live_arrow_or_pencil_path() {
    let viewport = viewport([8, 8]);
    for tool in [ToolKind::Arrow, ToolKind::Pencil] {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([1, 8, 8]));
        app.select_tool(tool);
        for event in [
            PresentationEvent::PointerDown {
                x: 2.0,
                y: 2.0,
                button: PointerButton::Left,
                modifiers: PresentationModifiers::NONE,
            },
            PresentationEvent::PointerMove { x: 6.0, y: 5.0 },
        ] {
            apply_app_event(&mut app, &viewport, event);
        }
        assert_eq!(app.tool_state.tool_kind(), Some(tool));
        for event in [
            PresentationEvent::KeyDown {
                virtual_key: 0x1b,
                repeated: false,
                modifiers: PresentationModifiers::NONE,
            },
            PresentationEvent::PointerUp {
                x: 6.0,
                y: 5.0,
                button: PointerButton::Left,
            },
        ] {
            apply_app_event(&mut app, &viewport, event);
        }
        assert_eq!(app.annotations.as_slice(), &[]);
        assert_eq!(app.tool_state.tool_kind(), None);
    }
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

#[test]
fn image_drawings_cancel_when_the_source_slice_changes_during_drag() {
    use crate::tools::interaction::ImagePoint;

    for tool in [ToolKind::Arrow, ToolKind::Pencil] {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 4, 3]));
        app.active_tool = tool;
        app.on_drag_start(tool, Some(ImagePoint::new(0.5, 0.5)));
        app.on_drag(Some(ImagePoint::new(1.5, 1.5)));
        app.set_slice_for_axis(0, 1);
        app.on_drag_end(Some(ImagePoint::new(2.5, 2.5)));

        assert_eq!(app.annotations.as_slice(), &[]);
        assert_eq!(app.tool_state, crate::tools::interaction::ToolState::Idle);
    }
}
