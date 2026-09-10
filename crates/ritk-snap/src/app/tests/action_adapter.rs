//! End-to-end checks for the format-neutral presentation-to-viewer seam.

use super::test_volume;
use crate::app::action_adapter::{
    ViewerActionDisposition, ViewerActionError, ViewerInputError, ViewerViewport,
    ViewerViewportError,
};
use crate::app::SnapApp;
use crate::label::LabelEditor;
use crate::presentation::{
    PointerButton, PresentationDispatcher, PresentationEvent, PresentationModifiers, ViewerAction,
    ViewportPoint,
};
use crate::tools::interaction::ToolState;
use crate::tools::kind::ToolKind;
use crate::ui::{LinkedCursor, ViewTransform};
use ritk_annotation::LabelId;
use std::sync::Arc;

fn viewport(source_size: [usize; 2]) -> ViewerViewport {
    ViewerViewport::new(
        0,
        egui::Pos2::ZERO,
        egui::vec2(1.0, 1.0),
        source_size,
        ViewTransform::default(),
    )
    .expect("test viewport geometry is valid")
}

fn apply_events(
    app: &mut SnapApp,
    dispatcher: &mut PresentationDispatcher,
    viewport: &ViewerViewport,
    event: PresentationEvent,
) {
    let actions = dispatcher
        .dispatch(&[event])
        .expect("test presentation event is valid");
    for action in actions.iter() {
        app.apply_viewer_action(action, Some(viewport))
            .expect("test viewer action is supported");
    }
}

fn apply_app_event(app: &mut SnapApp, viewport: &ViewerViewport, event: PresentationEvent) {
    app.apply_presentation_events(&[event], Some(viewport))
        .expect("test presentation event is supported");
}

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

#[test]
fn wheel_actions_preserve_zoom_and_slice_navigation_semantics() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);

    app.apply_presentation_events(
        &[PresentationEvent::PointerWheel {
            x: 2.0,
            y: 2.0,
            delta_x: 0.0,
            delta_y: 120.0,
            modifiers: PresentationModifiers::new(true, false, false, false),
        }],
        Some(&viewport),
    )
    .expect("Ctrl+wheel zoom is supported");
    assert!(app.zoom > 1.0);
    assert_eq!(app.viewer_state.slice_index, 0);
    let zoomed = app.zoom;

    app.apply_presentation_events(
        &[PresentationEvent::PointerWheel {
            x: 2.0,
            y: 2.0,
            delta_x: 0.0,
            delta_y: -120.0,
            modifiers: PresentationModifiers::NONE,
        }],
        Some(&viewport),
    )
    .expect("plain wheel slice navigation is supported");
    assert_eq!(app.viewer_state.slice_index, 1);
    assert_eq!(app.zoom, zoomed);
}

#[test]
fn wheel_actions_use_meta_as_the_platform_zoom_modifier() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);
    app.apply_presentation_events(
        &[PresentationEvent::PointerWheel {
            x: 2.0,
            y: 2.0,
            delta_x: 0.0,
            delta_y: 120.0,
            modifiers: PresentationModifiers::new(false, false, false, true),
        }],
        Some(&viewport),
    )
    .expect("Meta+wheel zoom is supported");
    assert!(app.zoom > 1.0);
    assert_eq!(app.viewer_state.slice_index, 0);
}

#[test]
fn wheel_actions_ignore_events_outside_the_viewport() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);
    let disposition = app
        .apply_presentation_events(
            &[PresentationEvent::PointerWheel {
                x: 20.0,
                y: 20.0,
                delta_x: 0.0,
                delta_y: 120.0,
                modifiers: PresentationModifiers::new(true, false, false, false),
            }],
            Some(&viewport),
        )
        .expect("outside wheel is ignored");
    assert_eq!(
        disposition,
        ViewerActionDisposition::Continue { repaint: false }
    );
    assert_eq!(app.zoom, 1.0);
    assert_eq!(app.viewer_state.slice_index, 0);
}

#[test]
fn out_of_range_wheel_keeps_the_dispatcher_transactional() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);
    let error = app
        .apply_presentation_events(
            &[
                PresentationEvent::PointerDown {
                    x: 2.0,
                    y: 2.0,
                    button: PointerButton::Left,
                },
                PresentationEvent::PointerWheel {
                    x: 2.0,
                    y: 2.0,
                    delta_x: 0.0,
                    delta_y: f64::MAX,
                    modifiers: PresentationModifiers::new(true, false, false, false),
                },
            ],
            Some(&viewport),
        )
        .expect_err("out-of-range zoom delta");
    assert!(matches!(
        error,
        ViewerInputError::Action(ViewerActionError::WheelDeltaOutOfRange { .. })
    ));
    app.apply_presentation_events(
        &[PresentationEvent::PointerUp {
            x: 2.0,
            y: 2.0,
            button: PointerButton::Left,
        }],
        Some(&viewport),
    )
    .expect_err("rejected batch must not press the pointer");
    assert_eq!(app.zoom, 1.0);
}

#[test]
fn lost_pointer_cancellation_releases_dispatcher_for_the_next_press() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan;
    let viewport = viewport([16, 16]);
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 2.25,
            y: 2.75,
            button: PointerButton::Left,
        },
    );
    app.cancel_presentation_gesture();
    assert!(app.tool_state.is_idle());

    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 3.5,
            y: 3.5,
            button: PointerButton::Left,
        },
    );
    assert!(matches!(app.tool_state, ToolState::Panning { .. }));
}

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

#[test]
fn focus_loss_cancels_a_gesture_and_lifecycle_actions_exit() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan;
    let viewport = viewport([16, 16]);
    let mut dispatcher = PresentationDispatcher::new();
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerDown {
            x: 2.0,
            y: 2.0,
            button: PointerButton::Left,
        },
    );
    assert!(matches!(app.tool_state, ToolState::Panning { .. }));

    let focus_loss = ViewerAction::FocusChanged { focused: false };
    let disposition = app
        .apply_viewer_action(&focus_loss, Some(&viewport))
        .expect("focus loss is supported");
    assert_eq!(
        disposition,
        ViewerActionDisposition::Continue { repaint: true }
    );
    assert!(app.tool_state.is_idle());

    let close = ViewerAction::CloseRequested;
    assert_eq!(
        app.apply_viewer_action(&close, None)
            .expect("close is supported"),
        ViewerActionDisposition::Exit
    );
}

#[test]
fn unsupported_buttons_and_invalid_viewports_are_typed_failures() {
    let invalid_axis = ViewerViewport::new(
        3,
        egui::Pos2::ZERO,
        egui::vec2(1.0, 1.0),
        [4, 4],
        ViewTransform::default(),
    );
    assert!(matches!(
        invalid_axis,
        Err(ViewerViewportError::Axis { axis: 3 })
    ));

    let invalid_geometry = ViewerViewport::new(
        0,
        egui::Pos2::ZERO,
        egui::vec2(0.0, 1.0),
        [4, 4],
        ViewTransform::default(),
    );
    assert!(matches!(
        invalid_geometry,
        Err(ViewerViewportError::InvalidScreenGeometry)
    ));

    let mut app = SnapApp::default();
    let viewport = viewport([4, 4]);
    let right_press = ViewerAction::PointerPressed {
        button: PointerButton::Right,
        position: ViewportPoint::new(1.0, 1.0),
    };
    assert!(matches!(
        app.apply_viewer_action(&right_press, Some(&viewport)),
        Err(ViewerActionError::UnsupportedPointerButton {
            button: PointerButton::Right
        })
    ));

    let mut app = SnapApp::default();
    let result = app.apply_presentation_events(
        &[PresentationEvent::PointerDown {
            x: 1.0,
            y: 1.0,
            button: PointerButton::Right,
        }],
        Some(&viewport),
    );
    assert!(matches!(
        result,
        Err(crate::app::action_adapter::ViewerInputError::Action(
            ViewerActionError::UnsupportedPointerButton {
                button: PointerButton::Right
            }
        ))
    ));
    assert!(matches!(
        app.presentation_dispatcher
            .dispatch(&[PresentationEvent::PointerUp {
                x: 1.0,
                y: 1.0,
                button: PointerButton::Right,
            }]),
        Err(
            crate::presentation::ActionDispatchError::PointerReleaseWithoutPress {
                button: PointerButton::Right
            }
        )
    ));
}
