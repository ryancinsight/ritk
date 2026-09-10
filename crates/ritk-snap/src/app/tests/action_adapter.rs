//! End-to-end checks for the format-neutral presentation-to-viewer seam.

use super::test_volume;
use crate::app::action_adapter::{
    ViewerActionDisposition, ViewerActionError, ViewerViewport, ViewerViewportError,
};
use crate::app::SnapApp;
use crate::presentation::{
    PointerButton, PresentationDispatcher, PresentationEvent, ViewerAction, ViewportPoint,
};
use crate::tools::interaction::ToolState;
use crate::tools::kind::ToolKind;
use crate::ui::ViewTransform;

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
            x: 10,
            y: 10,
            button: PointerButton::Left,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 30, y: 0 },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 30,
            y: 0,
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
            x: 20,
            y: 20,
            button: PointerButton::Left,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerMove { x: 30, y: 15 },
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
            x: 1,
            y: 2,
            button: PointerButton::Left,
        },
    );
    apply_events(
        &mut app,
        &mut dispatcher,
        &viewport,
        PresentationEvent::PointerUp {
            x: 1,
            y: 2,
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
            x: 2,
            y: 2,
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
        position: ViewportPoint::new(1, 1),
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
            x: 1,
            y: 1,
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
                x: 1,
                y: 1,
                button: PointerButton::Right,
            }]),
        Err(
            crate::presentation::ActionDispatchError::PointerReleaseWithoutPress {
                button: PointerButton::Right
            }
        )
    ));
}
