//! Gesture lifecycle: cancellation, focus loss, and typed refusals.

use crate::app::action_adapter::ViewerActionDisposition;
use crate::app::viewer_viewport::{ViewerViewport, ViewerViewportError};
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationDispatcher, PresentationEvent, ViewerAction};
use crate::tools::interaction::ToolState;
use crate::tools::kind::ToolKind;
use crate::ui::ViewTransform;

use super::{apply_app_event, apply_events, viewport};

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
    let invalid_axis =
        ViewerViewport::new(3, [0.0, 0.0], [1.0, 1.0], [4, 4], ViewTransform::default());
    assert!(matches!(
        invalid_axis,
        Err(ViewerViewportError::Axis { axis: 3 })
    ));

    let invalid_geometry =
        ViewerViewport::new(0, [0.0, 0.0], [0.0, 1.0], [4, 4], ViewTransform::default());
    assert!(matches!(
        invalid_geometry,
        Err(ViewerViewportError::InvalidScreenGeometry)
    ));
}
