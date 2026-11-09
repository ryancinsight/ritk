//! A pane keeps the pointer until the last held button is released.

use super::super::layout::{PanelGrid, WorkspaceLayout};
use super::super::window_controls::WindowAction;
use super::*;
use crate::presentation::PointerButton;
use crate::tools::kind::ToolKind;
use metis_platform::native::MouseButton;

fn down(x: i32, y: i32, button: MouseButton) -> WindowEvent {
    WindowEvent::PointerDown { x, y, button }
}

fn up(x: i32, y: i32, button: MouseButton) -> WindowEvent {
    WindowEvent::PointerUp { x, y, button }
}

#[test]
fn releasing_a_second_button_keeps_the_pane_that_owns_the_gesture() {
    let (mut session, _root) = session();
    let (x, y) = session.viewports[0].center();
    let (other_x, other_y) = session.viewports[1].center();
    let owning_axis = session.viewports[0].axis();

    session
        .handle_events(&[
            down(x, y, MouseButton::X2),
            down(x, y, MouseButton::Right),
            up(x, y, MouseButton::Right),
        ])
        .expect("a second button is ordinary input");
    assert_eq!(session.active_view, Some(0));
    assert!(session.captured_buttons.contains(PointerButton::X2));
    assert!(!session.captured_buttons.contains(PointerButton::Right));

    session
        .handle_events(&[
            WindowEvent::PointerMove {
                x: other_x,
                y: other_y,
            },
            up(other_x, other_y, MouseButton::X2),
        ])
        .expect("the owning button ends the gesture in its own pane");

    assert_eq!(session.active_view, None);
    assert!(session.captured_buttons.is_empty());
    assert_eq!(session.app.axis, owning_axis);
    assert!(
        session.app.annotations.is_empty(),
        "a release outside the owning pane records no cross-plane length"
    );
}

#[test]
fn a_button_pressed_in_another_panel_reaches_the_panel_that_owns_the_gesture() {
    let (mut session, _root) = session();
    session
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid"),
        ))
        .expect("switch to two panels");
    let (x, y) = session.viewports[0].center();
    let (other_x, other_y) = session.viewports[1].center();

    session
        .handle_events(&[
            down(x, y, MouseButton::Middle),
            WindowEvent::PointerMove {
                x: other_x,
                y: other_y,
            },
            down(other_x, other_y, MouseButton::Right),
            up(other_x, other_y, MouseButton::Right),
        ])
        .expect("the second button routes with the captured panel");

    let owner = session.app.pointer.owner.map(|owner| owner.button);
    assert_eq!(owner, Some(PointerButton::Middle));
    assert_eq!(session.active_view, Some(0));

    session
        .handle_events(&[up(other_x, other_y, MouseButton::Middle)])
        .expect("the owning button ends the gesture");
    assert_eq!(session.app.pointer.owner, None);
    assert_eq!(session.active_view, None);
}

#[test]
fn a_release_after_focus_loss_is_ordinary_input() {
    let (mut session, _root) = session();
    let (x, y) = session.viewports[0].center();

    session
        .handle_events(&[down(x, y, MouseButton::Right), WindowEvent::FocusLost])
        .expect("focus loss cancels the gesture");
    session
        .handle_events(&[up(x, y, MouseButton::Right)])
        .expect("the release has no gesture left to end");

    assert_eq!(session.active_view, None);
    assert_eq!(session.app.pointer.owner, None);
    session
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("the button starts a new gesture");
    assert_eq!(session.active_view, Some(0));
}

#[test]
fn a_press_of_a_held_button_recaptures_at_the_pointer() {
    let (mut session, _root) = session();
    let (x, y) = session.viewports[0].center();
    let (other_x, other_y) = session.viewports[1].center();

    session
        .handle_events(&[
            down(x, y, MouseButton::Right),
            down(other_x, other_y, MouseButton::Right),
        ])
        .expect("a lost release is ordinary host input");

    assert_eq!(session.active_view, Some(1));
    assert!(session.captured_buttons.contains(PointerButton::Right));
    assert_eq!(
        session.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Right)
    );

    session
        .handle_events(&[up(other_x, other_y, MouseButton::Right)])
        .expect("the release ends the new capture");
    assert_eq!(session.active_view, None);
    assert!(session.captured_buttons.is_empty());
    assert_eq!(session.app.pointer.owner, None);
}

#[test]
fn selecting_a_tool_from_the_toolbar_ends_a_pending_measurement() {
    let (mut session, _root) = session();
    let (x, y) = session.viewports[0].center();
    session.app.active_tool = ToolKind::MeasureLength;
    session
        .handle_events(&[down(x, y, MouseButton::Left), up(x, y, MouseButton::Left)])
        .expect("place the first anchor");
    assert!(session.app.tool_state.is_pending_measurement());
    session
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("a display gesture suspends the measurement");
    assert_ne!(session.app.pointer.suspended_measurement, None);

    session
        .apply_window_action(WindowAction::SelectTool(ToolKind::Pan))
        .expect("select another tool");

    assert_eq!(session.app.active_tool, ToolKind::Pan);
    assert!(session.app.tool_state.is_idle());
    assert_eq!(session.app.pointer.suspended_measurement, None);
}

#[test]
fn a_stationary_release_leaves_a_display_gesture_where_its_last_move_put_it() {
    let (mut session, _root) = session();
    let (x, y) = session.viewports[0].center();

    session
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("begin zooming");
    session
        .handle_events(&[WindowEvent::PointerMove { x, y: y - 40 }])
        .expect("drag the zoom");
    let zoom = session.app.zoom;
    assert_ne!(zoom, 1.0, "the drag must change the zoom");
    session
        .handle_events(&[up(x, y - 40, MouseButton::Right)])
        .expect("release where the last move ended");
    assert_eq!(session.app.zoom, zoom);

    let (x, y) = session.viewports[0].center();
    session
        .handle_events(&[down(x, y, MouseButton::X1)])
        .expect("begin panning");
    session
        .handle_events(&[WindowEvent::PointerMove { x: x + 40, y }])
        .expect("drag the pan");
    let pan = session.app.pan_offset;
    assert_ne!(pan.x(), 0.0, "the drag must change the pan");
    session
        .handle_events(&[up(x + 40, y, MouseButton::X1)])
        .expect("release where the last move ended");
    assert_eq!(session.app.pan_offset, pan);
}

#[test]
fn reset_view_restores_the_fit_and_ends_a_pending_measurement() {
    use crate::tools::interaction::{ImagePoint, ToolState, ViewportOffset};

    let (mut session, _root) = session();
    session.app.zoom = 3.25;
    session.app.pan_offset = ViewportOffset::new(24.0, -8.0);
    session.app.tool_state = ToolState::MeasureLength1 {
        p1: ImagePoint::new(1.0, 2.0),
    };

    session
        .apply_window_action(WindowAction::ResetView)
        .expect("reset the view");

    assert_eq!(session.app.zoom, 1.0);
    assert_eq!(session.app.pan_offset, ViewportOffset::new(0.0, 0.0));
    assert!(session.app.tool_state.is_idle());
}
