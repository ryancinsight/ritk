//! A pane keeps the pointer until the last held button is released.

use super::super::layout::{PanelGrid, WorkspaceLayout};
use super::*;
use crate::presentation::PointerButton;
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
