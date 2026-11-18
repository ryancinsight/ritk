use super::measurement_overlay::axial_panel;
use super::support::session;
use super::*;
use metis_platform::native::{ModifierState, MouseButton};

#[test]
fn native_polygon_edge_insert_escape_preserves_the_annotation() {
    let (mut session, _root) = session();
    let (pane, ..) = axial_panel(&session);
    let (center_x, center_y) = session.viewports[pane].center();
    let first = (center_x - 12, center_y);
    let second = (center_x + 12, center_y);
    let third = (center_x, center_y + 12);
    let button = MouseButton::Left;
    let down = |(x, y)| WindowEvent::PointerDown { x, y, button };
    let up = |(x, y)| WindowEvent::PointerUp { x, y, button };
    session
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x4f,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            down(first),
            up(first),
            down(second),
            up(second),
            down(third),
            WindowEvent::PointerDoubleClick {
                x: third.0,
                y: third.1,
                button,
            },
            up(third),
            WindowEvent::KeyDown {
                virtual_key: 0x4d,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("native O and double-click create the polygon");
    let before = session.app.annotations.clone();
    let midpoint = (center_x, center_y);
    let moved = (center_x + 4, center_y + 4);
    session
        .handle_events(&[
            down(midpoint),
            up(midpoint),
            down(midpoint),
            WindowEvent::PointerMove {
                x: moved.0,
                y: moved.1,
            },
        ])
        .expect("native Pan selects and previews the polygon edge");
    assert_eq!(session.app.pointer.selected_polygon, Some(0));
    let pending = session
        .app
        .pointer
        .pending_polygon_move
        .as_ref()
        .expect("edge drag owns a pending insertion");
    assert_ne!(pending.candidate, before[0]);
    session
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x1b,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            up(moved),
        ])
        .expect("native Escape cancels polygon edge insertion");
    assert_eq!(session.app.annotations, before);
    assert!(session.app.pointer.pending_polygon_move.is_none());
}
