use super::measurement_overlay::axial_panel;
use super::support::session;
use super::*;
use crate::tools::interaction::{Annotation, AnnotationLocation};
use crate::tools::kind::ToolKind;
use metis_platform::native::{ModifierState, MouseButton};

#[test]
fn native_polygon_edge_insert_escape_preserves_the_annotation() {
    let (mut session, _root) = session();
    session.app.select_tool(ToolKind::Pan);
    let (pane, x, y, dx, dy) = axial_panel(&session);
    let location = AnnotationLocation {
        plane: session.views[pane].axis,
        slice: session.views[pane].slice_index,
    };
    session.app.annotations.push(Annotation::OpenPolygon {
        location: Some(location),
        points: vec![[0.0, 0.0], [0.0, 20.0], [20.0, 20.0]],
        length_mm: 40.0,
    });
    session.refresh_frame().expect("compose polygon mark");
    let edge = (x + 10.0 * dx, y);
    let moved = (x + 5.0 * dx, y);
    let button = MouseButton::Left;
    let down = |(x, y)| WindowEvent::PointerDown { x, y, button };
    let up = |(x, y)| WindowEvent::PointerUp { x, y, button };
    let before = session.app.annotations.clone();
    session
        .handle_events(&[
            down(edge),
            up(edge),
            down(edge),
            WindowEvent::PointerMove {
                x: moved.0,
                y: moved.1,
            },
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
