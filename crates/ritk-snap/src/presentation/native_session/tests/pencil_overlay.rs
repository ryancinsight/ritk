use super::super::layout::{append_measurement_preview, OverlayCommands};
use super::measurement_overlay::{axial_panel, composed, draw_lines};
use super::support::session;
use super::*;
use crate::tools::interaction::{Annotation, AnnotationLocation, ToolState};
use crate::tools::kind::ToolKind;
use metis_platform::native::MouseButton;

#[test]
#[cfg(windows)]
fn native_pencil_records_bounded_mark_and_respects_source_slice() {
    let (mut session, _root) = session();
    let (pane, x, y, _, _) = axial_panel(&session);
    let slice = session.views[pane].slice_index;
    let slice_count = session.views[pane].slice_count;
    let button = MouseButton::Left;
    let down = |x, y| WindowEvent::PointerDown { x, y, button };
    let up = |x, y| WindowEvent::PointerUp { x, y, button };
    session
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x50,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            down(x, y),
            WindowEvent::PointerMove { x: x + 8, y: y + 2 },
            WindowEvent::PointerMove { x: x + 8, y: y + 2 },
        ])
        .expect("start pencil capture");
    assert_eq!(session.app.tool_state.tool_kind(), Some(ToolKind::Pencil));
    let mut preview = OverlayCommands::with_capacity(4).expect("reserve pencil preview");
    append_measurement_preview(
        &mut preview,
        &session.views[pane],
        &session.viewports[pane],
        &session.app.tool_state,
        session.app.pointer.drag_measurement_location,
    )
    .expect("draw pencil preview");
    assert_eq!(draw_lines(preview.into_display_list().commands()).len(), 1);
    session
        .handle_events(&[up(x + 8, y + 2)])
        .expect("release pencil");
    let [Annotation::Pencil { location, points }] = session.app.annotations.as_slice() else {
        panic!("pencil release stores one mark");
    };
    assert_eq!(*location, Some(AnnotationLocation { plane: 0, slice }));
    assert_eq!(points.len(), 2);
    assert_eq!(draw_lines(&composed(&session, None)).len(), 1);
    session.app.set_slice_for_axis(0, (slice + 1) % slice_count);
    session.refresh_frame().expect("compose adjacent image");
    assert!(composed(&session, None)
        .iter()
        .all(|command| !matches!(command, metis_ui_lang::DisplayCommand::DrawLine { .. })));
}

#[test]
#[cfg(windows)]
fn native_pencil_escape_discards_capture() {
    let (mut session, _root) = session();
    let (_, x, y, _, _) = axial_panel(&session);
    session
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x50,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove { x: x + 4, y: y + 4 },
            WindowEvent::KeyDown {
                virtual_key: 0x1b,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerUp {
                x: x + 4,
                y: y + 4,
                button: MouseButton::Left,
            },
        ])
        .expect("cancel pencil capture");
    assert_eq!(session.app.tool_state, ToolState::Idle);
    assert!(session.app.annotations.is_empty());
}
