use super::pointer_capture::{down, up};
use super::*;
use metis_platform::native::MouseButton::Left;
use metis_ui_lang::ScrollbarGeometry;

fn bar(session: &NativeViewerSession, pane: usize) -> ScrollbarGeometry {
    session.viewports[pane]
        .scrollbar
        .expect("stack scrollbar")
        .geometry
}

fn send(session: &mut NativeViewerSession, events: &[WindowEvent]) {
    session
        .handle_events(events)
        .expect("deliver native pointer input");
}

#[test]
fn slice_scrollbar_owns_the_pane_across_batches_and_clamps_both_ends() {
    let (mut session, _root) = session();
    let scrollbar = bar(&session, 0);
    let thumb = scrollbar.thumb();
    let track = scrollbar.track();
    assert_ne!(
        session.framebuffer.get_pixel(thumb.x, thumb.y),
        session.framebuffer.get_pixel(track.x, track.y)
    );
    assert_eq!(session.app.axis_slice_info(0), (1, 3));
    let pan = session.app.pan_offset;
    session.app.active_tool = crate::tools::kind::ToolKind::Pan;

    send(&mut session, &[down(thumb.x, thumb.y, Left)]);
    assert_eq!(session.app.axis_slice_info(0), (1, 3));
    send(
        &mut session,
        &[WindowEvent::PointerMove {
            x: 0,
            y: track.y + track.height + 40,
        }],
    );
    assert_eq!(session.app.axis_slice_info(0), (2, 3));
    let bottom = bar(&session, 0).thumb();
    assert_eq!(bottom.y + bottom.height, track.y + track.height);

    send(&mut session, &[up(0, track.y - 40, Left)]);
    assert_eq!(session.app.axis_slice_info(0), (0, 3));
    assert_eq!(session.app.pan_offset, pan);
}

#[test]
fn slice_scrollbar_track_selects_only_its_axis() {
    let (mut session, _root) = session();
    let axis = session.viewports[2].axis();
    let track = bar(&session, 2).track();
    let depth = session.app.axis_slice_info(0);
    send(
        &mut session,
        &[down(track.x, track.y, Left), up(track.x, track.y, Left)],
    );
    assert_eq!(session.app.axis, axis);
    assert_eq!(session.app.axis_slice_info(axis), (0, 4));
    assert_eq!(session.app.axis_slice_info(0), depth);
}

#[test]
fn slice_scrollbar_cancellation_ends_drag_without_selecting_a_slice() {
    for cancel in [WindowEvent::FocusLost, WindowEvent::PointerCaptureLost] {
        let (mut session, _root) = session();
        let scrollbar = bar(&session, 0);
        let thumb = scrollbar.thumb();
        let track = scrollbar.track();
        let y = track.y + track.height - 1;
        send(
            &mut session,
            &[
                down(thumb.x, thumb.y, Left),
                cancel,
                WindowEvent::PointerMove { x: thumb.x, y },
                up(thumb.x, y, Left),
            ],
        );
        assert_eq!(session.app.axis_slice_info(0), (1, 3));
    }
}

#[test]
fn slice_scrollbar_remains_draggable_in_four_row_grid() {
    let (mut session, _root) = session();
    let grid = super::layout::PanelGrid::new(5, 4).expect("four-row grid");
    session
        .set_workspace_layout(super::layout::WorkspaceLayout::Panels(grid))
        .expect("populate grid");
    session.refresh_frame().expect("compose grid");
    let scrollbar = bar(&session, 0);
    let thumb = scrollbar.thumb();
    let track = scrollbar.track();
    send(
        &mut session,
        &[
            down(thumb.x, thumb.y, Left),
            up(thumb.x, track.y + track.height, Left),
        ],
    );
    assert_eq!(session.app.axis_slice_info(0), (2, 3));
}
