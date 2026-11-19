//! A pane keeps the pointer until the last held button is released.

use super::super::layout::{PanelGrid, WorkspaceLayout};
use super::super::window_controls::WindowAction;
use super::*;
use crate::presentation::{PointerButton, ViewportPoint};
use crate::tools::interaction::ViewportOffset;
use crate::tools::kind::ToolKind;
use crate::ui::{pan_from_drag_delta, zoom_from_drag_delta};
use eunomia::FloatElement;
use metis_platform::native::MouseButton;

pub(super) fn down(x: i32, y: i32, button: MouseButton) -> WindowEvent {
    WindowEvent::PointerDown { x, y, button }
}

pub(super) fn up(x: i32, y: i32, button: MouseButton) -> WindowEvent {
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
    assert_eq!(session.app.pointer.navigation_drag_viewport, None);
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

fn navigation_session() -> (NativeViewerSession, tempfile::TempDir) {
    let (mut session, root) = session();
    let texel_width =
        session.viewports[0].image.width / f64::from(session.views[0].frame().width());
    let texel_height =
        session.viewports[0].image.height / f64::from(session.views[0].frame().height());
    session.app.zoom = f32::from_f64(4.0 / texel_width.min(texel_height));
    session
        .refresh_frame()
        .expect("render a viewport with at least four screen pixels per image pixel");
    (session, root)
}

fn apply_drag_moves(
    session: &mut NativeViewerSession,
    moves: &[WindowEvent; 3],
    separate_moves: bool,
) {
    if separate_moves {
        for event in moves {
            session
                .handle_events(std::slice::from_ref(event))
                .expect("apply one drag move batch");
        }
    } else {
        session
            .handle_events(moves)
            .expect("apply one combined drag move batch");
    }
}

fn drag_pan_in_batches(
    separate_moves: bool,
    step: i32,
) -> crate::tools::interaction::ViewportOffset {
    let (mut session, _root) = navigation_session();
    let (x, y) = session.viewports[0].center();
    session
        .handle_events(&[down(x, y, MouseButton::X1)])
        .expect("begin panning");
    let moves =
        [step, step * 2, step * 3].map(|delta_x| WindowEvent::PointerMove { x: x + delta_x, y });
    apply_drag_moves(&mut session, &moves, separate_moves);
    session
        .handle_events(&[up(x + step * 3, y, MouseButton::X1)])
        .expect("finish panning");
    assert_eq!(session.app.pointer.owner, None);
    assert_eq!(session.app.pointer.navigation_drag_viewport, None);
    assert_eq!(session.active_view, None);
    session.app.pan_offset
}

fn drag_zoom_in_batches(separate_moves: bool, step: i32) -> f32 {
    let (mut session, _root) = navigation_session();
    let (x, y) = session.viewports[0].center();
    session
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("begin zooming");
    let moves =
        [step, step * 2, step * 3].map(|delta_y| WindowEvent::PointerMove { x, y: y - delta_y });
    apply_drag_moves(&mut session, &moves, separate_moves);
    session
        .handle_events(&[up(x, y - step * 3, MouseButton::Right)])
        .expect("finish zooming");
    assert_eq!(session.app.pointer.owner, None);
    assert_eq!(session.app.pointer.navigation_drag_viewport, None);
    assert_eq!(session.active_view, None);
    session.app.zoom
}

#[test]
fn display_drags_use_full_displacement_independent_of_event_batches() {
    let (session, _root) = navigation_session();
    let mapping = session.viewports[0].mapping();
    let (x, y) = session.viewports[0].center();
    let start = mapping
        .map(ViewportPoint::new(f64::from(x), f64::from(y)))
        .expect("viewport center maps to the source image");
    let pan_end = mapping
        .map(ViewportPoint::new(f64::from(x + 3), f64::from(y)))
        .expect("pan endpoint maps to the source image");
    let zoom_end = mapping
        .map(ViewportPoint::new(f64::from(x), f64::from(y - 3)))
        .expect("zoom endpoint maps to the source image");
    let expected_pan = pan_from_drag_delta(
        session.app.pan_offset,
        ViewportOffset::new(pan_end.x() - start.x(), pan_end.y() - start.y()),
    )
    .expect("finite mapped pan displacement");
    let expected_zoom = zoom_from_drag_delta(session.app.zoom, zoom_end.y() - start.y());

    let combined_pan = drag_pan_in_batches(false, 1);
    let separated_pan = drag_pan_in_batches(true, 1);
    let combined_zoom = drag_zoom_in_batches(false, 1);
    let separated_zoom = drag_zoom_in_batches(true, 1);

    assert_eq!(combined_pan, expected_pan);
    assert_eq!(separated_pan, expected_pan);
    assert_eq!(combined_pan, separated_pan);
    assert_eq!(combined_zoom, expected_zoom);
    assert_eq!(separated_zoom, expected_zoom);
    assert_eq!(combined_zoom, separated_zoom);
}

#[test]
fn captured_display_drags_continue_beyond_the_press_image_bounds() {
    let (session, _root) = navigation_session();
    let mapping = session.viewports[0].mapping();
    let (x, y) = session.viewports[0].center();
    let start = mapping
        .map(ViewportPoint::new(f64::from(x), f64::from(y)))
        .expect("viewport center maps to the source image");
    let pan_end = mapping
        .map_unbounded(ViewportPoint::new(f64::from(x + 24), f64::from(y)))
        .expect("captured pan maps beyond the press image bounds");
    let zoom_end = mapping
        .map_unbounded(ViewportPoint::new(f64::from(x), f64::from(y - 24)))
        .expect("captured zoom maps beyond the press image bounds");
    let expected_pan = pan_from_drag_delta(
        session.app.pan_offset,
        ViewportOffset::new(pan_end.x() - start.x(), pan_end.y() - start.y()),
    )
    .expect("finite mapped pan displacement");
    let expected_zoom = zoom_from_drag_delta(session.app.zoom, zoom_end.y() - start.y());

    assert_eq!(drag_pan_in_batches(false, 8), expected_pan);
    assert_eq!(drag_pan_in_batches(true, 8), expected_pan);
    assert_eq!(drag_zoom_in_batches(false, 8), expected_zoom);
    assert_eq!(drag_zoom_in_batches(true, 8), expected_zoom);
}
