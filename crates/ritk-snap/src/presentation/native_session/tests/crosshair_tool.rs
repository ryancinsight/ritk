//! Native Crosshair-tool routing and linked-voxel display tests.

use super::super::layout::{crosshair_overlay, screen_coordinate, CROSSHAIR_COLOR};
use super::support::{control_center, session};
use super::*;
use crate::presentation::native_session::window_controls::{Menu, WindowAction};
use crate::tools::kind::ToolKind;
use metis_platform::native::MouseButton;

fn click(x: i32, y: i32) -> [WindowEvent; 2] {
    [
        WindowEvent::PointerDown {
            x,
            y,
            button: MouseButton::Left,
        },
        WindowEvent::PointerUp {
            x,
            y,
            button: MouseButton::Left,
        },
    ]
}

#[test]
fn crosshair_tool_click_shows_the_selected_voxel_in_all_three_planes() {
    let (mut session, _root) = session();
    assert!(!session.app.show_crosshair);
    let pan = control_center(&session, None, WindowAction::SelectTool(ToolKind::Pan));
    session
        .handle_events(&click(pan.0, pan.1))
        .expect("select Pan");
    assert_eq!(session.app.active_tool, ToolKind::Pan);
    let (initial_pan_x, initial_pan_y) = session.viewports[0].center();
    session
        .handle_events(&click(initial_pan_x, initial_pan_y))
        .expect("initial Pan click");
    assert!(!session.app.show_crosshair);

    let tools = control_center(&session, None, WindowAction::OpenMenu(Menu::Tools));
    session
        .handle_events(&click(tools.0, tools.1))
        .expect("open the Tools menu");
    let crosshair = control_center(
        &session,
        Some(Menu::Tools),
        WindowAction::SelectTool(ToolKind::Crosshair),
    );
    session
        .handle_events(&click(crosshair.0, crosshair.1))
        .expect("select Crosshair from Tools");
    assert_eq!(session.app.active_tool, ToolKind::Crosshair);

    let axial = session
        .viewports
        .iter()
        .position(|viewport| viewport.axis() == 0)
        .expect("orthogonal layout has an axial pane");
    let [screen_x, screen_y] = session.viewports[axial].mapping().project([0.5, 0.5]);
    let point = [
        screen_coordinate(screen_x, "crosshair test x").expect("x fits native coordinates"),
        screen_coordinate(screen_y, "crosshair test y").expect("y fits native coordinates"),
    ];
    let slice = session.views[axial].slice_index;
    session
        .handle_events(&click(point[0], point[1]))
        .expect("place the linked cursor");

    assert_eq!(
        session
            .app
            .linked_cursor
            .expect("loaded linked cursor")
            .voxel(),
        [slice, 0, 0]
    );
    assert!(session.app.show_crosshair);
    let shape = session.app.loaded.as_ref().map(|volume| volume.shape);
    let cursor = session.app.linked_cursor.map(|cursor| cursor.voxel());
    let display = crosshair_overlay(
        &session.views,
        &session.viewports,
        shape,
        cursor,
        session.app.show_crosshair,
    )
    .expect("compose the visible linked cursor");
    assert_eq!(
        display
            .commands()
            .iter()
            .filter(|command| matches!(command, metis_ui_lang::DisplayCommand::DrawLine { color, .. } if *color == CROSSHAIR_COLOR))
            .count(),
        6
    );

    session
        .handle_events(&click(pan.0, pan.1))
        .expect("select Pan again");
    assert!(session.app.show_crosshair);
    let (pan_x, pan_y) = session.viewports[axial].center();
    session
        .handle_events(&click(pan_x, pan_y))
        .expect("Pan click");
    assert!(session.app.show_crosshair);
}
