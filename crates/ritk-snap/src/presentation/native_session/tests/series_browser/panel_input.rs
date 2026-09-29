use super::*;

#[test]
fn open_tools_popup_takes_priority_over_a_panel_close_control() {
    let (mut viewer, _initial_root) = session();
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(5, 1).expect("five-panel layout is supported"),
        ))
        .expect("select five-panel layout");
    viewer.app.active_tool = crate::tools::kind::ToolKind::Pan;
    viewer
        .refresh_frame()
        .expect("render the five-panel workspace");
    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 130,
                y: 12,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 130,
                y: 12,
                button: MouseButton::Left,
            },
        ])
        .expect("open the Tools menu");
    assert_eq!(
        viewer.window_chrome.open_menu(),
        Some(super::super::super::window_controls::Menu::Tools),
        "the tab click opens Tools before the overlapping panel action"
    );

    let first_panel = viewer.viewports[0];
    let overlapping_x = i32::try_from(
        first_panel
            .panel_x
            .saturating_add(first_panel.panel_width)
            .saturating_sub(7),
    )
    .expect("panel close center x fits i32");
    let overlapping_y = i32::try_from(first_panel.panel_y.saturating_sub(13))
        .expect("panel close center y fits i32");
    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: overlapping_x,
                y: overlapping_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: overlapping_x,
                y: overlapping_y,
                button: MouseButton::Left,
            },
        ])
        .expect("select the visible menu entry over the panel close control");

    assert_eq!(
        viewer.active_app().active_tool,
        crate::tools::kind::ToolKind::WindowLevel
    );
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(5),
        "the underlying panel control does not receive the menu click"
    );
}
