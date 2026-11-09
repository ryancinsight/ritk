use super::*;

#[test]
fn pane_input_after_layout_selection_uses_the_presented_layout_snapshot() {
    let (mut viewer, _initial_root) = session();
    viewer
        .refresh_frame()
        .expect("render the orthogonal workspace");
    let (sagittal_x, sagittal_y) = viewer.viewports[2].center();
    let (picker_x, picker_y) = control_center(
        &viewer,
        None,
        crate::presentation::native_session::window_controls::WindowAction::OpenMenu(
            crate::presentation::native_session::window_controls::Menu::GridPicker,
        ),
    );
    let (grid_x, grid_y) = control_center(
        &viewer,
        Some(crate::presentation::native_session::window_controls::Menu::GridPicker),
        crate::presentation::native_session::window_controls::WindowAction::SetLayout(
            WorkspaceLayout::Panels(PanelGrid::new(2, 1).expect("side-by-side grid")),
        ),
    );
    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: picker_x,
                y: picker_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: picker_x,
                y: picker_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: grid_x,
                y: grid_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: grid_x,
                y: grid_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: sagittal_x,
                y: sagittal_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: sagittal_x,
                y: sagittal_y,
                button: MouseButton::Left,
            },
        ])
        .expect("reduce pane events against the layout visible at batch start");

    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.viewports.len(), 2);
    assert_eq!(viewer.app.axis, 2);
}

/// Select a two-panel grid and press `button` over the old sagittal pane in
/// the same host batch, returning the press and picker positions.
fn press_after_selecting_two_panels(
    viewer: &mut crate::presentation::native_session::NativeViewerSession,
    button: MouseButton,
) -> ((i32, i32), (i32, i32)) {
    let (sagittal_x, sagittal_y) = viewer.viewports[2].center();
    viewer.app.active_tool = crate::tools::kind::ToolKind::Pan;
    let (picker_x, picker_y) = control_center(
        viewer,
        None,
        crate::presentation::native_session::window_controls::WindowAction::OpenMenu(
            crate::presentation::native_session::window_controls::Menu::GridPicker,
        ),
    );
    let (grid_x, grid_y) = control_center(
        viewer,
        Some(crate::presentation::native_session::window_controls::Menu::GridPicker),
        crate::presentation::native_session::window_controls::WindowAction::SetLayout(
            WorkspaceLayout::Panels(PanelGrid::new(2, 1).expect("side-by-side grid")),
        ),
    );

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: picker_x,
                y: picker_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: picker_x,
                y: picker_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: grid_x,
                y: grid_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: grid_x,
                y: grid_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: sagittal_x,
                y: sagittal_y,
                button,
            },
        ])
        .expect("apply layout selection against the presented orthogonal frame");
    ((sagittal_x, sagittal_y), (picker_x, picker_y))
}

#[test]
fn layout_change_cancels_a_captured_drag_before_remapping_panes() {
    let (mut viewer, _initial_root) = session();
    viewer
        .refresh_frame()
        .expect("render the orthogonal workspace");
    viewer.app.active_tool = crate::tools::kind::ToolKind::Pan;
    let ((sagittal_x, sagittal_y), (picker_x, picker_y)) =
        press_after_selecting_two_panels(&mut viewer, MouseButton::Left);

    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.viewports.len(), 2);
    assert_eq!(viewer.active_view, None);
    assert!(viewer.captured_buttons.is_empty());
    assert!(viewer.app.tool_state.is_idle());

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: picker_x,
                y: picker_y,
                button: MouseButton::Right,
            },
            WindowEvent::PointerUp {
                x: picker_x,
                y: picker_y,
                button: MouseButton::Right,
            },
        ])
        .expect("a different button does not release the canceled left gesture");

    viewer
        .handle_events(&[WindowEvent::PointerMove {
            x: sagittal_x + 16,
            y: sagittal_y + 12,
        }])
        .expect("ignore the canceled drag while the new layout is active");
    viewer
        .handle_events(&[WindowEvent::PointerUp {
            x: sagittal_x + 16,
            y: sagittal_y + 12,
            button: MouseButton::Left,
        }])
        .expect("the release of a canceled drag has no gesture to end");

    assert!(viewer.app.tool_state.is_idle());
    assert_eq!(viewer.active_view, None);
}

#[test]
fn layout_change_drops_the_release_of_a_non_primary_button() {
    let (mut viewer, _initial_root) = session();
    viewer
        .refresh_frame()
        .expect("render the orthogonal workspace");

    let ((x, y), _) = press_after_selecting_two_panels(&mut viewer, MouseButton::Right);

    assert!(viewer.captured_buttons.is_empty());
    let active_panel = viewer.active_panel;
    viewer
        .handle_events(&[WindowEvent::PointerUp {
            x,
            y,
            button: MouseButton::Right,
        }])
        .expect("the release of a cancelled right drag has no gesture to end");
    assert_eq!(viewer.active_view, None);
    assert_eq!(viewer.active_panel, active_panel);
}
