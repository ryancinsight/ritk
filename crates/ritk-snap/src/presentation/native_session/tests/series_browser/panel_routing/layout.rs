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

#[test]
fn layout_change_cancels_a_captured_drag_before_remapping_panes() {
    let (mut viewer, _initial_root) = session();
    viewer
        .refresh_frame()
        .expect("render the orthogonal workspace");
    let (sagittal_x, sagittal_y) = viewer.viewports[2].center();
    viewer.app.active_tool = crate::tools::kind::ToolKind::Pan;
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
        ])
        .expect("apply layout selection against the presented orthogonal frame");

    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.viewports.len(), 2);
    assert_eq!(viewer.active_view, None);
    assert_eq!(
        viewer.suppress_cancelled_pointer_release,
        Some(crate::presentation::PointerButton::Left)
    );
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
    assert_eq!(
        viewer.suppress_cancelled_pointer_release,
        Some(crate::presentation::PointerButton::Left)
    );

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
        .expect("consume the release for the canceled drag");

    assert_eq!(viewer.suppress_cancelled_pointer_release, None);
    assert!(viewer.app.tool_state.is_idle());
}
