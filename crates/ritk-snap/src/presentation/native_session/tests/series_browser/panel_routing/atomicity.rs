use super::*;

#[test]
fn rejected_comparison_batch_does_not_apply_an_earlier_panel_event() {
    let (mut viewer, _initial_root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.refresh_frame().expect("render both series");

    let primary_before = viewer.app.viewer_state.slice_index;
    let secondary_before = viewer.compare_panels[0].app.viewer_state.slice_index;
    let (left_x, left_y) = viewer.viewports[0].center();
    let (right_x, right_y) = viewer.viewports[1].center();
    let error = viewer
        .handle_events(&[
            WindowEvent::PointerWheel {
                x: left_x,
                y: left_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerUp {
                x: right_x,
                y: right_y,
                button: MouseButton::Left,
            },
        ])
        .expect_err("reject release without a press in the second panel");

    assert!(error.to_string().contains("without a press"));
    assert_eq!(viewer.app.viewer_state.slice_index, primary_before);
    assert_eq!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        secondary_before
    );
}

#[test]
fn rejected_native_batch_does_not_apply_pane_events_before_toolbar_actions() {
    let (mut viewer, _initial_root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.active_panel = 0;
    viewer.refresh_frame().expect("render both series");
    viewer.app.active_tool = crate::tools::kind::ToolKind::WindowLevel;

    let primary_slice = viewer.app.viewer_state.slice_index;
    let primary_tool = viewer.app.active_tool;
    let (left_x, left_y) = viewer.viewports[0].center();
    let (right_x, right_y) = viewer.viewports[1].center();
    let (pan_x, pan_y) = control_center(
        &viewer,
        None,
        crate::presentation::native_session::window_controls::WindowAction::SelectTool(
            crate::tools::kind::ToolKind::Pan,
        ),
    );
    let error = viewer
        .handle_events(&[
            WindowEvent::PointerWheel {
                x: left_x,
                y: left_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerDown {
                x: pan_x,
                y: pan_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: pan_x,
                y: pan_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: right_x,
                y: right_y,
                button: MouseButton::Left,
            },
        ])
        .expect_err("reject the unpressed release after all earlier inputs");

    assert!(error.to_string().contains("without a press"));
    assert_eq!(viewer.app.viewer_state.slice_index, primary_slice);
    assert_eq!(viewer.app.active_tool, primary_tool);
}
