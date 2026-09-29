use super::*;
use crate::presentation::ViewportPoint;
use crate::ui::pan::pan_from_drag_delta;

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

#[test]
fn closing_a_middle_panel_routes_later_input_to_its_new_owner() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(3, 1).expect("three-panel layout"),
        ))
        .expect("select three-panel layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load second series");
    viewer
        .assign_series_to_panel(2, 2)
        .expect("load third series");
    viewer.refresh_frame().expect("render three series");

    let closed = viewer.viewports[1];
    let close_x = i32::try_from(
        closed
            .panel_x
            .saturating_add(closed.panel_width)
            .saturating_sub(7),
    )
    .expect("panel close x fits i32");
    let close_y = i32::try_from(closed.panel_y.saturating_sub(13)).expect("panel close y fits i32");
    let moved_series = viewer.viewports[2];
    let (slice_x, slice_y) = moved_series.center();
    let moved_slice_before = viewer.compare_panels[1].app.viewer_state.slice_index;

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: close_x,
                y: close_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: close_x,
                y: close_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerWheel {
                x: slice_x,
                y: slice_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("route original panel-three input after closing panel two");

    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(THIRD_SERIES_UID)
    );
    assert_eq!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        moved_slice_before
            .checked_add(1)
            .expect("test slice index fits one navigation step")
    );
    assert_eq!(loaded_series_uid(&viewer.compare_panels[1].app), None);
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
}

#[test]
fn consecutive_maximize_and_close_actions_keep_the_original_panel_owners() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(3, 1).expect("three-panel layout"),
        ))
        .expect("select three-panel layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load second series");
    viewer
        .assign_series_to_panel(2, 2)
        .expect("load third series");
    viewer.refresh_frame().expect("render three series");

    let maximized = viewer.viewports[1];
    let maximize_x = i32::try_from(
        maximized
            .panel_x
            .saturating_add(maximized.panel_width)
            .saturating_sub(23),
    )
    .expect("panel maximize x fits i32");
    let first = viewer.viewports[0];
    let first_maximize_x = i32::try_from(
        first
            .panel_x
            .saturating_add(first.panel_width)
            .saturating_sub(23),
    )
    .expect("first panel maximize x fits i32");
    let third = viewer.viewports[2];
    let third_close_x = i32::try_from(
        third
            .panel_x
            .saturating_add(third.panel_width)
            .saturating_sub(7),
    )
    .expect("third panel close x fits i32");
    let maximize_y =
        i32::try_from(maximized.panel_y.saturating_sub(13)).expect("panel maximize y fits i32");
    let first_maximize_y =
        i32::try_from(first.panel_y.saturating_sub(13)).expect("first panel maximize y fits i32");
    let third_close_y =
        i32::try_from(third.panel_y.saturating_sub(13)).expect("third panel close y fits i32");

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: first_maximize_x,
                y: first_maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: first_maximize_x,
                y: first_maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: third_close_x,
                y: third_close_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: third_close_x,
                y: third_close_y,
                button: MouseButton::Left,
            },
        ])
        .expect("route panel chrome actions from the original frame");

    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(SECOND_SERIES_UID)
    );
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
}

#[test]
fn queued_input_keeps_its_original_panel_owner_after_maximize() {
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

    let hidden_panel = viewer.viewports[0];
    let hidden_slice = viewer.app.viewer_state.slice_index;
    let visible_slice = viewer.compare_panels[0].app.viewer_state.slice_index;
    let maximize_x = i32::try_from(
        viewer.viewports[1]
            .panel_x
            .saturating_add(viewer.viewports[1].panel_width)
            .saturating_sub(23),
    )
    .expect("panel maximize x fits i32");
    let maximize_y = i32::try_from(viewer.viewports[1].panel_y.saturating_sub(13))
        .expect("panel maximize y fits i32");
    let (hidden_x, hidden_y) = hidden_panel.center();

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerWheel {
                x: hidden_x,
                y: hidden_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("ignore input to the panel hidden by the maximize action");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(1)
    );
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(viewer.app.viewer_state.slice_index, visible_slice);
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(fixtures::SERIES_UID)
    );
    assert_eq!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        hidden_slice.saturating_add(1),
        "the event is routed against the image layout visible at batch start"
    );
}

#[test]
fn queued_drag_keeps_its_original_viewport_mapping_after_maximize() {
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
    viewer.app.active_tool = crate::tools::kind::ToolKind::Pan;

    let original_viewport = viewer.viewports[0];
    let primary_offset = viewer.app.pan_offset;
    let maximized_offset = viewer.compare_panels[0].app.pan_offset;
    let original_mapping = original_viewport.mapping();
    let (start_x, start_y) = original_viewport.center();
    let end_x = start_x.saturating_add(24);
    let end_y = start_y.saturating_add(12);
    let expected = pan_from_drag_delta(
        primary_offset,
        original_mapping
            .map(ViewportPoint::new(f64::from(start_x), f64::from(start_y)))
            .expect("drag starts over the original image"),
        original_mapping
            .map(ViewportPoint::new(f64::from(end_x), f64::from(end_y)))
            .expect("drag ends over the original image"),
    );
    let maximized = viewer.viewports[1];
    let maximize_x = i32::try_from(
        maximized
            .panel_x
            .saturating_add(maximized.panel_width)
            .saturating_sub(23),
    )
    .expect("panel maximize x fits i32");
    let maximize_y =
        i32::try_from(maximized.panel_y.saturating_sub(13)).expect("panel maximize y fits i32");

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove { x: end_x, y: end_y },
            WindowEvent::PointerUp {
                x: end_x,
                y: end_y,
                button: MouseButton::Left,
            },
        ])
        .expect("map queued drag coordinates against the visible frame");

    assert_eq!(viewer.compare_panels[0].app.pan_offset, expected);
    assert_eq!(
        viewer.app.pan_offset, maximized_offset,
        "the maximized panel keeps its independent view state"
    );
    assert_ne!(expected, primary_offset);
}
