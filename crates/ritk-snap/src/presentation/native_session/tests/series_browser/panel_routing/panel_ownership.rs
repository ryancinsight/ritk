use super::*;

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
