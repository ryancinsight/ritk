use super::*;

#[test]
fn comparison_displays_two_independent_series_and_routes_each_panel() {
    let (mut viewer, _initial_root) = session();
    let replacement = replacement_study();
    viewer
        .open_study_path(replacement.path())
        .expect("open two-series study");
    viewer.refresh_frame().expect("render first series");
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
        ])
        .expect("enable side-by-side comparison");
    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.active_panel, 1);
    click_series(&mut viewer, 1);

    {
        let primary = viewer
            .app
            .loaded
            .as_ref()
            .expect("primary series remains loaded");
        let secondary = viewer
            .compare_panels
            .first()
            .expect("comparison panel exists")
            .app
            .loaded
            .as_ref()
            .expect("second series is loaded");
        let primary_uid = primary
            .metadata
            .as_ref()
            .expect("primary metadata")
            .series_instance_uid
            .as_deref();
        let secondary_uid = secondary
            .metadata
            .as_ref()
            .expect("secondary metadata")
            .series_instance_uid
            .as_deref();
        assert_eq!(primary_uid, Some(fixtures::SERIES_UID));
        assert_eq!(secondary_uid, Some(SECOND_SERIES_UID));
        assert_ne!(primary_uid, secondary_uid);
    }
    assert!(viewer.viewports[0].contains(300.0, 300.0));
    assert!(viewer.viewports[1].contains(900.0, 300.0));
    for index in 0..2 {
        let (x, y) = viewer.viewports[index].center();
        assert_ne!(
            viewer.framebuffer.get_pixel(x, y),
            metis_platform::Color::BLACK,
            "comparison panel {index} must display its loaded series"
        );
    }

    let primary_slice = viewer.app.viewer_state.slice_index;
    let secondary_slice = viewer
        .compare_panels
        .first()
        .expect("comparison panel exists")
        .app
        .viewer_state
        .slice_index;
    let (right_x, right_y) = viewer.viewports[1].center();
    viewer
        .handle_events(&[WindowEvent::PointerWheel {
            x: right_x,
            y: right_y,
            delta_x: 0,
            delta_y: -120,
            modifiers: ModifierState::NONE,
        }])
        .expect("advance only the comparison panel");
    assert_eq!(viewer.app.viewer_state.slice_index, primary_slice);
    assert_ne!(
        viewer
            .compare_panels
            .first()
            .expect("comparison panel")
            .app
            .viewer_state
            .slice_index,
        secondary_slice
    );
}

#[test]
fn one_input_batch_routes_wheels_to_each_comparison_panel() {
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
    viewer
        .handle_events(&[
            WindowEvent::PointerWheel {
                x: left_x,
                y: left_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerWheel {
                x: right_x,
                y: right_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("route both panel wheels in one host batch");

    assert_ne!(viewer.app.viewer_state.slice_index, primary_before);
    assert_ne!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        secondary_before
    );
    assert_eq!(viewer.active_panel, 1);
}

#[test]
fn wheel_after_maximizing_a_panel_keeps_the_presented_series_identity() {
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

    let first_slice = viewer.app.viewer_state.slice_index;
    let second_slice = viewer.compare_panels[0].app.viewer_state.slice_index;
    let second_view = viewer.viewports[1];
    let (wheel_x, wheel_y) = second_view.center();
    let panel = second_view.panel_rect().expect("second panel bounds");
    let maximize_x = panel.x + panel.width - 23;
    let maximize_y = panel.y - 13;

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
                x: wheel_x,
                y: wheel_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("maximize the second series and route the following wheel");

    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_ne!(viewer.app.viewer_state.slice_index, second_slice);
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(fixtures::SERIES_UID)
    );
    assert_eq!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        first_slice
    );
}

#[test]
fn wheel_after_closing_the_primary_routes_to_the_promoted_series() {
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

    let second_slice = viewer.compare_panels[0].app.viewer_state.slice_index;
    let first_view = viewer.viewports[0];
    let second_view = viewer.viewports[1];
    let (wheel_x, wheel_y) = second_view.center();
    let panel = first_view.panel_rect().expect("first panel bounds");
    let close_x = panel.x + panel.width - 7;
    let close_y = panel.y - 13;

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
                x: wheel_x,
                y: wheel_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("close the primary series and route the following wheel");

    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_ne!(viewer.app.viewer_state.slice_index, second_slice);
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(1)
    );
}
