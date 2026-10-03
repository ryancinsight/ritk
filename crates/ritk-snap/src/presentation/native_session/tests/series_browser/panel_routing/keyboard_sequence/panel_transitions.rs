use super::*;

#[test]
fn queued_primary_close_advances_the_surviving_series() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(3, 1).expect("three-panel grid is valid"),
        ))
        .expect("select three-panel layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer
        .assign_series_to_panel(2, 2)
        .expect("load the third series into panel three");
    viewer.refresh_frame().expect("render all three series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("focus the primary panel");
    assert_eq!(viewer.active_panel, 0);

    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x73,
                repeated: false,
                modifiers: ModifierState::CONTROL,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("advance the promoted primary series after closing its predecessor");

    assert_eq!(viewer.primary_series_index, Some(2));
    assert_eq!(viewer.compare_panels[0].series_index, Some(2));
    assert_eq!(loaded_series_uid(&viewer.app), Some(THIRD_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(THIRD_SERIES_UID)
    );
    assert_eq!(viewer.active_panel, 0);
}

#[test]
fn queued_restore_then_tab_browses_the_newly_focused_panel() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.refresh_frame().expect("render both series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("focus the primary panel");
    viewer
        .toggle_panel_maximize(0)
        .expect("maximize the primary panel");
    viewer.refresh_frame().expect("render the maximized panel");

    let viewport = viewer.viewports[0];
    let restore_x = i32::try_from(
        viewport
            .panel_x
            .saturating_add(viewport.panel_width)
            .saturating_sub(23),
    )
    .expect("restore control x fits i32");
    let restore_y =
        i32::try_from(viewport.panel_y.saturating_sub(13)).expect("restore control y fits i32");
    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: restore_x,
                y: restore_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: restore_x,
                y: restore_y,
                button: MouseButton::Left,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x09,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("restore, focus, then browse in event order");

    assert_eq!(viewer.maximized_panel.map(|panel| panel.panel_index), None);
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
    assert_eq!(viewer.active_panel, 1);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(2));
    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(THIRD_SERIES_UID)
    );
}

#[test]
fn offscreen_pointer_move_uses_the_batch_start_panel_route() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(2, 1)
        .expect("load the third series into panel two");
    viewer.refresh_frame().expect("render both series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("focus the primary panel");
    assert_eq!(viewer.active_panel, 0);

    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x09,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerMove { x: -1, y: -1 },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("browse the batch-start panel after moving outside the viewport");

    assert_eq!(viewer.active_panel, 0);
    assert_eq!(viewer.primary_series_index, Some(1));
    assert_eq!(viewer.compare_panels[0].series_index, Some(2));
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(THIRD_SERIES_UID)
    );
}

#[test]
fn pointer_capture_on_closed_panel_does_not_retarget_series_browsing() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(3, 1).expect("three-panel grid is valid"),
        ))
        .expect("select three-panel layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer
        .assign_series_to_panel(2, 2)
        .expect("load the third series into panel three");
    viewer.refresh_frame().expect("render all three series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("focus the primary panel");
    assert_eq!(viewer.active_panel, 0);
    let (x, y) = viewer.viewports[0].center();
    let (closed_panel_x, closed_panel_y) = viewer.viewports[2].center();

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x73,
                repeated: false,
                modifiers: ModifierState::CONTROL,
            },
            WindowEvent::PointerMove {
                x: closed_panel_x,
                y: closed_panel_y,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("keep pointer capture and browse after closing its panel");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
    assert_eq!(viewer.active_panel, 0);
    assert_eq!(viewer.primary_series_index, Some(2));
    assert_eq!(viewer.compare_panels[0].series_index, Some(2));
    assert_eq!(loaded_series_uid(&viewer.app), Some(THIRD_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(THIRD_SERIES_UID)
    );
}

#[test]
fn pointer_move_to_a_hidden_panel_does_not_retarget_keyboard_focus() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(2, 1)
        .expect("load the third series into panel two");
    viewer.refresh_frame().expect("render both series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("focus the primary panel");
    let maximize_panel = viewer.viewports[0];
    let hidden_panel = viewer.viewports[1];
    let maximize_x = i32::try_from(
        maximize_panel
            .panel_x
            .saturating_add(maximize_panel.panel_width)
            .saturating_sub(23),
    )
    .expect("maximize control x fits i32");
    let maximize_y = i32::try_from(maximize_panel.panel_y.saturating_sub(13))
        .expect("maximize control y fits i32");
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
            WindowEvent::PointerMove {
                x: hidden_x,
                y: hidden_y,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("browse the keyboard-focused panel after the other panel is hidden");

    assert_eq!(
        viewer.maximized_panel.map(|panel| panel.panel_index),
        Some(0)
    );
    assert_eq!(viewer.active_panel, 0);
    assert_eq!(viewer.primary_series_index, Some(1));
    assert_eq!(viewer.compare_panels[0].series_index, Some(2));
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(THIRD_SERIES_UID)
    );
}
