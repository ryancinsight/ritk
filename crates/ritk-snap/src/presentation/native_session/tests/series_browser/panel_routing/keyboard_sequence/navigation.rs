use super::*;

#[test]
fn queued_keyboard_navigation_uses_updated_panel_and_series() {
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
        .expect("return keyboard focus to the primary panel");
    assert_eq!(viewer.active_panel, 0);

    viewer
        .handle_events(&[
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
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("apply panel focus and both series advances in event order");

    assert_eq!(viewer.active_panel, 1);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(3));
    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(FOURTH_SERIES_UID)
    );
}

#[test]
fn queued_navigation_recomputes_after_a_real_series_load_failure() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    for entry in std::fs::read_dir(study.path()).expect("read fixture study") {
        let entry = entry.expect("read fixture entry");
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with(SECOND_SERIES_UID)
        {
            std::fs::remove_file(entry.path()).expect("remove the selected series payload");
        }
    }

    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("retain the current panel after both failed series loads");

    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        viewer
            .series_browser
            .as_ref()
            .map(|browser| browser.active_index()),
        Some(0)
    );
    assert_eq!(viewer.compare_panels[0].series_index, None);
}

#[test]
fn queued_navigation_from_an_empty_panel_uses_the_latest_browser_series() {
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
    viewer.refresh_frame().expect("render the primary series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("focus the populated primary panel before the queued browse");
    assert_eq!(viewer.active_panel, 0);

    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
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
        .expect("advance, focus the empty panel, then advance again");

    assert_eq!(viewer.primary_series_index, Some(1));
    assert_eq!(viewer.compare_panels[0].series_index, Some(2));
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(THIRD_SERIES_UID)
    );
    assert_eq!(
        viewer
            .series_browser
            .as_ref()
            .map(|browser| browser.active_index()),
        Some(2)
    );
}

#[test]
fn queued_viewer_keys_follow_panel_activation_in_the_same_batch() {
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
        .expect("return keyboard focus to the primary panel");
    assert_eq!(viewer.active_panel, 0);

    let primary_slice = viewer.app.viewer_state.slice_index;
    let secondary_slice = viewer.compare_panels[0].app.viewer_state.slice_index;
    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x09,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x28,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("apply panel focus before the following image navigation key");

    assert_eq!(viewer.active_panel, 1);
    assert_eq!(viewer.app.viewer_state.slice_index, primary_slice);
    assert_ne!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        secondary_slice
    );
}

#[test]
fn queued_viewer_keys_follow_panel_activation_while_maximized() {
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
    let primary_slice = viewer.app.viewer_state.slice_index;
    let secondary_slice = viewer.compare_panels[0].app.viewer_state.slice_index;
    viewer
        .toggle_panel_maximize(0)
        .expect("maximize the primary panel");
    viewer.refresh_frame().expect("render the maximized panel");

    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x09,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x28,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("advance the maximized panel before image navigation");

    assert_eq!(
        viewer.maximized_panel.map(|panel| panel.panel_index),
        Some(1)
    );
    assert_eq!(viewer.active_panel, 0);
    assert_eq!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        primary_slice
    );
    assert_ne!(viewer.app.viewer_state.slice_index, secondary_slice);
}

#[test]
fn queued_series_browse_follows_maximized_panel_activation() {
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

    viewer
        .handle_events(&[
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
        .expect("browse the series displayed after changing the maximized panel");

    assert_eq!(
        viewer.maximized_panel.map(|panel| panel.panel_index),
        Some(1)
    );
    assert_eq!(viewer.primary_series_index, Some(2));
    assert_eq!(viewer.compare_panels[0].series_index, Some(0));
    assert_eq!(loaded_series_uid(&viewer.app), Some(THIRD_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(fixtures::SERIES_UID)
    );
}
