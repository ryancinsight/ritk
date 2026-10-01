use super::*;

#[test]
fn closing_an_empty_selected_panel_preserves_the_survivor_assignments() {
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
    viewer.refresh_frame().expect("render the primary series");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x09,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("select the empty second panel");
    assert_eq!(viewer.active_panel, 1);

    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x73,
            repeated: false,
            modifiers: ModifierState::CONTROL,
        }])
        .expect("close the selected empty panel");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
    assert_eq!(viewer.active_panel, 1);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, None);
    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
}

#[test]
fn queued_series_browse_targets_the_surviving_panel_after_close() {
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
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x09,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x09,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("focus the second panel");
    assert_eq!(viewer.active_panel, 1);

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
        .expect("browse the surviving active panel after closing its predecessor");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
    assert_eq!(viewer.active_panel, 1);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(3));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(FOURTH_SERIES_UID)
    );
}

#[test]
fn queued_browse_after_close_uses_primary_for_an_empty_surviving_panel() {
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
        .assign_series_to_panel(2, 1)
        .expect("load the third series into panel two");
    viewer.refresh_frame().expect("render the populated panels");
    assert_eq!(viewer.active_panel, 1);

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
        .expect("close the populated panel, then browse from primary in its empty survivor");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
    assert_eq!(viewer.active_panel, 1);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(1));
    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(SECOND_SERIES_UID)
    );
    assert_eq!(
        viewer
            .series_browser
            .as_ref()
            .map(|browser| browser.active_index()),
        Some(1)
    );
}
