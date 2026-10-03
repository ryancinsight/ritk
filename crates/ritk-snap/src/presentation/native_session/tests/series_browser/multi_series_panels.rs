use super::*;

#[test]
fn multi_series_selection_opens_each_selected_study_series_in_its_own_panel() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    let catalog = viewer
        .series_browser
        .as_ref()
        .expect("study catalog remains available");
    let expected_uids = [0, 2, 3].map(|index| {
        catalog
            .choice(index)
            .map(|choice| choice.acquisition.series_instance_uid())
            .expect("fixture series has an instance UID")
            .to_owned()
    });

    viewer
        .open_selected_series(&[0, 2, 3])
        .expect("load selected series transactionally");
    viewer.refresh_frame().expect("render all selected series");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(3)
    );
    assert_eq!(viewer.viewports.len(), 3);
    assert_eq!(viewer.active_panel, 0);
    assert_eq!(viewer.primary_series_index, Some(0));
    let actual_uids = std::iter::once(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
    )
    .chain(viewer.compare_panels.iter().map(|panel| {
        panel
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref())
    }))
    .map(|uid| uid.expect("selected panel has DICOM metadata").to_owned())
    .collect::<Vec<_>>();
    assert_eq!(actual_uids, expected_uids);
    for (index, viewport) in viewer.viewports.iter().enumerate() {
        let (x, y) = viewport.center();
        assert_ne!(
            viewer.framebuffer.get_pixel(x, y),
            metis_platform::Color::BLACK,
            "selected panel {index} must display its own pixels"
        );
    }
}

#[test]
fn multi_series_selection_pads_rounded_grid_with_empty_panels() {
    let (mut viewer, _initial_root) = session();
    let study = tempfile::tempdir().expect("create seven-series study");
    for index in 0..7 {
        let uid = format!("2.25.202609050{index:02}");
        fixtures::write_study(study.path(), "MR", &uid).expect("write DICOM series");
    }
    viewer
        .open_study_path(study.path())
        .expect("open seven-series study");
    let selected = (0..7).collect::<Vec<_>>();

    viewer
        .open_selected_series(&selected)
        .expect("load all seven selected series");
    viewer.refresh_frame().expect("render the rounded grid");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::dimensions),
        Some((4, 2))
    );
    assert_eq!(viewer.viewports.len(), 8);
    assert_eq!(viewer.compare_panels.len(), 7);
    assert_eq!(viewer.compare_panels[5].series_index, Some(6));
    assert!(viewer.compare_panels[6].series_index.is_none());
}

#[test]
fn rejected_multi_series_selection_preserves_the_current_viewer() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    let original_uid = viewer
        .app
        .loaded
        .as_ref()
        .and_then(|volume| volume.metadata.as_ref())
        .and_then(|metadata| metadata.series_instance_uid.as_deref())
        .expect("initial viewer has DICOM series metadata")
        .to_owned();
    let error = viewer
        .open_selected_series(&[1, usize::MAX])
        .expect_err("reject an index outside the study catalog");

    assert!(error.to_string().contains("outside the catalog"));
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.workspace_layout, WorkspaceLayout::Orthogonal);
    assert_eq!(viewer.compare_panels.len(), 0);
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(original_uid.as_str())
    );
}

#[test]
fn control_click_opens_a_series_in_the_next_available_panel() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    let (x, y) = series_card_center(&viewer, 2);

    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x11,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::KeyUp {
                virtual_key: 0x11,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("Control-click the series into a new panel");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::dimensions),
        Some((2, 1))
    );
    assert_eq!(viewer.active_panel, 1);
    assert_eq!(viewer.compare_panels[0].series_index, Some(2));
}

#[test]
fn f4_picker_opens_every_keyboard_selected_series_in_the_native_window() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x73,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("open the multiple-series picker with F4");
    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x20,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::TextInput { character: ' ' },
            WindowEvent::KeyDown {
                virtual_key: 0x28,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x20,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::TextInput { character: ' ' },
            WindowEvent::KeyDown {
                virtual_key: 0x0d,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("select and open both series with the keyboard");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
    assert_eq!(viewer.viewports.len(), 2);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(1));
    assert!(!viewer.window_chrome.multi_series_dialog_is_open());
    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(SECOND_SERIES_UID)
    );

    viewer.refresh_frame().expect("render the series panels");
    let (x, y) = viewer.viewports[1].center();
    viewer
        .handle_events(&[WindowEvent::PointerWheel {
            x,
            y,
            delta_x: 120,
            delta_y: 0,
            modifiers: ModifierState::NONE,
        }])
        .expect("browse from the series under the horizontal wheel");

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
fn series_browsing_replaces_the_active_panel_without_switching_to_an_existing_panel() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid"),
        ))
        .expect("select two-panel grid");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load a second series into the right panel");
    viewer
        .assign_series_to_panel(0, 0)
        .expect("focus the left panel");

    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x27,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("browse to the next series from the active panel");

    assert_eq!(viewer.active_panel, 0);
    assert_eq!(viewer.primary_series_index, Some(1));
    assert_eq!(viewer.compare_panels[0].series_index, Some(1));
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(SECOND_SERIES_UID)
    );
}

#[test]
fn four_panel_layout_assigns_four_independent_dicom_series() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 2).expect("four-panel grid is within the picker bounds"),
        ))
        .expect("select four-panel layout");
    viewer.refresh_frame().expect("render four panel targets");

    click_series(&mut viewer, 1);
    for (panel_index, series_index) in [(2, 2), (3, 3)] {
        let (x, y) = viewer.viewports[panel_index].center();
        viewer
            .handle_events(&[
                WindowEvent::PointerDown {
                    x,
                    y,
                    button: MouseButton::Left,
                },
                WindowEvent::PointerUp {
                    x,
                    y,
                    button: MouseButton::Left,
                },
            ])
            .expect("choose the next empty series panel");
        click_series(&mut viewer, series_index);
    }

    assert_eq!(viewer.viewports.len(), 4);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels.len(), 3);
    let displayed_uids = std::iter::once(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
    )
    .chain(viewer.compare_panels.iter().map(|panel| {
        panel
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref())
    }))
    .collect::<Vec<_>>();
    assert_eq!(
        displayed_uids,
        [
            Some(fixtures::SERIES_UID),
            Some(SECOND_SERIES_UID),
            Some(THIRD_SERIES_UID),
            Some(FOURTH_SERIES_UID),
        ]
    );
    for index in 0..4 {
        let (x, y) = viewer.viewports[index].center();
        assert_ne!(
            viewer.framebuffer.get_pixel(x, y),
            metis_platform::Color::BLACK,
            "panel {index} must display its assigned series"
        );
    }
}

#[test]
fn selecting_a_series_assigned_to_a_hidden_panel_restores_its_panel() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 2).expect("four-panel grid is within the picker bounds"),
        ))
        .expect("select four-panel layout");
    viewer
        .assign_series_to_panel(3, 3)
        .expect("assign the fourth series to panel four");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("two-panel grid is within the picker bounds"),
        ))
        .expect("hide the lower row");

    click_series(&mut viewer, 3);

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::dimensions),
        Some((4, 1))
    );
    assert_eq!(viewer.active_panel, 3);
    assert_eq!(viewer.viewports.len(), 4);
    assert_eq!(viewer.compare_panels[2].series_index, Some(3));
    let (x, y) = viewer.viewports[3].center();
    assert_ne!(
        viewer.framebuffer.get_pixel(x, y),
        metis_platform::Color::BLACK
    );
}

#[test]
fn cine_advances_every_visible_series_panel_in_one_event_batch() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(3, 1).expect("three-panel grid is within the picker bounds"),
        ))
        .expect("select three-panel layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("assign the second series to panel two");
    viewer
        .assign_series_to_panel(2, 2)
        .expect("assign the third series to panel three");
    viewer.app.cine.set_fps(10.0);
    viewer.app.cine.set_enabled(true, 0.0);
    for panel in viewer.compare_panels.iter_mut().take(2) {
        panel.app.cine.set_fps(10.0);
        panel.app.cine.set_enabled(true, 0.0);
    }
    assert!(viewer.tick_cine_at(0.25));

    let updated_slices = [
        viewer.app.viewer_state.slice_index,
        viewer.compare_panels[0].app.viewer_state.slice_index,
        viewer.compare_panels[1].app.viewer_state.slice_index,
    ];
    assert_eq!(updated_slices, [0, 0, 0]);
}

#[test]
fn dragging_a_series_card_assigns_only_the_drop_target_panel() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 2).expect("four-panel grid is within the picker bounds"),
        ))
        .expect("select four-panel layout");
    viewer.refresh_frame().expect("render four panel targets");

    drag_series_to_panel(&mut viewer, 2, 3);

    assert_eq!(viewer.active_panel, 3);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, None);
    assert_eq!(viewer.compare_panels[1].series_index, None);
    assert_eq!(viewer.compare_panels[2].series_index, Some(2));
    assert_eq!(
        viewer.compare_panels[2]
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(THIRD_SERIES_UID)
    );
}

include!("comparison_startup.rs");
