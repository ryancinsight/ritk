use super::*;

#[test]
fn comparison_displays_two_independent_series_and_routes_each_panel() {
    let (mut viewer, _initial_root) = session();
    let replacement = replacement_study();
    viewer
        .open_study_path(replacement.path())
        .expect("open two-series study");
    viewer.refresh_frame().expect("render first series");

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 650,
                y: 65,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 650,
                y: 65,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: 632,
                y: 151,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 632,
                y: 151,
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

#[test]
fn initial_comparison_loads_distinct_requested_series_into_both_panels() {
    let root = replacement_study();
    let tree = scan_folder_for_series(root.path()).expect("scan comparison study");
    let browser =
        SeriesBrowser::from_tree(&tree, Some(fixtures::SERIES_UID)).expect("select primary series");
    let primary = load_volume_from_series_info(
        &browser
            .choice(browser.active_index())
            .expect("primary series")
            .acquisition,
    )
    .expect("load primary series");
    let mut app = crate::app::SnapApp::default();
    app.load_volume(primary, "primary".to_owned());
    let viewer = crate::presentation::native_session::NativeViewerSession::new_with_browser(
        app,
        std::sync::Arc::new(
            crate::presentation::native_session::NativeViewerObservation::default(),
        ),
        false,
        crate::launch::NativePresentationSelection::Fixed(
            crate::launch::NativePresentationMode::Orthogonal,
        ),
        false,
        Some(browser),
        Some(SECOND_SERIES_UID),
    )
    .expect("initialize two-series comparison");

    assert!(viewer.workspace_layout.is_grid());
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(
        viewer
            .compare_panels
            .first()
            .and_then(|panel| panel.series_index),
        Some(1)
    );
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(fixtures::SERIES_UID)
    );
    assert_eq!(
        viewer
            .compare_panels
            .first()
            .and_then(|panel| panel.app.loaded.as_ref())
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(SECOND_SERIES_UID)
    );
}
