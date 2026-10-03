use super::*;

fn series_uid(app: &crate::app::SnapApp) -> Option<&str> {
    app.loaded
        .as_ref()?
        .metadata
        .as_ref()?
        .series_instance_uid
        .as_deref()
}

#[test]
fn maximizing_and_switching_panels_preserves_each_series_state() {
    let (mut viewer, _root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("two-panel layout"),
        ))
        .expect("select comparison layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series");
    viewer.compare_panels[0].app.zoom = 1.75;
    viewer.compare_panels[0].app.viewer_state.slice_index = 2;

    assert!(viewer
        .toggle_panel_maximize(1)
        .expect("maximize the second panel"));
    viewer.refresh_frame().expect("render maximized series");
    assert_eq!(series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(
        series_uid(&viewer.compare_panels[0].app),
        Some(fixtures::SERIES_UID)
    );
    assert_eq!(viewer.app.zoom, 1.75);
    assert_eq!(viewer.app.viewer_state.slice_index, 2);
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(1)
    );

    assert!(viewer
        .activate_previous_panel()
        .expect("switch to the preceding maximized panel"));
    viewer.refresh_frame().expect("render the preceding series");
    assert_eq!(series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        viewer.maximized_panel.map(|panel| panel.panel_index),
        Some(0)
    );

    assert!(viewer
        .restore_maximized_panel()
        .expect("restore panel grid"));
    viewer.refresh_frame().expect("render restored comparison");
    assert_eq!(series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        series_uid(&viewer.compare_panels[0].app),
        Some(SECOND_SERIES_UID)
    );
    assert_eq!(viewer.compare_panels[0].app.zoom, 1.75);
    assert_eq!(viewer.compare_panels[0].app.viewer_state.slice_index, 2);
    assert_eq!(viewer.active_panel, 0);
}

#[test]
fn selecting_another_series_restores_panel_identity_before_activation() {
    let (mut viewer, _root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("two-panel layout"),
        ))
        .expect("select comparison layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series");
    viewer
        .toggle_panel_maximize(1)
        .expect("maximize the second series");

    viewer
        .select_series(0)
        .expect("select the original primary series");
    viewer.refresh_frame().expect("render restored comparison");

    assert_eq!(series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        series_uid(&viewer.compare_panels[0].app),
        Some(SECOND_SERIES_UID)
    );
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
    assert!(viewer.maximized_panel.is_none());
    assert_eq!(viewer.active_panel, 0);
    assert_eq!(
        viewer
            .series_browser
            .as_ref()
            .map(SeriesBrowser::active_index),
        Some(0)
    );
}

#[test]
fn control_click_opens_an_already_displayed_series_in_an_independent_panel() {
    let (mut viewer, _root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");

    assert!(viewer
        .assign_series_to_next_panel(0)
        .expect("open the displayed series in another panel"));

    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(0));
    assert_eq!(series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(
        series_uid(&viewer.compare_panels[0].app),
        Some(fixtures::SERIES_UID)
    );
    assert!(std::sync::Arc::ptr_eq(
        &viewer.app.loaded.as_ref().expect("primary volume").data,
        &viewer.compare_panels[0]
            .app
            .loaded
            .as_ref()
            .expect("comparison volume")
            .data
    ));
    assert_eq!(viewer.active_panel, 1);
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );

    viewer.compare_panels[0].app.zoom = 1.75;
    assert_ne!(viewer.app.zoom, viewer.compare_panels[0].app.zoom);
}

#[test]
fn control_click_preserves_populated_panels_when_one_is_maximized() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .open_selected_series(&[0, 1])
        .expect("open the first two series");
    viewer
        .toggle_panel_maximize(1)
        .expect("maximize the second series");

    viewer
        .assign_series_to_next_panel(2)
        .expect("append the third series to another panel");

    assert_eq!(series_uid(&viewer.app), Some(fixtures::SERIES_UID));
    assert_eq!(viewer.compare_panels[0].series_index, Some(1));
    assert_eq!(viewer.compare_panels[1].series_index, Some(2));
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(3)
    );
}

#[test]
fn control_click_preserves_populated_panels_hidden_by_a_smaller_grid() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .open_selected_series(&[0, 1, 2])
        .expect("open three series");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("two-panel layout"),
        ))
        .expect("hide the third panel without discarding its series");

    viewer
        .assign_series_to_next_panel(3)
        .expect("append a fourth series without replacing the hidden third");

    assert_eq!(viewer.compare_panels[0].series_index, Some(1));
    assert_eq!(viewer.compare_panels[1].series_index, Some(2));
    assert_eq!(viewer.compare_panels[2].series_index, Some(3));
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(4)
    );
}

#[test]
fn closing_panels_promotes_and_reflows_remaining_series() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 2).expect("four-panel layout"),
        ))
        .expect("select four-panel layout");
    for index in 1..4 {
        viewer
            .assign_series_to_panel(index, index)
            .expect("load series into panel");
    }

    assert!(viewer.close_panel(0).expect("close primary series"));
    assert_eq!(series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(
        series_uid(&viewer.compare_panels[0].app),
        Some(THIRD_SERIES_UID)
    );
    assert_eq!(
        series_uid(&viewer.compare_panels[1].app),
        Some(FOURTH_SERIES_UID)
    );
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(3)
    );

    assert!(viewer
        .close_panel(1)
        .expect("close the second visible series"));
    assert_eq!(series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(
        series_uid(&viewer.compare_panels[0].app),
        Some(FOURTH_SERIES_UID)
    );
    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(2)
    );
}

#[test]
fn close_all_clears_open_panels_and_keeps_the_study_catalog() {
    let (mut viewer, _root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("two-panel layout"),
        ))
        .expect("select comparison layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series");
    let catalog_len = viewer.series_browser.as_ref().expect("study catalog").len();

    assert!(viewer.close_all_panels().expect("close every panel"));
    assert_eq!(viewer.workspace_layout, WorkspaceLayout::Orthogonal);
    assert!(viewer.app.loaded.is_none());
    assert!(viewer
        .series_browser
        .as_ref()
        .is_some_and(|browser| browser.len() == catalog_len));
    assert!(viewer.compare_panels.is_empty());
}
