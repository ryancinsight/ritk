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
