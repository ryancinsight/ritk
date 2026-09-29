use super::*;

#[test]
fn oblique_startup_keeps_a_multi_series_selector_open_until_selection() {
    let root = tempfile::tempdir().expect("multi-series study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write first series");
    fixtures::write_study(root.path(), "MR", "2.25.20260905002").expect("write second series");
    let mut app = SnapApp::default();
    let selection = prepare_initial_study(&mut app, root.path(), false)
        .expect("discover initial series choices")
        .expect("multiple series require a chooser");

    let mut session = NativeViewerSession::new_with_selection(
        app,
        Arc::new(NativeViewerObservation::default()),
        false,
        NativePresentationSelection::Oblique,
        false,
        Some(selection),
    )
    .expect("empty oblique layout must present the series chooser");
    assert!(session.app.loaded.is_none());
    assert!(session.oblique_viewport.is_none());
    assert_eq!(session.selection.as_ref().expect("series chooser").len(), 2);

    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("load the selected initial series");

    assert!(session.app.loaded.is_some());
    assert!(session.selection.is_none());
    assert!(session.oblique_viewport.is_some());
    assert!(session
        .oblique
        .as_ref()
        .and_then(|view| view.plane)
        .is_some());
}

#[test]
fn selecting_a_new_geometry_rebuilds_the_oblique_frame_before_routing() {
    let mut session = oblique_session();
    let previous_shape = session.app.loaded.as_ref().expect("initial volume").shape;
    let root = tempfile::tempdir().expect("replacement study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
        .expect("write first replacement series");
    fixtures::write_study(root.path(), "MR", "2.25.20260905002")
        .expect("write second replacement series");
    session
        .open_study_path(root.path())
        .expect("discover replacement series");
    session
        .refresh_frame()
        .expect("present replacement chooser");

    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("select replacement geometry before routing input");

    let volume = session.app.loaded.as_ref().expect("replacement volume");
    assert_ne!(volume.shape, previous_shape);
    assert!(session.selection.is_none());
    let oblique = session.oblique.as_ref().expect("retained oblique view");
    let plane = oblique.plane.expect("replacement oblique plane");
    plane
        .validate_source(volume)
        .expect("oblique plane belongs to the selected source");
    assert_eq!(
        [oblique.frame.width(), oblique.frame.height()],
        [
            u32::try_from(plane.dimensions()[0]).expect("bounded plane width"),
            u32::try_from(plane.dimensions()[1]).expect("bounded plane height"),
        ]
    );
    session
        .oblique_viewport
        .expect("replacement viewport")
        .validate_dimensions(plane.dimensions())
        .expect("replacement pointer mapper matches the rendered plane");
}
