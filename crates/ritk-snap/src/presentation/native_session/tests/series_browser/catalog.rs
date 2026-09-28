use super::*;

#[test]
fn native_study_navigator_retains_and_loads_each_discovered_series() {
    let (mut viewer, _initial_root) = session();
    let replacement = replacement_study();
    viewer
        .open_study_path(replacement.path())
        .expect("open replacement study");

    let browser = viewer.series_browser.as_ref().expect("study navigator");
    assert_eq!(browser.len(), 2);
    assert_eq!(browser.active_index(), 0);
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .expect("active volume")
            .modality
            .as_deref(),
        Some("CT")
    );

    click_series(&mut viewer, 1);

    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .expect("selected volume")
            .metadata
            .as_ref()
            .expect("selected metadata")
            .series_instance_uid
            .as_deref(),
        Some(SECOND_SERIES_UID)
    );
    let browser = viewer.series_browser.as_ref().expect("retained navigator");
    assert_eq!(browser.len(), 2);
    assert_eq!(browser.active_index(), 1);
}

#[test]
fn failed_series_switch_preserves_the_active_volume_and_catalog() {
    let (mut viewer, _initial_root) = session();
    let replacement = replacement_study();
    viewer
        .open_study_path(replacement.path())
        .expect("open replacement study");
    let previous_volume = viewer.app.loaded.as_ref().expect("active volume");
    let previous_uid = previous_volume
        .metadata
        .as_ref()
        .expect("active metadata")
        .series_instance_uid;
    let previous_shape = previous_volume.shape;
    let missing_paths = viewer
        .series_browser
        .as_ref()
        .expect("study navigator")
        .choice(1)
        .expect("second series")
        .acquisition
        .file_paths
        .clone();
    for path in missing_paths {
        std::fs::remove_file(path).expect("remove selected series instance");
    }

    click_series(&mut viewer, 1);

    assert_eq!(
        viewer.app.loaded.as_ref().expect("active volume").shape,
        previous_shape
    );
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .expect("active volume")
            .metadata
            .as_ref()
            .expect("active metadata")
            .series_instance_uid,
        previous_uid
    );
    assert_eq!(
        viewer
            .series_browser
            .as_ref()
            .expect("retained navigator")
            .active_index(),
        0
    );
    assert!(viewer
        .app
        .status_message
        .contains("Series could not be opened"));
}

#[test]
fn study_browser_input_does_not_start_a_viewport_gesture() {
    let (mut viewer, _initial_root) = session();
    let flow = viewer
        .handle_events(&[WindowEvent::PointerDown {
            x: 16,
            y: 700,
            button: MouseButton::Left,
        }])
        .expect("navigator input");

    assert_eq!(flow, NativeFlow::Continue { repaint: false });
    assert_eq!(viewer.active_view, None);
}
