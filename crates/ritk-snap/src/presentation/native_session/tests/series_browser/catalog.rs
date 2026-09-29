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
fn undecodable_first_series_keeps_the_catalog_and_allows_a_later_series() {
    let root = tempfile::tempdir().expect("study root");
    fixtures::write_grayscale_presentation(root.path(), "MONOCHROME3", None)
        .expect("write a catalogued series with unsupported pixel presentation");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
        .expect("write a valid later series");

    let (mut viewer, _initial_root) = session();
    let previous_uid = viewer
        .app
        .loaded
        .as_ref()
        .and_then(|volume| volume.metadata.as_ref())
        .and_then(|metadata| metadata.series_instance_uid);
    viewer
        .open_study_path(root.path())
        .expect("keep the discoverable series catalog when the first decode fails");

    let browser = viewer.series_browser.as_ref().expect("retained catalog");
    assert_eq!(browser.len(), 2);
    assert_eq!(
        browser
            .choice(0)
            .expect("first discovered series")
            .acquisition
            .series_instance_uid(),
        "2.25.20260905005"
    );
    assert_eq!(viewer.primary_series_index, None);
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        previous_uid.as_deref()
    );
    assert!(viewer.app.status_message.contains("select another series"));

    click_series(&mut viewer, 1);

    assert_eq!(viewer.primary_series_index, Some(1));
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(fixtures::SERIES_UID)
    );
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
            x: 100,
            y: 700,
            button: MouseButton::Left,
        }])
        .expect("navigator input");

    assert_eq!(flow, NativeFlow::Continue { repaint: false });
    assert_eq!(viewer.active_view, None);
}

#[test]
fn horizontal_keys_and_wheel_switch_between_study_series() {
    let (mut viewer, _initial_root) = session();
    let replacement = replacement_study();
    viewer
        .open_study_path(replacement.path())
        .expect("open replacement study");

    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x27,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("select the next series with the right arrow");
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));

    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x25,
            repeated: true,
            modifiers: ModifierState::NONE,
        }])
        .expect("ignore a repeated left arrow");
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));

    let (x, y) = viewer.viewports[0].center();
    viewer
        .handle_events(&[WindowEvent::PointerWheel {
            x,
            y,
            delta_x: -120,
            delta_y: 0,
            modifiers: ModifierState::NONE,
        }])
        .expect("select the previous series with horizontal wheel input");
    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));

    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x23,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("select the last series with End");
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));

    viewer
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x24,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("select the first series with Home");
    assert_eq!(loaded_series_uid(&viewer.app), Some(fixtures::SERIES_UID));
}
