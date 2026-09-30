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
fn opening_a_study_skips_an_undecodable_first_series() {
    let root = tempfile::tempdir().expect("study root");
    fixtures::write_grayscale_presentation(root.path(), "MONOCHROME3", None)
        .expect("write a catalogued series with unsupported pixel presentation");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
        .expect("write a valid later series");

    let (mut viewer, _initial_root) = session();
    viewer
        .open_study_path(root.path())
        .expect("load the first readable series after an earlier decode failure");

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
    assert!(viewer
        .app
        .status_message
        .contains("skipped 1 unreadable series"));
}

#[test]
fn unopenable_replacement_study_preserves_every_populated_panel() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .open_selected_series(&[0, 2])
        .expect("fill both comparison panels");
    let original_primary = viewer
        .app
        .loaded
        .as_ref()
        .and_then(|volume| volume.metadata.as_ref())
        .and_then(|metadata| metadata.series_instance_uid.as_deref())
        .expect("primary DICOM series")
        .to_owned();
    let original_secondary = viewer.compare_panels[0]
        .app
        .loaded
        .as_ref()
        .and_then(|volume| volume.metadata.as_ref())
        .and_then(|metadata| metadata.series_instance_uid.as_deref())
        .expect("comparison DICOM series")
        .to_owned();
    let original_layout = viewer.workspace_layout;

    let invalid = tempfile::tempdir().expect("invalid replacement study root");
    fixtures::write_grayscale_presentation(invalid.path(), "MONOCHROME3", None)
        .expect("write an undecodable DICOM series");
    let error = viewer
        .open_study_path(invalid.path())
        .expect_err("reject a replacement with no readable series");

    assert!(error.to_string().contains("no series"));
    assert_eq!(viewer.workspace_layout, original_layout);
    assert_eq!(viewer.active_panel, 0);
    assert_eq!(viewer.compare_panels.len(), 1);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(2));
    assert_eq!(
        viewer
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(original_primary.as_str())
    );
    assert_eq!(
        viewer.compare_panels[0]
            .app
            .loaded
            .as_ref()
            .and_then(|volume| volume.metadata.as_ref())
            .and_then(|metadata| metadata.series_instance_uid.as_deref()),
        Some(original_secondary.as_str())
    );
    assert_eq!(
        viewer
            .series_browser
            .as_ref()
            .expect("current catalog remains active")
            .len(),
        4
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
