//! Value-semantic tests for bounded asynchronous viewer loads.

use super::*;
use crate::app::volume_input::VolumeInput;
use crate::dicom::loader::tests::fixtures;

#[test]
fn completed_task_publishes_a_real_dicom_volume() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
        .expect("write fixture study");
    let paths: Vec<std::path::PathBuf> = files
        .iter()
        .map(|(name, _)| root.path().join(name))
        .collect();
    let acquisition = std::sync::Arc::new(ritk_io::DicomSeriesInfo::new(
        fixtures::SERIES_UID,
        "CT series".to_owned(),
        "CT",
        String::new(),
        paths,
    ));
    let mut app = SnapApp::default();
    app.pending_load = Some(VolumeInput::Series(acquisition));

    app.process_pending_loads();
    app.wait_for_load_tasks();

    let volume = app.loaded.as_ref().expect("task publishes volume");
    assert_eq!(volume.shape, fixtures::SHAPE);
    assert_eq!(volume.spacing, fixtures::SPACING);
    assert_eq!(
        volume
            .metadata
            .as_ref()
            .expect("metadata")
            .series_instance_uid
            .as_deref(),
        Some(fixtures::SERIES_UID)
    );
}

fn series_input(uid: &str, files: Vec<std::path::PathBuf>) -> VolumeInput {
    VolumeInput::Series(std::sync::Arc::new(ritk_io::DicomSeriesInfo::new(
        uid,
        "CT series".to_owned(),
        "CT",
        String::new(),
        files,
    )))
}

#[test]
fn superseded_task_cannot_replace_newer_primary_study() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
        .expect("write fixture study");
    let valid: Vec<std::path::PathBuf> = files
        .iter()
        .map(|(name, _)| root.path().join(name))
        .collect();
    let missing = vec![root.path().join("missing.dcm")];
    let mut app = SnapApp::default();
    app.pending_load = Some(series_input("1.2.3.missing", missing));
    app.process_pending_loads();
    app.pending_load = Some(series_input(fixtures::SERIES_UID, valid));
    app.process_pending_loads();
    app.wait_for_load_tasks();

    let volume = app.loaded.as_ref().expect("newest task publishes");
    assert_eq!(
        volume
            .metadata
            .as_ref()
            .expect("metadata")
            .series_instance_uid
            .as_deref(),
        Some(fixtures::SERIES_UID)
    );
    assert!(!app.status_message.starts_with("Volume load failed:"));
}

#[test]
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
fn closing_study_cancels_pending_tasks_and_clears_requests() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
        .expect("write fixture study");
    let path = root.path().join(&files[0].0);
    let mut app = SnapApp::default();
    app.pending_load = Some(VolumeInput::Path(path));
    app.process_pending_loads();
    app.close_study();
    app.wait_for_load_tasks();

    assert!(app.loaded.is_none());
    assert!(app.pending_load.is_none());
    assert!(app.pending_secondary_load.is_none());
    assert_eq!(app.status_message, "Study closed.");
}

#[test]
fn current_task_failure_preserves_the_previous_study() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
        .expect("write fixture study");
    let valid: Vec<std::path::PathBuf> = files
        .iter()
        .map(|(name, _)| root.path().join(name))
        .collect();
    let mut app = SnapApp::default();
    app.load_primary(series_input(fixtures::SERIES_UID, valid));
    let previous = std::sync::Arc::clone(&app.loaded.as_ref().expect("initial study").data);

    let missing = vec![root.path().join("missing.dcm")];
    app.pending_load = Some(series_input("1.2.3.missing", missing));
    app.process_pending_loads();
    app.wait_for_load_tasks();

    assert!(std::sync::Arc::ptr_eq(
        &previous,
        &app.loaded.as_ref().expect("previous study retained").data
    ));
    assert!(app.status_message.starts_with("Volume load failed:"));
}
