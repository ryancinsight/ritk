//! Value-semantic tests for bounded asynchronous viewer loads.

use super::*;
use crate::app::volume_input::VolumeInput;
use crate::dicom::loader::tests::fixtures;

#[test]
fn completed_task_publishes_a_real_dicom_volume() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
        .expect("write fixture study");
    let path = root.path().join(&files[0].0);
    let mut app = SnapApp::default();
    app.pending_load = Some(VolumeInput::Path(path));

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

#[test]
fn superseded_task_cannot_replace_newer_primary_study() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
        .expect("write fixture study");
    let valid = root.path().join(&files[0].0);
    let missing = root.path().join("missing.dcm");
    let mut app = SnapApp::default();
    app.pending_load = Some(VolumeInput::Path(missing));
    app.process_pending_loads();
    app.pending_load = Some(VolumeInput::Path(valid));
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
fn closing_study_cancels_pending_tasks_and_clears_requests() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
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
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
        .expect("write fixture study");
    let valid = root.path().join(&files[0].0);
    let mut app = SnapApp::default();
    app.load_primary(VolumeInput::Path(valid));
    let previous = std::sync::Arc::clone(&app.loaded.as_ref().expect("initial study").data);

    app.pending_load = Some(VolumeInput::Path(root.path().join("missing.dcm")));
    app.process_pending_loads();
    app.wait_for_load_tasks();

    assert!(std::sync::Arc::ptr_eq(
        &previous,
        &app.loaded.as_ref().expect("previous study retained").data
    ));
    assert!(app.status_message.starts_with("Volume load failed:"));
}
