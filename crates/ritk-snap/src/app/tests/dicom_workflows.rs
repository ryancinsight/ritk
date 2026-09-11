//! Real Part 10 inputs through discovery, sidebar events, loading and session restore.

use crate::dicom::loader::tests::fixtures;

use crate::app::{state::SeriesLoadTarget, volume_input::VolumeInput, SnapApp};
use crate::dicom::series_tree::SeriesEntryView;
use crate::session::{StudySource, ViewerSessionSnapshot};

const SECONDARY_UID: &str = "2.25.20260905002";

fn assert_primary(app: &SnapApp, uid: &str) {
    let volume = app.loaded.as_ref().expect("primary volume loaded");
    assert_eq!(volume.shape, fixtures::SHAPE);
    assert_eq!(volume.spacing, fixtures::SPACING);
    assert_eq!(volume.origin, fixtures::ORIGIN);
    assert_eq!(volume.direction, fixtures::DIRECTION);
    let expected: Vec<_> = fixtures::SAMPLES
        .iter()
        .map(|&raw| 2.0 * f32::from(raw) - 20.0)
        .collect();
    assert_eq!(volume.data.as_slice(), expected);
    assert_eq!(
        volume
            .metadata
            .as_ref()
            .expect("metadata")
            .series_instance_uid
            .as_deref(),
        Some(uid)
    );
}

fn sidebar_frame(
    app: &mut SnapApp,
    context: &egui::Context,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut input = egui::RawInput::default();
    input.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1000.0, 900.0),
    ));
    input.events = events;
    context.run(input, |context| app.show_left_panel(context))
}

fn click_series(app: &mut SnapApp, context: &egui::Context, uid: &str) {
    let label = app
        .series_tree
        .find_by_uid(uid)
        .expect("discovered acquisition")
        .display_label();
    let output = sidebar_frame(app, context, Vec::new());
    let center = output
        .shapes
        .iter()
        .find_map(|shape| {
            if let egui::Shape::Text(text) = &shape.shape {
                (text.galley.text() == label).then(|| text.pos + text.galley.size() * 0.5)
            } else {
                None
            }
        })
        .expect("actual sidebar includes acquisition label");
    for pressed in [true, false] {
        drop(sidebar_frame(
            app,
            context,
            vec![
                egui::Event::PointerMoved(center),
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        ));
    }
}

#[test]
fn sidebar_loads_distinct_same_folder_primary_and_secondary_acquisitions() {
    let root = tempfile::tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT");
    fixtures::write_study(root.path(), "MR", SECONDARY_UID).expect("write MR");
    let mut app = SnapApp::default();
    app.scan_for_series(root.path().to_path_buf());
    assert_eq!(app.series_tree.total_series(), 2);
    let discovered = app
        .series_tree
        .find_by_uid(SECONDARY_UID)
        .expect("MR discovery");
    assert_eq!(discovered.folder(), root.path());
    assert_eq!(discovered.acquisition.file_paths.len(), 3);
    let context = egui::Context::default();
    click_series(&mut app, &context, fixtures::SERIES_UID);
    assert!(
        matches!(&app.pending_load, Some(VolumeInput::Series(info)) if info.series_instance_uid() == fixtures::SERIES_UID)
    );
    app.process_pending_loads();
    app.wait_for_load_tasks();
    assert_primary(&app, fixtures::SERIES_UID);
    app.series_load_target = SeriesLoadTarget::Secondary;
    click_series(&mut app, &context, SECONDARY_UID);
    assert!(
        matches!(&app.pending_secondary_load, Some(VolumeInput::Series(info)) if info.series_instance_uid() == SECONDARY_UID)
    );
    app.process_pending_loads();
    app.wait_for_load_tasks();
    assert_primary(&app, fixtures::SERIES_UID);
    assert_eq!(
        app.loaded_secondary
            .as_ref()
            .expect("secondary loaded")
            .modality
            .as_deref(),
        Some("MR")
    );
    assert_eq!(
        app.selected_series
            .as_ref()
            .expect("successful highlight")
            .series_instance_uid(),
        SECONDARY_UID
    );
    assert!(app.compare_side_by_side);
}

#[test]
fn failed_primary_and_secondary_replacements_preserve_loaded_state() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT");
    let selected = root.path().join(&files.first().expect("slice").0);
    let mut app = SnapApp::default();
    app.load_primary(VolumeInput::Path(selected.clone()));
    app.load_secondary(VolumeInput::Path(selected));
    assert_primary(&app, fixtures::SERIES_UID);
    let pixels = std::sync::Arc::clone(&app.loaded.as_ref().expect("primary").data);
    let secondary = std::sync::Arc::clone(&app.loaded_secondary.as_ref().expect("secondary").data);
    app.viewer_state.slice_index = 1;
    app.zoom = 2.5;
    let missing = root.path().join("absent.dcm");
    app.load_primary(VolumeInput::Path(missing.clone()));
    assert!(std::sync::Arc::ptr_eq(
        &pixels,
        &app.loaded.as_ref().expect("primary retained").data
    ));
    assert_eq!(app.viewer_state.slice_index, 1);
    assert_eq!(app.zoom, 2.5);
    assert!(app.status_message.starts_with("Volume load failed:"));
    app.load_secondary(VolumeInput::Path(missing));
    assert!(std::sync::Arc::ptr_eq(
        &secondary,
        &app.loaded_secondary
            .as_ref()
            .expect("secondary retained")
            .data
    ));
    assert!(app
        .status_message
        .starts_with("Secondary volume load failed:"));
}

#[test]
fn session_restores_exact_acquisition_and_presentation_after_decode() {
    let root = tempfile::tempdir().expect("study root");
    let files = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT");
    let selected = root.path().join(&files.first().expect("slice").0);
    let mut app = SnapApp::default();
    app.load_primary(VolumeInput::Path(selected));
    assert_primary(&app, fixtures::SERIES_UID);
    app.viewer_state.slice_index = 2;
    app.zoom = 2.5;
    let snapshot = app.session_snapshot();
    assert!(
        matches!(&snapshot.source, Some(StudySource::Dicom { series_uid, files }) if series_uid == fixtures::SERIES_UID && files.len() == 3)
    );
    let bytes = serde_json::to_vec(&snapshot).expect("serialize acquisition");
    fixtures::write_study(root.path(), "MR", SECONDARY_UID).expect("add unrelated series");
    let mut restored = SnapApp::default();
    restored
        .apply_session_snapshot(serde_json::from_slice(&bytes).expect("deserialize acquisition"))
        .expect("restore acquisition");
    assert_primary(&restored, fixtures::SERIES_UID);
    assert_eq!(restored.viewer_state.slice_index, 2);
    assert_eq!(restored.zoom, 2.5);
    let mut wrong_uid: ViewerSessionSnapshot = snapshot;
    if let Some(StudySource::Dicom { series_uid, .. }) = &mut wrong_uid.source {
        *series_uid = SECONDARY_UID.to_owned();
    }
    wrong_uid.zoom = 9.0;
    let session_path = root.path().join("session.json");
    crate::session::save_to_file(&wrong_uid, &session_path).expect("save wrong UID session");
    restored.load_session_from_path(&session_path);
    assert_primary(&restored, fixtures::SERIES_UID);
    assert_eq!(restored.zoom, 2.5);
    assert!(restored
        .status_message
        .contains("SeriesInstanceUID changed"));
    assert!(restored.status_message.starts_with("Session load failed"));
    wrong_uid.source = Some(StudySource::Path(root.path().join("missing.dcm")));
    crate::session::save_to_file(&wrong_uid, &session_path).expect("save missing study session");
    restored.load_session_from_path(&session_path);
    assert_primary(&restored, fixtures::SERIES_UID);
    assert_eq!(restored.zoom, 2.5);
    assert!(restored.status_message.starts_with("Session load failed"));
    crate::session::save_to_file(&restored.session_snapshot(), &session_path)
        .expect("save valid study session");
    restored.load_session_from_path(&session_path);
    assert_primary(&restored, fixtures::SERIES_UID);
    assert!(restored.status_message.starts_with("Loaded session from"));
}

/// The supported file-set index owns membership, including when its directory
/// contains other image acquisitions. This tests Part 10/reference semantics,
/// not the full media-directory IOD's linked-record offset constraints.
#[test]
fn dicomdir_index_opens_referenced_pixels_and_excludes_unreferenced_acquisition() {
    use ritk_io::file_set_index_fixture::{write_file_set_index, FileSetIdentity, FileSetMember};

    let root = tempfile::tempdir().expect("indexed study root");
    let images = root.path().join("IMAGES");
    let slices = fixtures::write_study(&images, "CT", fixtures::SERIES_UID)
        .expect("write indexed acquisition");
    fixtures::write_study(&images, "MR", SECONDARY_UID)
        .expect("write unreferenced distractor acquisition");
    let mut referenced_paths = Vec::new();
    let mut file_ids = Vec::new();
    let mut sop_instance_uids = Vec::new();
    for ((_, bytes), index) in slices.iter().zip(1..=3) {
        let file_id = format!("SLICE{index:03}");
        let path = images.join(&file_id);
        std::fs::write(&path, bytes).expect("write DICOM file-set member");
        referenced_paths.push(path.canonicalize().expect("resolve written member"));
        file_ids.push(format!("IMAGES\\{file_id}"));
        sop_instance_uids.push(format!("{}.{index}", fixtures::SERIES_UID));
    }
    // The reader validates the full media-directory record tree -- in-use
    // flags, root chain, and lower-level offsets -- so the index is built by
    // the format crate rather than assembled here.
    let members: Vec<_> = file_ids
        .iter()
        .zip(&sop_instance_uids)
        .map(|(file_id, sop_instance_uid)| FileSetMember {
            file_id,
            sop_class_uid: "1.2.840.10008.5.1.4.1.1.7",
            sop_instance_uid,
            transfer_syntax_uid: "1.2.840.10008.1.2.1",
        })
        .collect();
    let identity = FileSetIdentity {
        study_instance_uid: "2.25.20260905",
        series_instance_uid: fixtures::SERIES_UID,
        modality: "CT",
    };
    let index = root.path().join("DICOMDIR");
    write_file_set_index(&index, identity, &members);

    let mut app = SnapApp::default();
    app.pending_load = Some(VolumeInput::Path(index.clone()));
    app.process_pending_loads();
    app.wait_for_load_tasks();
    assert_primary(&app, fixtures::SERIES_UID);
    let volume = app.loaded.as_ref().expect("indexed study loaded");
    assert_eq!(volume.source.as_deref(), Some(index.as_path()));
    let mut actual_paths: Vec<_> = volume
        .metadata
        .as_ref()
        .expect("indexed metadata")
        .slices
        .iter()
        .map(|slice| slice.path.clone())
        .collect();
    actual_paths.sort();
    referenced_paths.sort();
    assert_eq!(actual_paths, referenced_paths);
    assert_eq!(volume.modality.as_deref(), Some("CT"));
    for input in [root.path(), index.as_path()] {
        app.scan_for_series(input.to_path_buf());
        assert_eq!(app.series_tree.total_series(), 1);
        let selected = app
            .series_tree
            .find_by_uid(fixtures::SERIES_UID)
            .expect("indexed CT");
        assert_eq!(selected.acquisition.file_paths, referenced_paths);
        assert!(app.series_tree.find_by_uid(SECONDARY_UID).is_none());
    }
    std::fs::write(&index, b"corrupt DICOMDIR").expect("corrupt index");
    app.scan_for_series(root.path().to_path_buf());
    assert!(app.status_message.starts_with("Scan failed"));
    assert_eq!(app.series_tree.total_series(), 1);
    write_file_set_index(&index, identity, &members);
    std::fs::remove_file(&referenced_paths[0]).expect("remove referenced member");
    app.scan_for_series(root.path().to_path_buf());
    assert!(app.status_message.starts_with("Scan failed"));
    assert_eq!(app.series_tree.total_series(), 1);
    assert_primary(&app, fixtures::SERIES_UID);
}

#[test]
fn selected_image_replaced_by_non_image_rejects_partial_reload() {
    use ritk_io::{DicomObjectModel, DicomObjectNode, DicomTag};

    let root = tempfile::tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT");
    let mut app = SnapApp::default();
    app.load_primary(VolumeInput::Path(root.path().to_path_buf()));
    assert_primary(&app, fixtures::SERIES_UID);
    app.zoom = 2.5;
    let acquisition = app.selected_series.clone().expect("selected acquisition");
    let pixels = app.loaded.as_ref().expect("loaded CT").data.clone();
    let mut report = DicomObjectModel::new();
    for (element, value) in [
        (0x0016, "1.2.840.10008.5.1.4.1.1.88.11"),
        (0x0018, "2.25.20260905998"),
    ] {
        report.insert(DicomObjectNode::text(
            DicomTag::new(0x0008, element),
            "UI",
            value,
        ));
    }
    report.insert(DicomObjectNode::text(
        DicomTag::new(0x0020, 0x000e),
        "UI",
        fixtures::SERIES_UID,
    ));
    ritk_io::write_dicom_object(&report, &acquisition.file_paths[0])
        .expect("replace one selected image with structured report");
    app.load_primary(VolumeInput::Series(acquisition));
    assert!(app.status_message.contains("membership changed"));
    assert_primary(&app, fixtures::SERIES_UID);
    assert_eq!(app.zoom, 2.5);
    assert!(std::sync::Arc::ptr_eq(
        &pixels,
        &app.loaded.as_ref().expect("retained CT").data
    ));
}
