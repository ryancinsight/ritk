//! Series scanning, ordering, and explicit-file selection.

use super::super::*;
use crate::dicom::series_tree::{SeriesEntry, SeriesEntryView};
use std::borrow::Cow;
use tempfile::tempdir;

use super::fixtures;

#[test]
fn sort_series_entries_is_deterministic() {
    let mut entries: Vec<SeriesEntry> = [
        ("UID-B", "z/path", "B", "P2", "MR", "S2", "20260102", "ST2"),
        ("UID-A2", "b/path", "A", "P1", "CT", "S1", "20260101", "ST1"),
        ("UID-A1", "a/path", "A", "P1", "CT", "S1", "20260101", "ST1"),
    ]
    .into_iter()
    .map(
        |(uid, folder, name, patient, modality, description, date, study)| SeriesEntry {
            acquisition: std::sync::Arc::new(ritk_io::DicomSeriesInfo::new(
                uid,
                description.to_owned(),
                modality,
                patient.to_owned(),
                vec![std::path::Path::new(folder).join("slice.dcm")],
            )),
            patient_name: Cow::Borrowed(name),
            study_date: Some(Cow::Borrowed(date)),
            study_uid: Some(Cow::Borrowed(study)),
        },
    )
    .collect();
    scan::sort_series_entries_deterministically(&mut entries);
    let ordered_uids: Vec<&str> = entries.iter().map(|e| e.series_uid()).collect();
    assert_eq!(ordered_uids, vec!["UID-A1", "UID-A2", "UID-B"]);
}

#[test]
fn test_scan_folder_for_series_empty_dir() {
    let dir = tempdir().expect("create empty study directory");
    let tree = scan_folder_for_series(dir.path()).expect("scan empty directory");
    assert_eq!(tree.total_series(), 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_dicom_series_from_stored_instances_empty_input_errors() {
    let error =
        load_dicom_series_from_stored_instances(&[]).expect_err("empty SCP batch must reject");
    assert_eq!(error.to_string(), "no SCP-received DICOM instances to load");
}

#[test]
fn explicit_file_selects_minority_series_while_mixed_batches_reject() {
    let root = tempdir().expect("mixed study root");
    let primary = fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID)
        .expect("write primary acquisition");
    let secondary_uid = "2.25.20260905002";
    let mut secondary = fixtures::write_study(root.path(), "MR", secondary_uid)
        .expect("write secondary acquisition");

    for expected_counts in [[3, 3], [3, 2]] {
        let named: Vec<(String, &[u8])> = primary
            .iter()
            .chain(&secondary)
            .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
            .collect();
        let directory_error = load_dicom_volume(root.path())
            .expect_err("mixed folder must require explicit acquisition selection");
        let bytes_error = load_dicom_series_from_named_bytes(&named)
            .expect_err("mixed byte batch must require explicit acquisition selection");
        for error in [directory_error, bytes_error] {
            let diagnostic = format!("{error:#}");
            assert!(
                diagnostic.contains("SeriesInstanceUID"),
                "ambiguity must identify the selection dimension: {diagnostic}"
            );
        }
        assert_eq!([primary.len(), secondary.len()], expected_counts);
        if secondary.len() == 3 {
            let (removed, _) = secondary.remove(0);
            std::fs::remove_file(root.path().join(removed))
                .expect("construct two-slice minority acquisition");
        }
    }

    let selected = root
        .path()
        .join(&secondary.first().expect("minority slice").0);
    let volume =
        load_volume_from_path(&selected).expect("open explicitly selected minority acquisition");
    assert_eq!(volume.shape, [2, 2, 4]);
    let expected: Vec<_> = fixtures::SAMPLES[..16]
        .iter()
        .map(|&raw| 2.0 * f32::from(raw) - 20.0)
        .collect();
    assert_eq!(volume.data.as_slice(), expected);
    assert_eq!(
        volume
            .metadata
            .as_ref()
            .expect("metadata retained")
            .series_instance_uid
            .as_deref(),
        Some(secondary_uid)
    );
    assert_eq!(volume.modality.as_deref(), Some("MR"));
    assert_eq!(volume.source.as_deref(), Some(selected.as_path()));
}
