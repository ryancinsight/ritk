//! Series scanning, ordering, and explicit-file selection.

use super::super::*;
use tempfile::tempdir;

use super::fixtures;

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
    let primary =
        fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
            .expect("write primary acquisition");
    let secondary_uid = "2.25.20260905002";
    let mut secondary = fixtures::write_study(root.path(), "MR", secondary_uid, &fixtures::SAMPLES)
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
    let expected = fixtures::displayed_samples(&fixtures::SAMPLES[..16]);
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

#[test]
fn series_uid_selection_loads_the_requested_acquisition_from_a_mixed_directory() {
    let root = tempdir().expect("mixed study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
        .expect("write primary acquisition");
    let secondary_uid = "2.25.20260905002";
    fixtures::write_study(root.path(), "MR", secondary_uid, &fixtures::SAMPLES)
        .expect("write secondary acquisition");

    let volume = load_volume_from_series_uid(root.path(), secondary_uid)
        .expect("explicit series UID must select one acquisition");

    assert_eq!(volume.shape, [3, 2, 4]);
    assert_eq!(volume.modality.as_deref(), Some("MR"));
    assert_eq!(
        volume
            .metadata
            .as_ref()
            .expect("metadata retained")
            .series_instance_uid
            .as_deref(),
        Some(secondary_uid)
    );
    assert_eq!(
        volume.source.as_deref().and_then(std::path::Path::parent),
        Some(root.path())
    );
}

#[test]
fn series_uid_selection_rejects_an_unknown_acquisition_without_fallback() {
    let root = tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
        .expect("write acquisition");

    let error = load_volume_from_series_uid(root.path(), "2.25.404")
        .expect_err("unknown series UID must fail closed");
    let message = format!("{error:#}");
    assert!(
        message.contains("SeriesInstanceUID"),
        "diagnostic: {message}"
    );
    assert!(message.contains("2.25.404"), "diagnostic: {message}");
}

#[test]
fn series_uid_selection_rejects_invalid_uid_syntax_before_discovery() {
    let root = tempdir().expect("study root");
    let error = load_volume_from_series_uid(root.path(), "1..02")
        .expect_err("invalid UID syntax must fail before scanning");
    assert_eq!(error.to_string(), "invalid SeriesInstanceUID selection");
}
