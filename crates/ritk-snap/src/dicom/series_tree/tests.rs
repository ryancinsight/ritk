use super::*;
use std::path::PathBuf;

/// Build a synthetic `SeriesEntry` for testing purposes.
fn make_entry(
    patient_id: &str,
    patient_name: &str,
    study_uid: Option<&str>,
    study_date: Option<&str>,
    series_uid: &str,
    folder: &str,
    modality: &str,
    num_slices: usize,
) -> SeriesEntry {
    SeriesEntry {
        series_uid: series_uid.to_string(),
        folder: PathBuf::from(folder),
        patient_name: patient_name.to_string(),
        patient_id: patient_id.to_string(),
        modality: modality.to_string(),
        series_description: format!("{modality} series"),
        num_slices,
        study_date: study_date.map(str::to_string),
        study_uid: study_uid.map(str::to_string),
    }
}

/// Three series across two patients must produce exactly two patient nodes.
///
/// Patient A has two series in the same study; patient B has one series.
/// Postcondition: `tree.patients.len() == 2`, `total_series() == 3`.
#[test]
fn test_from_entries_groups_by_patient() {
    let entries = vec![
        make_entry(
            "P001",
            "Alice",
            Some("ST1"),
            Some("20230101"),
            "S1",
            "/a/s1",
            "CT",
            50,
        ),
        make_entry(
            "P001",
            "Alice",
            Some("ST1"),
            Some("20230101"),
            "S2",
            "/a/s2",
            "CT",
            30,
        ),
        make_entry(
            "P002",
            "Bob",
            Some("ST2"),
            Some("20230202"),
            "S3",
            "/b/s1",
            "MR",
            20,
        ),
    ];
    let tree = SeriesTree::from_entries(entries);

    assert_eq!(
        tree.patients.len(),
        2,
        "two distinct patient IDs must produce two PatientNodes"
    );
    assert_eq!(
        tree.total_series(),
        3,
        "total_series() must equal the number of input entries"
    );

    // Patient A must have exactly one study with two series.
    let alice = tree
        .patients
        .iter()
        .find(|p| p.patient_id == "P001")
        .unwrap();
    assert_eq!(alice.studies.len(), 1, "Alice must have one study");
    assert_eq!(
        alice.studies[0].series.len(),
        2,
        "Alice's study must contain both CT series"
    );

    // Patient B must have exactly one study with one series.
    let bob = tree
        .patients
        .iter()
        .find(|p| p.patient_id == "P002")
        .unwrap();
    assert_eq!(bob.studies.len(), 1, "Bob must have one study");
    assert_eq!(
        bob.studies[0].series.len(),
        1,
        "Bob's study must contain exactly one MR series"
    );
}

/// `total_series()` must return the exact number of entries inserted.
///
/// Tested with five entries spanning three patients to exercise the
/// summation path across non-trivial tree depth.
#[test]
fn test_total_series_count() {
    let entries = vec![
        make_entry("P1", "Alice", Some("ST1"), None, "S1", "/p1/s1", "CT", 10),
        make_entry("P1", "Alice", Some("ST1"), None, "S2", "/p1/s2", "CT", 20),
        make_entry("P2", "Bob", Some("ST2"), None, "S3", "/p2/s1", "MR", 15),
        make_entry("P3", "Carol", Some("ST3"), None, "S4", "/p3/s1", "PT", 60),
        make_entry("P3", "Carol", Some("ST3"), None, "S5", "/p3/s2", "PT", 60),
    ];
    let tree = SeriesTree::from_entries(entries);
    assert_eq!(
        tree.total_series(),
        5,
        "total_series() must equal 5 for five distinct entries"
    );
}

/// `from_entries` with an empty input must produce an empty tree with
/// zero patients and `total_series() == 0`.
#[test]
fn test_from_entries_empty_input() {
    let tree = SeriesTree::from_entries(vec![]);
    assert_eq!(
        tree.patients.len(),
        0,
        "empty input must produce zero patient nodes"
    );
    assert_eq!(
        tree.total_series(),
        0,
        "total_series() must be 0 for empty input"
    );
}

/// `find_by_folder` must locate an entry by its exact folder path.
#[test]
fn test_find_by_folder_found() {
    let entries = vec![
        make_entry(
            "P1",
            "Alice",
            Some("ST1"),
            None,
            "S1",
            "/data/scan1",
            "CT",
            50,
        ),
        make_entry(
            "P1",
            "Alice",
            Some("ST1"),
            None,
            "S2",
            "/data/scan2",
            "MR",
            30,
        ),
    ];
    let tree = SeriesTree::from_entries(entries);
    let found = tree.find_by_folder(Path::new("/data/scan2"));
    assert!(found.is_some(), "find_by_folder must find '/data/scan2'");
    assert_eq!(
        found.unwrap().series_uid,
        "S2",
        "found entry must be the MR series with uid S2"
    );
}

/// `find_by_folder` must return `None` for a path not in the tree.
#[test]
fn test_find_by_folder_not_found() {
    let tree = SeriesTree::from_entries(vec![make_entry(
        "P1",
        "Alice",
        Some("ST1"),
        None,
        "S1",
        "/data/s1",
        "CT",
        10,
    )]);
    assert!(
        tree.find_by_folder(Path::new("/data/nonexistent"))
            .is_none(),
        "find_by_folder must return None for an absent path"
    );
}

/// `display_label()` must be non-empty and contain the slice count.
#[test]
fn test_series_entry_display_label_contains_slice_count() {
    let entry = make_entry("P1", "Alice", None, None, "S1", "/s1", "CT", 42);
    let label = entry.display_label();
    assert!(!label.is_empty(), "display_label() must not be empty");
    assert!(
        label.contains("42"),
        "display_label() must contain the slice count '42'; got: {label}"
    );
}

#[test]
fn test_series_entry_from_dicom_series_info_uses_file_parent_and_slice_count() {
    let info = DicomSeriesInfo {
        series_instance_uid: "1.2.3".to_string(),
        series_description: "Axial CT".to_string(),
        modality: "CT".to_string(),
        patient_id: "P001".to_string(),
        file_paths: vec![
            PathBuf::from("C:/study/series/slice_0001.dcm"),
            PathBuf::from("C:/study/series/slice_0002.dcm"),
        ],
    };
    let entry = SeriesEntry::from_dicom_series_info(info);
    assert_eq!(entry.series_uid, "1.2.3");
    assert_eq!(entry.folder, PathBuf::from("C:/study/series"));
    assert_eq!(entry.patient_id, "P001");
    assert_eq!(entry.modality, "CT");
    assert_eq!(entry.series_description, "Axial CT");
    assert_eq!(entry.num_slices, 2);
}

/// `modality_icon()` must return a non-empty string for every supported
/// modality and for unknown modalities.
#[test]
fn test_modality_icon_non_empty() {
    let modalities = [
        "CT", "MR", "PT", "NM", "US", "CR", "DR", "DX", "MG", "XA", "RF", "OT", "",
    ];
    for m in modalities {
        let entry = make_entry("P1", "X", None, None, "S1", "/s1", m, 1);
        let icon = entry.modality_icon();
        assert!(
            !icon.is_empty(),
            "modality_icon() must not be empty for modality '{m}'"
        );
    }
}

/// Two series with the same patient_id but different study_uids must
/// produce two distinct StudyNodes under one PatientNode.
#[test]
fn test_from_entries_splits_different_studies() {
    let entries = vec![
        make_entry("P1", "Alice", Some("STUDY-A"), None, "S1", "/s1", "CT", 10),
        make_entry("P1", "Alice", Some("STUDY-B"), None, "S2", "/s2", "MR", 20),
    ];
    let tree = SeriesTree::from_entries(entries);

    assert_eq!(
        tree.patients.len(),
        1,
        "same patient_id must produce one PatientNode"
    );
    let alice = &tree.patients[0];
    assert_eq!(
        alice.studies.len(),
        2,
        "two distinct study_uids must produce two StudyNodes"
    );
    assert_eq!(tree.total_series(), 2);
}
