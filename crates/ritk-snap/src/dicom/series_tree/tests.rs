use super::*;
use std::path::{Path, PathBuf};

/// Build a synthetic `SeriesEntry` for testing purposes.
fn make_entry<'a>(
    patient_id: &'a str,
    patient_name: &'a str,
    study_uid: Option<&'a str>,
    study_date: Option<&'a str>,
    series_uid: &'a str,
    folder: &'a str,
    modality: &'a str,
    num_slices: usize,
) -> SeriesEntry<'a> {
    SeriesEntry {
        series_uid: Cow::Borrowed(series_uid),
        folder: Cow::Borrowed(Path::new(folder)),
        patient_name: Cow::Borrowed(patient_name),
        patient_id: Cow::Borrowed(patient_id),
        modality: Cow::Borrowed(modality),
        series_description: Cow::Owned(format!("{modality} series")),
        num_slices,
        study_date: study_date.map(Cow::Borrowed),
        study_uid: study_uid.map(Cow::Borrowed) }
}

/// Three series across two patients must produce exactly two patient nodes.
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

/// `from_entries` with an empty input must produce an empty tree.
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
    let info = DicomSeriesInfo::new(
        "1.2.3",
        "Axial CT".to_string(),
        "CT",
        "P001".to_string(),
        vec![
            PathBuf::from("C:/study/series/slice_0001.dcm"),
            PathBuf::from("C:/study/series/slice_0002.dcm"),
        ],
    );
    let entry = SeriesEntry::from_dicom_series_info(info);
    assert_eq!(entry.series_uid, "1.2.3");
    assert_eq!(entry.folder.as_ref(), Path::new("C:/study/series"));
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

// â”€â”€ New Optimization & Architecture Verification Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Test that GAT-based `SeriesEntryView` works for both `SeriesEntry` and `SeriesNode`.
#[test]
fn test_gat_series_entry_view() {
    let entry = make_entry("P1", "Alice", None, None, "S1", "/s1", "CT", 10);
    let node = SeriesNode {
        series_uid: Cow::Borrowed("S1"),
        folder: Cow::Borrowed(Path::new("/s1")),
        modality: Cow::Borrowed("CT"),
        series_description: Cow::Borrowed("CT series"),
        num_slices: 10 };

    fn check_gat<V: SeriesEntryView>(view: &V) {
        assert_eq!(view.series_uid().as_ref(), "S1");
        assert_eq!(view.folder().as_ref(), Path::new("/s1"));
        assert_eq!(view.modality().as_ref(), "CT");
        assert_eq!(view.num_slices(), 10);
    }

    check_gat(&entry);
    check_gat(&node);
}

/// Test that `ModalityMapper` using const generics correctly matches icons.
#[test]
fn test_const_generic_modality_mapper() {
    let custom_icons: [(&str, &str); 2] = [("CT", "â˜¢CT"), ("MR", "â˜¢MR")];
    let mapper = ModalityMapper::new(custom_icons);

    assert_eq!(mapper.get_icon("CT"), "â˜¢CT");
    assert_eq!(mapper.get_icon("MR"), "â˜¢MR");
    assert_eq!(mapper.get_icon("US"), "ðŸ—‚"); // Default fallback
}

/// Test that monomorphized `format_series_label` produces the expected format.
#[test]
fn test_monomorphized_format_series_label() {
    let entry = make_entry(
        "P1",
        "Alice",
        None,
        None,
        "S1",
        "/path/to/series",
        "CT",
        123,
    );
    let label = format_series_label(&entry, &DEFAULT_MODALITY_MAPPER);
    assert_eq!(label, "ðŸ« [CT] CT series (123 slices)");

    // Test fallback when description is empty
    let empty_desc_entry = SeriesEntry {
        series_uid: Cow::Borrowed("S1"),
        folder: Cow::Borrowed(Path::new("/path/to/my_folder")),
        patient_name: Cow::Borrowed(""),
        patient_id: Cow::Borrowed(""),
        modality: Cow::Borrowed("MR"),
        series_description: Cow::Borrowed(""),
        num_slices: 15,
        study_date: None,
        study_uid: None };
    let label2 = format_series_label(&empty_desc_entry, &DEFAULT_MODALITY_MAPPER);
    assert_eq!(label2, "ðŸ§  [MR] my_folder (15 slices)");
}

/// Benchmark test to empirically validate the O(N) linear time construction complexity.
#[test]
fn test_bench_tree_construction() {
    use std::time::Instant;

    let mut entries = Vec::new();
    // Generate 5000 entries across 50 patients and 250 studies.
    for i in 0..5000 {
        let patient_id = format!("P{}", i % 50);
        let study_uid = format!("ST{}", i % 250);
        let series_uid = format!("S{}", i);
        let folder = format!("/path/to/patient_{}/study_{}/series_{}", i % 50, i % 250, i);
        entries.push(SeriesEntry {
            series_uid: Cow::Owned(series_uid),
            folder: Cow::Owned(PathBuf::from(folder)),
            patient_name: Cow::Owned(format!("Patient {}", i % 50)),
            patient_id: Cow::Owned(patient_id),
            modality: Cow::Borrowed("CT"),
            series_description: Cow::Borrowed("Bench CT"),
            num_slices: 100,
            study_date: Some(Cow::Borrowed("20260615")),
            study_uid: Some(Cow::Owned(study_uid)) });
    }

    let start = Instant::now();
    let tree = SeriesTree::from_entries(entries);
    let duration = start.elapsed();
    println!(
        "\n[BENCHMARK] Built SeriesTree of 5000 entries in {:?}",
        duration
    );
    assert_eq!(tree.total_series(), 5000);
}
