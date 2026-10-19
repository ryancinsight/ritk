//! DICOMDIR index authority and directory-record membership regressions.

use super::super::super::loader::load_dicom_from_series;
use super::super::super::scan::{scan_dicom_path, scan_dicom_path_with_budget};
use super::super::super::DicomReadBudget;
use super::fixtures::{
    self, index, instance, set_record_in_use, set_record_next, set_record_sop_instance, OTHER,
    SERIES,
};
use ritk_dicom::ParseBudget;

#[test]
fn lowercase_index_is_authoritative_for_discovery_and_loading() {
    let directory = tempfile::tempdir().expect("directory");
    std::fs::create_dir(directory.path().join("IMAGES")).expect("images");
    std::fs::write(
        directory.path().join("IMAGES").join("ONE"),
        instance(Some(SERIES), 1, 41),
    )
    .expect("image");
    std::fs::write(
        directory.path().join("unrelated.dcm"),
        instance(Some(OTHER), 2, 99),
    )
    .expect("unrelated");
    let selected = directory.path().join("dicomdir");
    index(&selected, &["IMAGES\\ONE"]);
    for input in [directory.path(), selected.as_path()] {
        let series = scan_dicom_path(input).expect("index scan");
        assert_eq!(series.num_slices, 1);
        assert_eq!(series.metadata.series_instance_uid.as_deref(), Some(SERIES));
        let discovered = crate::scan_dicom_directory(input).expect("index discovery");
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].series_instance_uid(), SERIES);
        assert_eq!(
            discovered[0].file_paths,
            vec![directory
                .path()
                .join("IMAGES")
                .join("ONE")
                .canonicalize()
                .expect("reference")]
        );
    }
}

#[test]
fn invalid_index_references_never_fall_back_to_neighbors() {
    let directory = tempfile::tempdir().expect("directory");
    std::fs::write(
        directory.path().join("valid.dcm"),
        instance(Some(SERIES), 1, 41),
    )
    .expect("valid");
    let selected = directory.path().join("DICOMDIR");
    for reference in [
        "missing",
        "..\\outside.dcm",
        "/outside.dcm",
        "C:\\outside.dcm",
        "IMAGES/ONE",
        "",
    ] {
        index(&selected, &[reference]);
        for input in [directory.path(), selected.as_path()] {
            let error = scan_dicom_path(input).expect_err("invalid index reference");
            assert!(format!("{error:#}").contains("DICOMDIR"));
            let error = crate::scan_dicom_directory(input).expect_err("discovery must also reject");
            assert!(format!("{error:#}").contains("DICOMDIR"));
        }
    }
    std::fs::write(&selected, [0, 1, 2]).expect("corrupt index");
    assert!(format!(
        "{:#}",
        scan_dicom_path(&selected).expect_err("corrupt index")
    )
    .contains("DICOMDIR"));
}

#[test]
fn indexed_scan_applies_budget_to_dicomdir_before_materialization() {
    let directory = fixtures::indexed_study();
    let index = directory.path().join("DICOMDIR");
    let index_length = std::fs::metadata(&index).expect("index metadata").len();
    let budget = DicomReadBudget::try_new(
        ParseBudget::new(
            usize::try_from(index_length)
                .expect("fixture index length fits usize")
                .saturating_sub(1),
            100,
            100,
        ),
        usize::try_from(index_length).expect("fixture index length fits usize"),
        100,
    )
    .expect("workflow ceilings must be nonzero");

    let error = scan_dicom_path_with_budget(&index, &budget)
        .expect_err("DICOMDIR must be rejected before parser materialization");
    assert!(
        format!("{error:#}").contains("DICOM file exceeds parse budget"),
        "unexpected DICOMDIR budget error: {error:#}"
    );
}

#[test]
fn malformed_selected_file_cannot_open_valid_neighbors() {
    let directory = tempfile::tempdir().expect("directory");
    std::fs::write(
        directory.path().join("valid.dcm"),
        instance(Some(SERIES), 1, 41),
    )
    .expect("valid");
    let selected = directory.path().join("broken.dcm");
    std::fs::write(&selected, [0, 1, 2]).expect("corrupt file");
    let error = scan_dicom_path(&selected).expect_err("selected file must parse");
    assert!(format!("{error:#}").contains("selected DICOM instance"));
}

#[cfg(unix)]
#[test]
fn index_rejects_symlink_outside_root() {
    let directory = tempfile::tempdir().expect("directory");
    let outside = tempfile::NamedTempFile::new().expect("outside");
    std::os::unix::fs::symlink(outside.path(), directory.path().join("LINK")).expect("symlink");
    let selected = directory.path().join("DICOMDIR");
    index(&selected, &["LINK"]);
    let error = scan_dicom_path(selected).expect_err("outside-root symlink");
    assert!(error.to_string().contains("escapes"));
}

#[cfg(unix)]
#[test]
fn index_rejects_final_symlink_inside_root() {
    let directory = fixtures::indexed_study();
    let target = directory.path().join("IMAGES").join("ONE");
    std::os::unix::fs::symlink(&target, directory.path().join("LINK")).expect("symlink");
    let selected = directory.path().join("DICOMDIR");
    index(&selected, &["LINK"]);

    let error = scan_dicom_path(selected).expect_err("final symlink");
    assert!(format!("{error:#}").contains("DICOMDIR referenced file"));
}

#[test]
fn indexed_folder_loads_complete_known_series() {
    let directory = fixtures::indexed_study();
    let series = scan_dicom_path(directory.path()).expect("indexed folder");
    assert_eq!(series.num_slices, 3);
    assert_eq!(series.metadata.series_instance_uid.as_deref(), Some(SERIES));
    let backend = coeus_core::SequentialBackend;
    let (image, metadata) = load_dicom_from_series(series, &backend).expect("indexed decode");
    assert_eq!(image.shape(), [3, 2, 2]);
    assert_eq!(metadata.origin, [10.0, 20.0, 2.0]);
    assert_eq!(metadata.spacing, [2.0, 0.5, 0.5]);
    assert_eq!(
        image
            .data_cow_on(&backend)
            .expect("test tensor is valid")
            .as_ref(),
        &[11.0, 11.0, 11.0, 11.0, 13.0, 13.0, 13.0, 13.0, 17.0, 17.0, 17.0, 17.0]
    );
}

#[test]
fn inactive_image_records_are_excluded_from_authoritative_membership() {
    let directory = fixtures::indexed_study();
    let index = directory.path().join("DICOMDIR");
    set_record_in_use(&index, 3, 0);

    let series = scan_dicom_path(&index).expect("active image records");
    assert_eq!(series.num_slices, 2);
    assert_eq!(
        series
            .metadata
            .slices
            .iter()
            .filter_map(|slice| slice.instance_number)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn unreachable_image_records_do_not_enter_authoritative_membership() {
    let directory = fixtures::indexed_study();
    let index = directory.path().join("DICOMDIR");
    let bytes = std::fs::read(&index).expect("read linked index");
    let item_offsets: Vec<_> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, tag)| *tag == [0xfe, 0xff, 0x00, 0xe0])
        .map(|(offset, _)| u32::try_from(offset).expect("fixture offset"))
        .collect();
    assert_eq!(item_offsets.len(), 6);
    set_record_next(&index, 4, 0);
    set_record_next(&index, 3, item_offsets[5]);

    let series = scan_dicom_path(&index).expect("reachable image records");
    assert_eq!(series.num_slices, 2);
    assert_eq!(
        series
            .metadata
            .slices
            .iter()
            .filter_map(|slice| slice.instance_number)
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
}

#[test]
fn directory_record_cycles_are_rejected_before_membership() {
    let directory = fixtures::indexed_study();
    let index = directory.path().join("DICOMDIR");
    let bytes = std::fs::read(&index).expect("read linked index");
    let item_offsets: Vec<_> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, tag)| *tag == [0xfe, 0xff, 0x00, 0xe0])
        .map(|(offset, _)| u32::try_from(offset).expect("fixture offset"))
        .collect();
    set_record_next(&index, 5, item_offsets[3]);

    let error = scan_dicom_path(&index).expect_err("record cycle");
    let message = format!("{error:#}");
    assert!(message.contains("cycle") || message.contains("multiple incoming"));
}

#[test]
fn directory_record_offsets_must_stay_inside_the_sequence() {
    let directory = fixtures::indexed_study();
    let index = directory.path().join("DICOMDIR");
    set_record_next(&index, 3, u32::MAX - 1);

    let error = scan_dicom_path(index).expect_err("out-of-sequence directory link");
    assert!(format!("{error:#}").contains("outside DirectoryRecordSequence"));
}

#[test]
fn directory_record_identity_mismatch_is_rejected() {
    let directory = fixtures::indexed_study();
    let index = directory.path().join("DICOMDIR");
    set_record_sop_instance(&index, 3, "2.25.74001.2");

    let error = scan_dicom_path(&index).expect_err("mismatched SOP instance");
    assert!(format!("{error:#}").contains("ReferencedSOPInstanceUID"));
}

#[test]
fn explicit_index_and_flat_members_agree_on_geometry_and_values() {
    let directory = fixtures::indexed_study();
    let index = scan_dicom_path(directory.path().join("DICOMDIR")).expect("explicit index");
    let flat = scan_dicom_path(directory.path().join("IMAGES")).expect("flat members");
    assert_eq!(index.metadata.dimensions, flat.metadata.dimensions);
    assert_eq!(index.metadata.spacing, flat.metadata.spacing);
    assert_eq!(index.metadata.origin, flat.metadata.origin);
    assert_eq!(index.metadata.direction, flat.metadata.direction);
    assert_eq!(
        index.metadata.direction,
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    );
    let backend = coeus_core::SequentialBackend;
    let (indexed_image, _) = load_dicom_from_series(index, &backend).expect("indexed image");
    let (flat_image, _) = load_dicom_from_series(flat, &backend).expect("flat image");
    assert_eq!(
        indexed_image
            .data_cow_on(&backend)
            .expect("test tensor is valid")
            .as_ref(),
        flat_image
            .data_cow_on(&backend)
            .expect("test tensor is valid")
            .as_ref()
    );
    assert_eq!(
        flat_image
            .data_cow_on(&backend)
            .expect("test tensor is valid")
            .as_ref(),
        &[11.0, 11.0, 11.0, 11.0, 13.0, 13.0, 13.0, 13.0, 17.0, 17.0, 17.0, 17.0]
    );
}
