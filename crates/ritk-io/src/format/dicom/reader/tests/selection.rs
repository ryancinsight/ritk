//! Synthetic acquisition identity and authoritative file-set regressions.

use super::super::loader::load_dicom_from_series;
use super::super::scan::{
    scan_dicom_files, scan_dicom_part10_bytes, scan_dicom_path, scan_dicom_path_with_budget,
};
use super::super::types::DicomReadBudget;
mod fixtures;
use fixtures::{
    index, instance, set_record_in_use, set_record_next, set_record_sop_instance, OTHER, SERIES,
};
use ritk_dicom::ParseBudget;

#[test]
fn decoded_pixels_are_the_validated_bytes_after_file_replacement() {
    let directory = tempfile::tempdir().expect("directory");
    let path = directory.path().join("selected.dcm");
    let original = instance(Some(SERIES), 1, 43);
    std::fs::write(&path, &original).expect("original instance");
    let snapshots = [
        scan_dicom_files(std::slice::from_ref(&path)).expect("exact members"),
        scan_dicom_path(&path).expect("selected instance"),
    ];

    // Replacing the path is deterministic: it occurs strictly after metadata
    // validation and before pixel decode, without a timing-dependent race.
    std::fs::write(&path, instance(Some(OTHER), 2, 97)).expect("replacement instance");
    let backend = coeus_core::SequentialBackend;
    for snapshot in snapshots {
        assert_eq!(
            snapshot.metadata.slices[0].part10_bytes.as_deref(),
            Some(original.as_slice())
        );
        let (image, metadata) =
            load_dicom_from_series(snapshot, &backend).expect("snapshot decode");
        assert_eq!(metadata.series_instance_uid.as_deref(), Some(SERIES));
        assert_eq!(metadata.origin, [10.0, 20.0, 2.0]);
        assert_eq!(metadata.spacing, [2.0, 0.5, 0.5]);
        assert!(
            metadata
                .slices
                .iter()
                .all(|slice| slice.part10_bytes.is_none()),
            "decoded metadata must not retain the scan payload"
        );
        assert_eq!(image.spacing().to_array(), [2.0, 0.5, 0.5]);
        assert_eq!(image.data_cow_on(&backend).as_ref(), &[43.0; 4]);
    }
}

#[test]
fn selected_file_and_exact_members_preserve_minority_pixels_and_metadata() {
    let directory = tempfile::tempdir().expect("directory");
    let first = directory.path().join("B1.dcm");
    let second = directory.path().join("B2.dcm");
    std::fs::write(&first, instance(Some(SERIES), 1, 31)).expect("first");
    std::fs::write(&second, instance(Some(SERIES), 2, 37)).expect("second");
    for number in 3..6 {
        std::fs::write(
            directory.path().join(format!("A{number}.dcm")),
            instance(Some(OTHER), number, 99),
        )
        .expect("other acquisition");
    }
    let selected = scan_dicom_path(&first).expect("selected acquisition");
    let exact = scan_dicom_files(&[second, first]).expect("exact acquisition");
    let backend = coeus_core::SequentialBackend;
    for scanned in [selected, exact] {
        assert_eq!(
            scanned.metadata.series_instance_uid.as_deref(),
            Some(SERIES)
        );
        assert_eq!(
            scanned.metadata.series_description.as_deref(),
            Some("Acquisition 31")
        );
        assert_eq!(scanned.metadata.origin, [10.0, 20.0, 2.0]);
        assert_eq!(scanned.metadata.spacing, [2.0, 0.5, 0.5]);
        let (image, _) = load_dicom_from_series(scanned, &backend).expect("decode selected series");
        assert_eq!(image.shape(), [2, 2, 2]);
        assert_eq!(
            image.data_cow_on(&backend).as_ref(),
            &[31.0, 31.0, 31.0, 31.0, 37.0, 37.0, 37.0, 37.0]
        );
    }
}

#[test]
fn mixed_byte_batches_reject_both_ties_and_majorities() {
    let first = instance(Some(SERIES), 1, 11);
    let second = instance(Some(SERIES), 2, 13);
    let other = instance(Some(OTHER), 3, 17);
    for files in [
        vec![("first", first.as_slice()), ("other", other.as_slice())],
        vec![
            ("first", first.as_slice()),
            ("second", second.as_slice()),
            ("other", other.as_slice()),
        ],
    ] {
        let error = scan_dicom_part10_bytes(&files).expect_err("explicit identity required");
        assert!(error.to_string().contains("ambiguous"));
    }
}

#[test]
fn missing_invalid_and_malformed_batch_members_never_disappear() {
    let valid = instance(Some(SERIES), 1, 11);
    for invalid in [
        instance(None, 2, 13),
        instance(Some(""), 2, 13),
        instance(Some("2.25.A"), 2, 13),
        instance(Some("2.025.1"), 2, 13),
    ] {
        let error = scan_dicom_part10_bytes(&[("valid", &valid), ("invalid", &invalid)])
            .expect_err("invalid identity");
        assert!(format!("{error:#}").contains("SeriesInstanceUID"));
    }
    let error = scan_dicom_part10_bytes(&[("valid", &valid), ("broken", &[0, 1, 2])])
        .expect_err("malformed member");
    assert!(format!("{error:#}").contains("failed to parse DICOM byte payload"));
}

#[test]
fn named_bytes_and_disk_preserve_identical_values_and_geometry() {
    let directory = tempfile::tempdir().expect("directory");
    let first = instance(Some(SERIES), 1, 11);
    let second = instance(Some(SERIES), 2, 13);
    std::fs::write(directory.path().join("1.dcm"), &first).expect("first");
    std::fs::write(directory.path().join("2.dcm"), &second).expect("second");
    let disk = scan_dicom_path(directory.path()).expect("disk");
    let bytes = scan_dicom_part10_bytes(&[("second", &second), ("first", &first)]).expect("bytes");
    assert_eq!(
        disk.metadata.series_instance_uid,
        bytes.metadata.series_instance_uid
    );
    assert_eq!(disk.metadata.origin, bytes.metadata.origin);
    assert_eq!(disk.metadata.spacing, bytes.metadata.spacing);
    assert_eq!(disk.metadata.direction, bytes.metadata.direction);
    let backend = coeus_core::SequentialBackend;
    let (disk, _) = load_dicom_from_series(disk, &backend).expect("disk decode");
    let (bytes, _) = load_dicom_from_series(bytes, &backend).expect("bytes decode");
    assert_eq!(
        disk.data_cow_on(&backend).as_ref(),
        bytes.data_cow_on(&backend).as_ref()
    );
    assert_eq!(
        disk.data_cow_on(&backend).as_ref(),
        &[11.0, 11.0, 11.0, 11.0, 13.0, 13.0, 13.0, 13.0]
    );
}

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
        image.data_cow_on(&backend).as_ref(),
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
        indexed_image.data_cow_on(&backend).as_ref(),
        flat_image.data_cow_on(&backend).as_ref()
    );
    assert_eq!(
        flat_image.data_cow_on(&backend).as_ref(),
        &[11.0, 11.0, 11.0, 11.0, 13.0, 13.0, 13.0, 13.0, 17.0, 17.0, 17.0, 17.0]
    );
}
