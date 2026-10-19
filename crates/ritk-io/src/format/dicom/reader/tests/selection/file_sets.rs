//! File-set selection and batch identity regressions.

use super::super::super::loader::load_dicom_from_series;
use super::super::super::scan::{scan_dicom_files, scan_dicom_part10_bytes, scan_dicom_path};
use super::fixtures::{instance, OTHER, SERIES};
use crate::DicomStudyCatalog;

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
        assert_eq!(
            image
                .data_cow_on(&backend)
                .expect("test tensor is valid")
                .as_ref(),
            &[43.0; 4]
        );
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
            image
                .data_cow_on(&backend)
                .expect("test tensor is valid")
                .as_ref(),
            &[31.0, 31.0, 31.0, 31.0, 37.0, 37.0, 37.0, 37.0]
        );
    }
}

#[test]
fn study_catalog_exposes_and_loads_each_series_independently() {
    let directory = tempfile::tempdir().expect("directory");
    for (name, uid, number, value) in [
        ("first.dcm", SERIES, 1, 31),
        ("second.dcm", SERIES, 2, 37),
        ("other.dcm", OTHER, 3, 91),
    ] {
        std::fs::write(
            directory.path().join(name),
            instance(Some(uid), number, value),
        )
        .expect("catalog fixture instance");
    }

    let catalog = DicomStudyCatalog::scan(directory.path()).expect("study catalog");
    assert_eq!(catalog.series().len(), 2);
    let first = catalog.select(SERIES).expect("first series");
    assert_eq!(first.instance_count(), 2);
    assert_eq!(first.modality(), "CT");
    assert_eq!(first.series_description, "Acquisition 31");
    let other = catalog.select(OTHER).expect("other series");
    assert_eq!(other.instance_count(), 1);
    assert_eq!(other.series_description, "Acquisition 91");
    let missing = catalog
        .select("2.25.73999")
        .expect_err("unknown series UID must reject");
    assert!(missing.to_string().contains("2.25.73999"));

    let backend = coeus_core::SequentialBackend;
    let (image, metadata) = catalog.load(OTHER, &backend).expect("selected load");
    assert_eq!(metadata.series_instance_uid.as_deref(), Some(OTHER));
    assert_eq!(
        metadata.series_description.as_deref(),
        Some("Acquisition 91")
    );
    assert_eq!(image.shape(), [1, 2, 2]);
    assert_eq!(image.origin().to_array(), [10.0, 20.0, 6.0]);
    assert_eq!(image.spacing().to_array(), [2.0, 0.5, 0.5]);
    assert_eq!(
        metadata.direction,
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        image
            .data_cow_on(&backend)
            .expect("test tensor is valid")
            .as_ref(),
        &[91.0; 4]
    );
}

#[test]
fn study_catalog_rejects_selected_member_replaced_by_another_series() {
    let directory = tempfile::tempdir().expect("directory");
    let path = directory.path().join("selected.dcm");
    std::fs::write(&path, instance(Some(OTHER), 1, 91)).expect("catalog instance");
    let catalog = DicomStudyCatalog::scan(directory.path()).expect("study catalog");

    std::fs::write(&path, instance(Some(SERIES), 2, 31)).expect("replacement instance");

    let error = catalog
        .load(OTHER, &coeus_core::SequentialBackend)
        .expect_err("replacement SeriesInstanceUID must reject");
    assert!(
        format!("{error:#}").contains("selected DICOM series changed from"),
        "unexpected replacement error: {error:#}"
    );
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
        disk.data_cow_on(&backend)
            .expect("test tensor is valid")
            .as_ref(),
        bytes
            .data_cow_on(&backend)
            .expect("test tensor is valid")
            .as_ref()
    );
    assert_eq!(
        disk.data_cow_on(&backend)
            .expect("test tensor is valid")
            .as_ref(),
        &[11.0, 11.0, 11.0, 11.0, 13.0, 13.0, 13.0, 13.0]
    );
}
