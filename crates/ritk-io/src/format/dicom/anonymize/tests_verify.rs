//! Tests for the export-time DICOM metadata verification gate.
//!
//! These tests exercise `verify::*` against in-memory objects written to
//! temporary files, covering the standard PACS anonymization export failure
//! modes: corrupt writes, diverged series UIDs, leaked identifiers, and
//! geometry/pixel drift.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::super::anonymize::verify::{
    ensure_dicom_file_clean, verify_dicom_directory, verify_dicom_file, verify_dicom_instances,
    VerifyIssue, VerifyOptions,
};
use ritk_dicom::dataset::{DataElement, DicomTag, PrimitiveValue, VR};
use ritk_dicom::dataset::{Dataset, DicomFile, FileMetaBuilder};

pub(super) fn build_object(study_uid: &str, series_uid: &str, sop_uid: &str) -> DicomFile {
    let mut obj = Dataset::new();
    // Identification / UIDs
    obj.put(DataElement::new(
        DicomTag::new(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from(study_uid),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from(series_uid),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_uid),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0020, 0x0052),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.1.2.3.4.5.6"),
    ));
    // Geometry: 2x2x1, 16-bit
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    // PixelData: 2*2*2 = 8 bytes (16-bit).
    obj.put(DataElement::new(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(ritk_dicom::dataset::SmallVec::from_vec(vec![0, 1, 2, 3])),
    ));
    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid(sop_uid)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("synthetic metadata valid")
}

pub(super) fn write_to_temp(obj: &DicomFile, name: &str) -> std::path::PathBuf {
    // Leak the tempdir guard so the file survives for the duration of the test;
    // the OS temp dir is cleaned up eventually. Tests never accumulate enough
    // to matter.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(name);
    obj.write_to_file(&path).expect("write test object");
    let path = path.clone();
    std::mem::forget(dir);
    path
}

#[test]
fn clean_object_passes_export_gate() {
    let obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    let path = write_to_temp(&obj, "clean.dcm");
    let report = verify_dicom_file(&path, &VerifyOptions::default()).unwrap();
    assert!(
        report.is_clean(),
        "clean object must pass: {:?}",
        report.issues
    );
}

#[test]
fn truncated_file_reports_parse_failure() {
    let obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("truncated.dcm");
    obj.write_to_file(&path).expect("write");
    // Truncate the file to 12 bytes: header prefix survives, payload does not.
    let bytes = std::fs::read(&path).expect("read");
    std::fs::write(&path, &bytes[..bytes.len() / 2]).expect("truncate");

    let report = verify_dicom_file(&path, &VerifyOptions::default()).unwrap();
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, VerifyIssue::ParseFailure(_))),
        "truncated file must report parse failure, got {:?}",
        report.issues
    );
}

#[test]
fn missing_uid_reported_when_required() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.remove_element(DicomTag::new(0x0020, 0x000D)); // StudyInstanceUID
    let path = write_to_temp(&obj, "missing_study.dcm");

    let report = verify_dicom_file(&path, &VerifyOptions::default()).unwrap();
    assert!(report
        .issues
        .iter()
        .any(|i| matches!(i, VerifyIssue::MissingUid { tag } if tag == "StudyInstanceUID")));
}

#[test]
fn invalid_uid_format_reported() {
    // Leading-zero component is forbidden by PS 3.5.
    let obj = build_object("1.2.03.4", "1.2.3.4.5.7", "1.2.3.4.5.8");
    let path = write_to_temp(&obj, "invalid_uid.dcm");

    let report = verify_dicom_file(&path, &VerifyOptions::default()).unwrap();
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, VerifyIssue::InvalidUid { .. })),
        "leading-zero UID component must be reported, got {:?}",
        report.issues
    );
}

#[test]
fn prohibited_value_leak_reported() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.put(DataElement::new(
        DicomTag::new(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Doe^John"),
    ));
    let path = write_to_temp(&obj, "leak.dcm");

    let opts = VerifyOptions {
        prohibited_values: vec!["Doe^John".to_owned()],
        ..Default::default()
    };
    let report = verify_dicom_file(&path, &opts).unwrap();
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, VerifyIssue::PatientLeak { value } if value == "Doe^John")),
        "prohibited value must be reported, got {:?}",
        report.issues
    );
}

#[test]
fn directory_cross_file_uid_divergence_reported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    let b = build_object("1.2.3.4.5.6", "1.2.3.4.5.99", "1.2.3.4.5.9"); // different series UID
    a.write_to_file(dir.path().join("a.dcm")).expect("write a");
    b.write_to_file(dir.path().join("b.dcm")).expect("write b");

    let report = verify_dicom_directory(dir.path(), &VerifyOptions::default()).unwrap();
    assert_eq!(report.file_count, 2);
    assert!(
        report.files.iter().any(|f| f.issues.iter().any(
            |i| matches!(i, VerifyIssue::UidMismatch { tag, .. } if tag == "SeriesInstanceUID")
        )),
        "diverged series UIDs must be reported"
    );
}

#[test]
fn directory_duplicate_sop_uid_reported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    let b = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8"); // same SOP UID
    a.write_to_file(dir.path().join("a.dcm")).expect("write a");
    b.write_to_file(dir.path().join("b.dcm")).expect("write b");

    let report = verify_dicom_directory(dir.path(), &VerifyOptions::default()).unwrap();
    assert!(
        report.files.iter().any(|f| f
            .issues
            .iter()
            .any(|i| matches!(i, VerifyIssue::DuplicateSopInstanceUid { .. }))),
        "duplicate SOPInstanceUID must be reported"
    );
}

#[test]
fn clean_directory_passes_gate() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (i, sop) in ["1.2.3.4.5.8", "1.2.3.4.5.9", "1.2.3.4.5.10"]
        .iter()
        .enumerate()
    {
        let obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", sop);
        obj.write_to_file(dir.path().join(format!("s{i}.dcm")))
            .expect("write");
    }
    let report = verify_dicom_directory(dir.path(), &VerifyOptions::default()).unwrap();
    assert_eq!(report.file_count, 3);
    assert_eq!(report.clean_count, 3);
    assert!(
        report.is_clean(),
        "clean directory must pass: {:?}",
        report.files
    );
}

#[test]
fn ensure_clean_errors_on_first_issue() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.remove_element(DicomTag::new(0x0020, 0x000E)); // SeriesInstanceUID
    let path = write_to_temp(&obj, "bad.dcm");

    let err = ensure_dicom_file_clean(&path, &VerifyOptions::default()).unwrap_err();
    assert!(err.to_string().contains("verification failed"));
}

fn leak_options(values: &[&str]) -> VerifyOptions {
    VerifyOptions {
        prohibited_values: values.iter().map(|value| (*value).to_owned()).collect(),
        ..Default::default()
    }
}

#[test]
fn a_prohibited_value_inside_a_sequence_item_is_reported() {
    let mut item = Dataset::new();
    item.put(DataElement::new(
        DicomTag::new(0x0040, 0xA123),
        VR::PN,
        PrimitiveValue::from("Doe^John"),
    ));
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.put(DataElement::new(
        DicomTag::new(0x0040, 0xA730),
        VR::SQ,
        ritk_dicom::dataset::Sequence::from(vec![item]),
    ));
    let path = write_to_temp(&obj, "nested.dcm");

    let report = verify_dicom_file(&path, &leak_options(&["Doe^John"])).unwrap();

    assert_eq!(
        report.issues,
        [VerifyIssue::PatientLeak {
            value: "Doe^John".to_owned()
        }]
    );
}

#[test]
fn a_short_identifier_is_found_as_a_word_and_not_inside_uids_or_numbers() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x103E),
        VR::LO,
        PrimitiveValue::from("series 17 axial 740"),
    ));
    let path = write_to_temp(&obj, "short.dcm");

    for inside in ["7", "74", "40"] {
        let report = verify_dicom_file(&path, &leak_options(&[inside])).unwrap();
        assert_eq!(
            report.issues,
            [],
            "{inside} adjoins a digit wherever it occurs"
        );
    }

    let word = verify_dicom_file(&path, &leak_options(&["17"])).unwrap();
    assert_eq!(
        word.issues,
        [VerifyIssue::PatientLeak {
            value: "17".to_owned()
        }]
    );
}

#[test]
fn an_empty_prohibited_value_matches_nothing() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    // Between the space and the hyphen an empty string has no letter or digit
    // on either side, so only the emptiness rule keeps it from matching.
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x103E),
        VR::LO,
        PrimitiveValue::from("ward - 7"),
    ));
    let path = write_to_temp(&obj, "empty.dcm");
    let report = verify_dicom_file(&path, &leak_options(&[""])).unwrap();
    assert_eq!(report.issues, []);
}

#[test]
fn a_missing_frame_of_reference_passes_and_a_malformed_one_is_reported() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.remove_element(DicomTag::new(0x0020, 0x0052));
    let absent = write_to_temp(&obj, "no_frame.dcm");
    assert_eq!(
        verify_dicom_file(&absent, &VerifyOptions::default())
            .unwrap()
            .issues,
        []
    );

    obj.put(DataElement::new(
        DicomTag::new(0x0020, 0x0052),
        VR::UI,
        PrimitiveValue::from("1.02.3"),
    ));
    let malformed = write_to_temp(&obj, "bad_frame.dcm");
    assert_eq!(
        verify_dicom_file(&malformed, &VerifyOptions::default())
            .unwrap()
            .issues,
        [VerifyIssue::InvalidUid {
            tag: "FrameOfReferenceUID".to_owned(),
            value: "1.02.3".to_owned()
        }]
    );
}

#[test]
fn an_instance_set_spanning_two_series_and_studies_passes() {
    let paths = [
        write_to_temp(&build_object("1.2.1", "1.2.1.1", "1.2.1.1.1"), "a.dcm"),
        write_to_temp(&build_object("1.2.1", "1.2.1.2", "1.2.1.2.1"), "b.dcm"),
        write_to_temp(&build_object("1.2.2", "1.2.2.1", "1.2.2.1.1"), "c.dcm"),
    ];
    assert_eq!(
        verify_dicom_instances(&paths, &VerifyOptions::default()),
        []
    );
}

#[test]
fn an_instance_set_reports_a_repeated_sop_uid_and_a_series_in_two_studies() {
    let paths = [
        write_to_temp(&build_object("1.2.1", "1.2.1.1", "1.2.1.1.1"), "a.dcm"),
        write_to_temp(&build_object("1.2.1", "1.2.1.1", "1.2.1.1.1"), "b.dcm"),
        write_to_temp(&build_object("1.2.9", "1.2.1.1", "1.2.1.1.2"), "c.dcm"),
    ];
    assert_eq!(
        verify_dicom_instances(&paths, &VerifyOptions::default()),
        [
            (
                paths[1].clone(),
                VerifyIssue::DuplicateSopInstanceUid {
                    value: "1.2.1.1.1".to_owned()
                }
            ),
            (
                paths[2].clone(),
                VerifyIssue::UidMismatch {
                    tag: "StudyInstanceUID".to_owned(),
                    expected: "1.2.1".to_owned(),
                    found: "1.2.9".to_owned()
                }
            ),
        ]
    );
}

#[test]
fn an_occurrence_overlapping_a_rejected_one_is_still_found() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x103E),
        VR::LO,
        PrimitiveValue::from("112-12-12"),
    ));
    let path = write_to_temp(&obj, "overlap.dcm");

    let report = verify_dicom_file(&path, &leak_options(&["12-12"])).unwrap();

    assert_eq!(
        report.issues,
        [VerifyIssue::PatientLeak {
            value: "12-12".to_owned()
        }]
    );
}

#[test]
fn a_uid_or_date_is_leaked_only_as_a_whole_value() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.put(DataElement::new(
        DicomTag::new(0x0010, 0x0030),
        VR::DA,
        PrimitiveValue::from("19750101"),
    ));
    let path = write_to_temp(&obj, "values.dcm");

    for whole in ["19750101", "1.2.3.4.5.7"] {
        let report = verify_dicom_file(&path, &leak_options(&[whole])).unwrap();
        assert_eq!(
            report.issues,
            [VerifyIssue::PatientLeak {
                value: whole.to_owned()
            }]
        );
    }
    for part in ["7", "1975", "1.2.3"] {
        let report = verify_dicom_file(&path, &leak_options(&[part])).unwrap();
        assert_eq!(report.issues, [], "{part} is only part of a value");
    }
}

#[test]
fn optional_uids_keep_their_format_unchecked() {
    let obj = build_object("1.02.3", "1.2.3.4.5.7", "1.2.3.4.5.8");
    let path = write_to_temp(&obj, "optional.dcm");
    let options = VerifyOptions {
        require_uids: false,
        ..Default::default()
    };
    assert_eq!(verify_dicom_file(&path, &options).unwrap().issues, []);
}

#[test]
fn a_file_meta_uid_naming_another_instance_is_reported() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.update_meta(|meta| meta.media_storage_sop_instance_uid = "1.2.3.4.5.99".to_owned());
    let path = write_to_temp(&obj, "stale-meta.dcm");

    let report = verify_dicom_file(&path, &VerifyOptions::default()).unwrap();

    assert_eq!(
        report.issues,
        [VerifyIssue::UidMismatch {
            tag: "MediaStorageSOPInstanceUID".to_owned(),
            expected: "1.2.3.4.5.8".to_owned(),
            found: "1.2.3.4.5.99".to_owned(),
        }]
    );
}
