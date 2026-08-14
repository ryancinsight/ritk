//! Tests for the export-time DICOM metadata verification gate.
//!
//! These tests exercise `verify::*` against in-memory objects written to
//! temporary files, covering the standard PACS anonymization export failure
//! modes: corrupt writes, diverged series UIDs, leaked identifiers, and
//! geometry/pixel drift.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::super::anonymize::verify::{
    ensure_dicom_file_clean, verify_dicom_directory, verify_dicom_file, VerifyIssue, VerifyOptions,
};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{meta::FileMetaTableBuilder, FileDicomObject, InMemDicomObject};

fn build_object(
    study_uid: &str,
    series_uid: &str,
    sop_uid: &str,
) -> FileDicomObject<InMemDicomObject> {
    let mut obj = InMemDicomObject::new_empty();
    // Identification / UIDs
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from(study_uid),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from(series_uid),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_uid),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0052),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.1.2.3.4.5.6"),
    ));
    // Geometry: 2x2x1, 16-bit
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    // PixelData: 2*2*2 = 8 bytes (16-bit).
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(dicom::core::smallvec::SmallVec::from_vec(vec![0, 1, 2, 3])),
    ));
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid(sop_uid)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("synthetic metadata valid")
}

fn write_to_temp(obj: &FileDicomObject<InMemDicomObject>, name: &str) -> std::path::PathBuf {
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
    obj.remove_element(Tag(0x0020, 0x000D)); // StudyInstanceUID
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
fn geometry_pixel_mismatch_reported() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    // Declare 4x4 but keep 2x2 pixel data → length mismatch.
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(4_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(4_u16),
    ));
    let path = write_to_temp(&obj, "geometry_mismatch.dcm");

    let report = verify_dicom_file(&path, &VerifyOptions::default()).unwrap();
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, VerifyIssue::GeometryMismatch { .. })),
        "geometry mismatch must be reported, got {:?}",
        report.issues
    );
}

#[test]
fn prohibited_value_leak_reported() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
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
    obj.remove_element(Tag(0x0020, 0x000E)); // SeriesInstanceUID
    let path = write_to_temp(&obj, "bad.dcm");

    let err = ensure_dicom_file_clean(&path, &VerifyOptions::default()).unwrap_err();
    assert!(err.to_string().contains("verification failed"));
}
