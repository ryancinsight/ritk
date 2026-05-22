//! Extended integration tests for DICOM anonymization (PS 3.15 Annex E).
//!
//! This file covers `anonymize_object` core integration tests and the
//! Enhanced profile private tag removal scenario. File roundtrip, statistics,
//! and clean_private_tags tests are in `tests_anonymize_stats.rs`.
//!
//! # Test Coverage Index (this file)
//! 1. anonymize_object_replaces_patient_name_with_default
//! 2. anonymize_object_replaces_patient_id_with_default
//! 3. anonymize_object_removes_institution_name
//! 4. anonymize_object_replace_uids_changes_sop_uid
//! 5. anonymize_object_uid_replacement_is_deterministic
//! 6. anonymize_object_pixel_data_preserved
//! 7. anonymize_object_sop_class_preserved
//! 8. anonymize_object_configurable_patient_name
//! 9. anonymize_object_accession_number_emptied
//! 10. anonymize_object_enhanced_removes_private_tags

use super::{AnonymizationProfile, AnonymizeOptions};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{meta::FileMetaTableBuilder, FileDicomObject, InMemDicomObject};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Build a minimal, self-consistent `FileDicomObject` with:
/// - PatientName (0010,0010) PN "Doe^John"
/// - PatientID (0010,0020) LO "PAT001"
/// - PatientBirthDate (0010,0030) DA "19900101"
/// - InstitutionName (0008,0080) LO "General Hospital"
/// - AccessionNumber (0008,0050) SH "ACC123"
/// - SOPInstanceUID (0008,0018) UI "1.2.3.4.5"
/// - StudyInstanceUID (0020,000D) UI "1.2.3.4.6"
/// - SeriesInstanceUID (0020,000E) UI "1.2.3.4.7"
/// - SOPClassUID (0008,0016) UI "1.2.840.10008.5.1.4.1.1.2"
/// - Modality (0008,0060) CS "CT"
/// - Rows (0028,0010) US 512
/// - Columns (0028,0011) US 512
/// - PixelSpacing (0028,0030) DS "0.5\\0.5"
/// - ImagePositionPatient (0020,0032) DS "0\\0\\0"
/// - ImageOrientationPatient (0020,0037) DS "1\\0\\0\\0\\1\\0"
/// - SliceThickness (0018,0050) DS "1.0"
fn make_test_object() -> FileDicomObject<InMemDicomObject> {
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Doe^John"),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0020),
        VR::LO,
        PrimitiveValue::from("PAT001"),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0030),
        VR::DA,
        PrimitiveValue::from("19900101"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0080),
        VR::LO,
        PrimitiveValue::from("General Hospital"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0050),
        VR::SH,
        PrimitiveValue::from("ACC123"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("1.2.3.4.5"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from("1.2.3.4.6"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from("1.2.3.4.7"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("CT"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(512u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(512u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0030),
        VR::DS,
        PrimitiveValue::from("0.5\\0.5"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0032),
        VR::DS,
        PrimitiveValue::from("0\\0\\0"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0037),
        VR::DS,
        PrimitiveValue::from("1\\0\\0\\0\\1\\0"),
    ));
    obj.put(DataElement::new(
        Tag(0x0018, 0x0050),
        VR::DS,
        PrimitiveValue::from("1.0"),
    ));
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("1.2.3.4.5")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("FileMetaTableBuilder must succeed for test fixture")
}

// ─── anonymize_object (core integration) ─────────────────────────────────────

#[test]
fn anonymize_object_replaces_patient_name_with_default() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    let name = anon
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must be present after Dummy action")
        .to_str()
        .expect("PatientName must be readable as string")
        .trim()
        .to_owned();
    assert_eq!(
        name, "ANONYMOUS",
        "PatientName must be 'ANONYMOUS' after Basic anonymization"
    );
}

#[test]
fn anonymize_object_replaces_patient_id_with_default() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    let pid = anon
        .element(Tag(0x0010, 0x0020))
        .expect("PatientID must be present after Dummy action")
        .to_str()
        .expect("PatientID must be readable as string")
        .trim()
        .to_owned();
    assert_eq!(
        pid, "ANON001",
        "PatientID must be 'ANON001' after Basic anonymization"
    );
}

#[test]
fn anonymize_object_removes_institution_name() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    assert!(
        anon.element(Tag(0x0008, 0x0080)).is_err(),
        "InstitutionName must be absent after Remove action in Basic profile"
    );
}

#[test]
fn anonymize_object_replace_uids_changes_sop_uid() {
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::BasicReplaceUids,
        ..AnonymizeOptions::default()
    };
    let (anon, _result) =
        super::anonymize_object(make_test_object(), &opts).expect("anonymize_object must succeed");
    let new_uid = anon
        .element(Tag(0x0008, 0x0018))
        .expect("SOPInstanceUID must exist after ReplaceUid")
        .to_str()
        .expect("SOPInstanceUID must be readable as string")
        .trim()
        .to_owned();
    assert_ne!(
        new_uid, "1.2.3.4.5",
        "SOPInstanceUID must differ from original after ReplaceUid"
    );
    assert!(
        new_uid.starts_with("2.25."),
        "Replaced UID must use ISO 9834-8 UUID arc root 2.25., got: {new_uid}"
    );
}

#[test]
fn anonymize_object_uid_replacement_is_deterministic() {
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::BasicReplaceUids,
        ..AnonymizeOptions::default()
    };
    let uid_a = super::anonymize_object(make_test_object(), &opts)
        .expect("first call must succeed")
        .0
        .element(Tag(0x0008, 0x0018))
        .expect("SOPInstanceUID must exist")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    let uid_b = super::anonymize_object(make_test_object(), &opts)
        .expect("second call must succeed")
        .0
        .element(Tag(0x0008, 0x0018))
        .expect("SOPInstanceUID must exist")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(
        uid_a, uid_b,
        "UID replacement must be deterministic: same original UID must always map to the same output"
    );
}

#[test]
fn anonymize_object_pixel_data_preserved() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    let sop_class = anon
        .element(Tag(0x0008, 0x0016))
        .expect("SOPClassUID must be present")
        .to_str()
        .expect("SOPClassUID must be readable as string")
        .trim()
        .to_owned();
    assert_eq!(
        sop_class, "1.2.840.10008.5.1.4.1.1.2",
        "SOPClassUID must be preserved (Keep action)"
    );
}

#[test]
fn anonymize_object_sop_class_preserved() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    let sop_class = anon
        .element(Tag(0x0008, 0x0016))
        .expect("SOPClassUID must exist after anonymization")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(
        sop_class, "1.2.840.10008.5.1.4.1.1.2",
        "SOPClassUID must remain unchanged"
    );
}

#[test]
fn anonymize_object_configurable_patient_name() {
    let obj = make_test_object();
    let opts = AnonymizeOptions {
        patient_name: "REDACTED".to_owned(),
        patient_id: "REDACTED_ID".to_owned(),
        ..AnonymizeOptions::default()
    };
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    let name = anon
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must exist")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(name, "REDACTED", "PatientName must use configured value");
    let pid = anon
        .element(Tag(0x0010, 0x0020))
        .expect("PatientID must exist")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(pid, "REDACTED_ID", "PatientID must use configured value");
}

#[test]
fn anonymize_object_accession_number_emptied() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    let acc = anon
        .element(Tag(0x0008, 0x0050))
        .expect("AccessionNumber must be present after Empty action");
    let acc_str = acc.to_str().expect("must be readable");
    let val = acc_str.trim();
    assert!(
        val.is_empty(),
        "AccessionNumber must be empty after Z/D action, got: '{val}'"
    );
}

// ─── anonymize_object — extended scenarios ─────────────────────────────────────

#[test]
fn anonymize_object_enhanced_removes_private_tags() {
    let mut obj = make_test_object();
    // Insert synthetic private elements.
    obj.put(DataElement::new(
        Tag(0x0009, 0x0010),
        VR::LO,
        PrimitiveValue::from("vendor_data"),
    ));
    obj.put(DataElement::new(
        Tag(0x0019, 0x0010),
        VR::LO,
        PrimitiveValue::from("another_private"),
    ));
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::Enhanced,
        ..AnonymizeOptions::default()
    };
    let (anon, result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    assert!(
        anon.element(Tag(0x0009, 0x0010)).is_err(),
        "private tag (0009,0010) must be removed in Enhanced profile"
    );
    assert!(
        anon.element(Tag(0x0019, 0x0010)).is_err(),
        "private tag (0019,0010) must be removed in Enhanced profile"
    );
    assert_eq!(
        result.private_tags_removed, 2,
        "AnonymizeResult must report exactly 2 private tags removed"
    );
}
