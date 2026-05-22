//! Anonymization file roundtrip, statistics, and clean_private_tags tests.
//!
//! This file covers the file I/O roundtrip test, per-object statistics
//! tracking, UID cross-reference consistency, and the `clean_private_tags`
//! option behaviour.
//!
//! # Test Coverage Index (this file)
//! 1. anonymize_dicom_file_roundtrip_strips_patient_identifying_data
//! 2. anonymize_object_statistics_match_operations
//! 3. anonymize_object_uid_statistics
//! 4. anonymize_object_uid_cross_reference_consistency
//! 5. clean_private_tags_false_preserves_private_elements
//! 6. clean_private_tags_true_removes_private_elements
//! 7. clean_private_tags_true_preserves_standard_elements

use super::{generate_uid_from_hash, AnonymizationProfile, AnonymizeOptions, TagAction};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{meta::FileMetaTableBuilder, FileDicomObject, InMemDicomObject};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Return the action for `tag` within `actions`, or `None` if absent.
#[allow(dead_code)]
fn find_action(actions: &[(Tag, TagAction)], tag: Tag) -> Option<TagAction> {
    actions
        .iter()
        .find(|(t, _)| *t == tag)
        .copied()
        .map(|(_, a)| a)
}

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

// ─── anonymize_dicom_file (roundtrip with tempfile) ───────────────────────────

#[test]
fn anonymize_dicom_file_roundtrip_strips_patient_identifying_data() {
    let tmp = tempfile::tempdir().expect("tempdir must be creatable");
    let input_path = tmp.path().join("input.dcm");
    let output_path = tmp.path().join("output.dcm");
    make_test_object()
        .write_to_file(&input_path)
        .expect("test fixture must write to disk");
    super::anonymize_dicom_file(&input_path, &output_path, &AnonymizeOptions::default())
        .expect("anonymize_dicom_file must succeed");
    let result = dicom::object::open_file(&output_path).expect("anonymized file must be readable");
    let name = result
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must exist in output")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(name, "ANONYMOUS", "PatientName roundtrip value mismatch");
    // InstitutionName must be absent.
    assert!(
        result.element(Tag(0x0008, 0x0080)).is_err(),
        "InstitutionName must be absent in anonymized output"
    );
    // Modality must be preserved.
    let modality = result
        .element(Tag(0x0008, 0x0060))
        .expect("Modality must exist in output")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(modality, "CT", "Modality must be preserved");
}

// ─── Statistics tracking ──────────────────────────────────────────────────────

#[test]
fn anonymize_object_statistics_match_operations() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (_anon, result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    // Basic profile: at minimum, the test object has InstitutionName (Remove)
    // and several other tags that will be acted upon.
    assert!(
        result.tags_deleted > 0,
        "tags_deleted must be > 0 (InstitutionName, etc. are removed), got {}",
        result.tags_deleted
    );
    assert!(
        result.tags_zeroed > 0,
        "tags_zeroed must be > 0 (PatientName, PatientID, PatientBirthDate, AccessionNumber, etc.), got {}",
        result.tags_zeroed
    );
    assert_eq!(
        result.uids_remapped, 0,
        "uids_remapped must be 0 in Basic profile (no ReplaceUid actions)"
    );
}

#[test]
fn anonymize_object_uid_statistics() {
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::BasicReplaceUids,
        ..AnonymizeOptions::default()
    };
    let (_anon, result) =
        super::anonymize_object(make_test_object(), &opts).expect("anonymize_object must succeed");
    // Test object has SOPInstanceUID (0008,0018), StudyInstanceUID (0020,000D),
    // SeriesInstanceUID (0020,000E) — 3 UID elements that get ReplaceUid.
    assert_eq!(
        result.uids_remapped, 3,
        "uids_remapped must be 3 (SOPInstanceUID, StudyInstanceUID, SeriesInstanceUID), got {}",
        result.uids_remapped
    );
    assert_eq!(
        result.uid_map.len(),
        3,
        "uid_map must contain 3 entries, got {}",
        result.uid_map.len()
    );
    // All mapped UIDs must use 2.25. root.
    for (original, replacement) in &result.uid_map {
        assert!(
            replacement.starts_with("2.25."),
            "Mapped UID for {original} must use 2.25. root, got: {replacement}"
        );
    }
}

// ─── UID cross-reference consistency ──────────────────────────────────────────

#[test]
fn anonymize_object_uid_cross_reference_consistency() {
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::BasicReplaceUids,
        ..AnonymizeOptions::default()
    };
    let (_anon, result) =
        super::anonymize_object(make_test_object(), &opts).expect("anonymize_object must succeed");
    // The uid_map must contain mappings for the original UIDs in the test object.
    assert!(
        result.uid_map.contains_key("1.2.3.4.5"),
        "uid_map must contain mapping for original SOPInstanceUID"
    );
    assert!(
        result.uid_map.contains_key("1.2.3.4.6"),
        "uid_map must contain mapping for original StudyInstanceUID"
    );
    assert!(
        result.uid_map.contains_key("1.2.3.4.7"),
        "uid_map must contain mapping for original SeriesInstanceUID"
    );
    // Verify the hash function produces the same result as the map entries.
    let expected_sop = generate_uid_from_hash("1.2.3.4.5", &opts.uid_salt);
    assert_eq!(
        result.uid_map.get("1.2.3.4.5"),
        Some(&expected_sop),
        "uid_map entry for SOPInstanceUID must match generate_uid_from_hash output"
    );
}

// ─── clean_private_tags ──────────────────────────────────────────────────────

/// `clean_private_tags = false` preserves private elements in the output.
#[test]
fn clean_private_tags_false_preserves_private_elements() {
    let mut obj = make_test_object();
    // Insert a synthetic private element: group 0x0009 (odd) is private.
    obj.put(DataElement::new(
        Tag(0x0009, 0x0010),
        VR::UT,
        PrimitiveValue::from("vendor_specific_data"),
    ));
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::Basic,
        clean_private_tags: false,
        ..AnonymizeOptions::default()
    };
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    assert!(
        anon.element(Tag(0x0009, 0x0010)).is_ok(),
        "private tag (0009,0010) must be preserved when clean_private_tags=false"
    );
}

/// `clean_private_tags = true` removes all private elements.
#[test]
fn clean_private_tags_true_removes_private_elements() {
    let mut obj = make_test_object();
    obj.put(DataElement::new(
        Tag(0x0009, 0x0010),
        VR::UT,
        PrimitiveValue::from("vendor_data"),
    ));
    obj.put(DataElement::new(
        Tag(0x0019, 0x0010),
        VR::LO,
        PrimitiveValue::from("another_private"),
    ));
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::Basic,
        clean_private_tags: true,
        ..AnonymizeOptions::default()
    };
    let (anon, result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    assert!(
        anon.element(Tag(0x0009, 0x0010)).is_err(),
        "private tag (0009,0010) must be removed when clean_private_tags=true"
    );
    assert!(
        anon.element(Tag(0x0019, 0x0010)).is_err(),
        "private tag (0019,0010) must be removed when clean_private_tags=true"
    );
    assert_eq!(
        result.private_tags_removed, 2,
        "AnonymizeResult.private_tags_removed must be 2"
    );
}

/// Standard (even-group) elements must not be removed by `clean_private_tags`.
#[test]
fn clean_private_tags_true_preserves_standard_elements() {
    let obj = make_test_object();
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::Basic,
        clean_private_tags: true,
        ..AnonymizeOptions::default()
    };
    let (anon, _result) =
        super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    // PatientName (0010,0010) is an even-group tag, anonymized to ANONYMOUS
    let name = anon
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must exist in output after clean_private_tags")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(
        name, "ANONYMOUS",
        "PatientName must be anonymized, not removed"
    );
}
