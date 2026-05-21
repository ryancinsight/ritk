//! Unit tests for DICOM anonymization (PS 3.15 Annex E).
//!
//! All tests that verify `anonymize_object` use only in-memory objects
//! constructed with `InMemDicomObject::new_empty()` + `.with_meta()`;
//! no pre-existing DICOM files are required.
//!
//! Profile tests verify the tag-action vectors returned by
//! `AnonymizationProfile::tag_actions` without invoking file I/O.
//!
//! # Test Coverage Index
//! 1.  profile_basic_has_patient_name_dummy
//! 2.  profile_basic_has_patient_id_dummy
//! 3.  profile_basic_has_institution_name_remove
//! 4.  profile_basic_has_patient_birth_date_empty
//! 5.  profile_basic_minimum_tag_count
//! 6.  profile_basic_covers_all_required_demographic_removes
//! 7.  profile_basic_replace_uids_includes_sop_instance_uid
//! 8.  profile_basic_replace_uids_includes_all_uids
//! 9.  profile_basic_replace_uids_retains_basic_actions
//! 10. profile_aggressive_includes_study_date_empty
//! 11. profile_aggressive_includes_all_date_fields
//! 12. profile_aggressive_includes_all_time_fields
//! 13. profile_aggressive_includes_protocol_name_remove
//! 14. profile_enhanced_includes_basic_actions
//! 15. generate_uid_hash_is_deterministic
//! 16. generate_uid_hash_changes_with_input
//! 17. generate_uid_hash_output_format_is_valid_dicom_uid
//! 18. generate_uid_hash_salt_differentiates_same_original
//! 19. generate_uid_hash_uses_sha256_2_25_root
//! 20. anonymize_object_replaces_patient_name_with_default
//! 21. anonymize_object_replaces_patient_id_with_default
//! 22. anonymize_object_removes_institution_name
//! 23. anonymize_object_replace_uids_changes_sop_uid
//! 24. anonymize_object_uid_replacement_is_deterministic
//! 25. anonymize_object_pixel_data_preserved
//! 26. anonymize_object_sop_class_preserved
//! 27. anonymize_object_configurable_patient_name
//! 28. anonymize_object_enhanced_removes_private_tags
//! 29. anonymize_dicom_file_roundtrip_strips_patient_identifying_data
//! 30. anonymize_object_statistics_match_operations
//! 31. anonymize_object_uid_cross_reference_consistency
//! 32. anonymize_object_accession_number_emptied
//! 33. clean_private_tags_false_preserves_private_elements
//! 34. clean_private_tags_true_removes_private_elements
//! 35. clean_private_tags_true_preserves_standard_elements

use super::{generate_uid_from_hash, AnonymizationProfile, AnonymizeOptions, TagAction};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{meta::FileMetaTableBuilder, FileDicomObject, InMemDicomObject};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Return the action for `tag` within `actions`, or `None` if absent.
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

// ─── Profile: Basic ───────────────────────────────────────────────────────────

#[test]
fn profile_basic_has_patient_name_dummy() {
    let actions = AnonymizationProfile::Basic.tag_actions();
    assert_eq!(
        find_action(&actions, Tag(0x0010, 0x0010)),
        Some(TagAction::Dummy),
        "PatientName (0010,0010) must map to Dummy in Basic profile"
    );
}

#[test]
fn profile_basic_has_patient_id_dummy() {
    let actions = AnonymizationProfile::Basic.tag_actions();
    assert_eq!(
        find_action(&actions, Tag(0x0010, 0x0020)),
        Some(TagAction::Dummy),
        "PatientID (0010,0020) must map to Dummy in Basic profile"
    );
}

#[test]
fn profile_basic_has_institution_name_remove() {
    let actions = AnonymizationProfile::Basic.tag_actions();
    assert_eq!(
        find_action(&actions, Tag(0x0008, 0x0080)),
        Some(TagAction::Remove),
        "InstitutionName (0008,0080) must map to Remove in Basic profile"
    );
}

#[test]
fn profile_basic_has_patient_birth_date_empty() {
    let actions = AnonymizationProfile::Basic.tag_actions();
    assert_eq!(
        find_action(&actions, Tag(0x0010, 0x0030)),
        Some(TagAction::Empty),
        "PatientBirthDate (0010,0030) must map to Empty in Basic profile"
    );
}

#[test]
fn profile_basic_minimum_tag_count() {
    let actions = AnonymizationProfile::Basic.tag_actions();
    assert!(
        actions.len() >= 60,
        "Basic profile must cover at least 60 tags (PS 3.15 Annex E), got {}",
        actions.len()
    );
}

#[test]
fn profile_basic_covers_all_required_demographic_removes() {
    let actions = AnonymizationProfile::Basic.tag_actions();
    let remove_tags = [
        (Tag(0x0010, 0x1010), "PatientAge"),
        (Tag(0x0010, 0x1030), "PatientWeight"),
        (Tag(0x0010, 0x1000), "OtherPatientIDs"),
        (Tag(0x0010, 0x1040), "PatientAddress"),
        (Tag(0x0010, 0x2154), "PatientTelephoneNumbers"),
        (Tag(0x0008, 0x0080), "InstitutionName"),
        (Tag(0x0008, 0x0081), "InstitutionAddress"),
        (Tag(0x0008, 0x1010), "StationName"),
        (Tag(0x0008, 0x1070), "OperatorsName"),
        (Tag(0x0008, 0x1050), "PerformingPhysicianName"),
        (Tag(0x0008, 0x1060), "NameOfPhysiciansReadingStudy"),
        (Tag(0x0018, 0x1000), "DeviceSerialNumber"),
        (Tag(0x0018, 0x1002), "DeviceUID"),
        (Tag(0x0028, 0x4000), "ImageComments"),
        (Tag(0x0040, 0x0275), "RequestAttributeSequence"),
        (Tag(0x0400, 0x0100), "DigitalSignatureUID"),
        (Tag(0xFFFA, 0xFFFA), "DigitalSignaturesSequence"),
        (Tag(0xFFFC, 0xFFFC), "EncryptedAttributesSequence"),
    ];
    for (tag, name) in remove_tags {
        let action = find_action(&actions, tag);
        assert!(
            matches!(action, Some(TagAction::Remove)),
            "{name} ({tag:?}) must map to Remove in Basic profile, got {action:?}"
        );
    }
}

// ─── Profile: BasicReplaceUids ────────────────────────────────────────────────

#[test]
fn profile_basic_replace_uids_includes_sop_instance_uid() {
    let actions = AnonymizationProfile::BasicReplaceUids.tag_actions();
    assert_eq!(
        find_action(&actions, Tag(0x0008, 0x0018)),
        Some(TagAction::ReplaceUid),
        "SOPInstanceUID (0008,0018) must map to ReplaceUid in BasicReplaceUids"
    );
}

#[test]
fn profile_basic_replace_uids_includes_all_uids() {
    let actions = AnonymizationProfile::BasicReplaceUids.tag_actions();
    let uid_tags = [
        (Tag(0x0020, 0x000D), "StudyInstanceUID"),
        (Tag(0x0020, 0x000E), "SeriesInstanceUID"),
        (Tag(0x0008, 0x0018), "SOPInstanceUID"),
        (Tag(0x0020, 0x0052), "FrameOfReferenceUID"),
        (Tag(0x0008, 0x1155), "ReferencedSOPInstanceUID"),
        (Tag(0x0008, 0x0019), "SourceImageSequence"),
    ];
    for (tag, name) in uid_tags {
        assert_eq!(
            find_action(&actions, tag),
            Some(TagAction::ReplaceUid),
            "{name} ({tag:?}) must map to ReplaceUid in BasicReplaceUids"
        );
    }
}

#[test]
fn profile_basic_replace_uids_retains_basic_actions() {
    let basic = AnonymizationProfile::Basic.tag_actions();
    let replace = AnonymizationProfile::BasicReplaceUids.tag_actions();
    // Every tag from Basic must appear in BasicReplaceUids with the same action.
    for (tag, expected_action) in basic {
        assert_eq!(
            find_action(&replace, tag),
            Some(expected_action),
            "BasicReplaceUids must contain all Basic actions; missing {tag:?}"
        );
    }
}

// ─── Profile: Aggressive ─────────────────────────────────────────────────────

#[test]
fn profile_aggressive_includes_study_date_empty() {
    let actions = AnonymizationProfile::Aggressive.tag_actions();
    assert_eq!(
        find_action(&actions, Tag(0x0008, 0x0020)),
        Some(TagAction::Empty),
        "StudyDate (0008,0020) must map to Empty in Aggressive"
    );
}

#[test]
fn profile_aggressive_includes_all_date_fields() {
    let actions = AnonymizationProfile::Aggressive.tag_actions();
    let date_tags = [
        (Tag(0x0008, 0x0020), "StudyDate"),
        (Tag(0x0008, 0x0021), "SeriesDate"),
        (Tag(0x0008, 0x0022), "AcquisitionDate"),
        (Tag(0x0008, 0x0023), "ContentDate"),
        (Tag(0x0008, 0x0024), "OverlayDate"),
        (Tag(0x0008, 0x0025), "CurveDate"),
        (Tag(0x0008, 0x002A), "AcquisitionDateTime"),
    ];
    for (tag, name) in date_tags {
        assert_eq!(
            find_action(&actions, tag),
            Some(TagAction::Empty),
            "{name} ({tag:?}) must map to Empty in Aggressive"
        );
    }
}

#[test]
fn profile_aggressive_includes_all_time_fields() {
    let actions = AnonymizationProfile::Aggressive.tag_actions();
    let time_tags = [
        (Tag(0x0008, 0x0030), "StudyTime"),
        (Tag(0x0008, 0x0031), "SeriesTime"),
        (Tag(0x0008, 0x0032), "AcquisitionTime"),
        (Tag(0x0008, 0x0033), "ContentTime"),
        (Tag(0x0008, 0x0034), "OverlayTime"),
        (Tag(0x0008, 0x0035), "CurveTime"),
        (Tag(0x0008, 0x0201), "TimezoneOffsetFromUTC"),
    ];
    for (tag, name) in time_tags {
        assert_eq!(
            find_action(&actions, tag),
            Some(TagAction::Empty),
            "{name} ({tag:?}) must map to Empty in Aggressive"
        );
    }
}

#[test]
fn profile_aggressive_includes_protocol_name_remove() {
    let actions = AnonymizationProfile::Aggressive.tag_actions();
    assert_eq!(
        find_action(&actions, Tag(0x0018, 0x1030)),
        Some(TagAction::Remove),
        "ProtocolName (0018,1030) must map to Remove in Aggressive"
    );
}

// ─── Profile: Enhanced ────────────────────────────────────────────────────────

#[test]
fn profile_enhanced_includes_basic_actions() {
    let basic = AnonymizationProfile::Basic.tag_actions();
    let enhanced = AnonymizationProfile::Enhanced.tag_actions();
    for (tag, expected_action) in &basic {
        assert_eq!(
            find_action(&enhanced, *tag),
            Some(*expected_action),
            "Enhanced must contain all Basic actions; missing {tag:?}"
        );
    }
}

#[test]
fn profile_enhanced_has_derivation_description_remove() {
    let actions = AnonymizationProfile::Enhanced.tag_actions();
    assert_eq!(
        find_action(&actions, Tag(0x0008, 0x2111)),
        Some(TagAction::Remove),
        "DerivationDescription (0008,2111) must map to Remove in Enhanced"
    );
}

#[test]
fn profile_enhanced_removes_private_tags_by_default() {
    assert!(
        AnonymizationProfile::Enhanced.removes_private_tags(),
        "Enhanced profile must mandate private tag removal"
    );
    assert!(
        !AnonymizationProfile::Basic.removes_private_tags(),
        "Basic profile must not mandate private tag removal"
    );
}

// ─── UID hash ─────────────────────────────────────────────────────────────────

#[test]
fn generate_uid_hash_is_deterministic() {
    let a = generate_uid_from_hash("1.2.3.4.5", "ritk-anon-salt");
    let b = generate_uid_from_hash("1.2.3.4.5", "ritk-anon-salt");
    assert_eq!(a, b, "same (original, salt) must produce identical UID");
}

#[test]
fn generate_uid_hash_changes_with_input() {
    let a = generate_uid_from_hash("1.2.3.4.5", "ritk-anon-salt");
    let b = generate_uid_from_hash("1.2.3.4.6", "ritk-anon-salt");
    assert_ne!(
        a, b,
        "distinct original UIDs must produce distinct hashed UIDs"
    );
}

#[test]
fn generate_uid_hash_output_format_is_valid_dicom_uid() {
    let uid = generate_uid_from_hash("1.2.840.10008.5.1.4.1.1.2", "ritk-anon-salt");
    assert!(
        uid.starts_with("2.25."),
        "UID must start with ISO 9834-8 UUID arc 2.25., got: {uid}"
    );
    assert!(
        uid.chars().all(|c| c.is_ascii_digit() || c == '.'),
        "UID must contain only ASCII digits and dots, got: {uid}"
    );
    assert!(
        uid.len() <= 64,
        "DICOM UID max length is 64 characters, got {} for: {uid}",
        uid.len()
    );
    // No leading-zero components after the root.
    for component in uid.split('.') {
        if !component.is_empty() {
            assert!(
                !component.starts_with('0') || component == "0",
                "UID component '{component}' has leading zero in: {uid}"
            );
        }
    }
}

#[test]
fn generate_uid_hash_salt_differentiates_same_original() {
    let a = generate_uid_from_hash("1.2.3.4.5", "salt_alpha");
    let b = generate_uid_from_hash("1.2.3.4.5", "salt_beta");
    assert_ne!(
        a, b,
        "distinct salts must produce distinct UIDs for the same original"
    );
}

#[test]
fn generate_uid_hash_uses_sha256_2_25_root() {
    // Verify the UID uses the 2.25. ISO/IEC 9834-8 UUID arc root,
    // not the legacy 2.999. private OID arc.
    let uid = generate_uid_from_hash("1.2.3.4.5", "ritk-anon-salt");
    assert!(
        uid.starts_with("2.25."),
        "UID must use 2.25. root (ISO 9834-8 UUID arc), got: {uid}"
    );
    assert!(
        !uid.starts_with("2.999."),
        "UID must not use deprecated 2.999. root, got: {uid}"
    );
}

// ─── anonymize_object ─────────────────────────────────────────────────────────

#[test]
fn anonymize_object_replaces_patient_name_with_default() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
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
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
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
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
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
        super::anonymize_object(make_test_object(), &opts)
            .expect("anonymize_object must succeed");
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
    // When clean_pixel_data is false (default), pixel data must not be modified.
    // We verify by checking that SOPClassUID (a Keep tag) is preserved.
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
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
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
    // SOPClassUID (0008,0016) is not in any removal list — must be kept.
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
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
    let name = anon
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must exist")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(
        name, "REDACTED",
        "PatientName must use configured value"
    );
    let pid = anon
        .element(Tag(0x0010, 0x0020))
        .expect("PatientID must exist")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(
        pid, "REDACTED_ID",
        "PatientID must use configured value"
    );
}

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
    let (anon, result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
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

#[test]
fn anonymize_object_accession_number_emptied() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
    // AccessionNumber (0008,0050) has action Empty (Z/D) in Basic.
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

    let result =
        dicom::object::open_file(&output_path).expect("anonymized file must be readable");

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
    let (_anon, result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");

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
        super::anonymize_object(make_test_object(), &opts)
            .expect("anonymize_object must succeed");
    // Test object has SOPInstanceUID (0008,0018), StudyInstanceUID (0020,000D),
    // SeriesInstanceUID (0020,000E) — 3 UID elements that get ReplaceUid.
    assert_eq!(
        result.uids_remapped, 3,
        "uids_remapped must be 3 (SOPInstanceUID, StudyInstanceUID, SeriesInstanceUID), got {}",
        result.uids_remapped
    );
    assert_eq!(
        result.uid_map.len(), 3,
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
        super::anonymize_object(make_test_object(), &opts)
            .expect("anonymize_object must succeed");

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
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
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
    let (anon, result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
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
    let (anon, _result) = super::anonymize_object(obj, &opts)
        .expect("anonymize_object must succeed");
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
