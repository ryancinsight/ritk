//! Unit tests for DICOM anonymization (PS 3.15 Annex E).
//!
//! This file covers profile tag-action verification and UID hash generation.
//! Integration tests for `anonymize_object` and related functions are in
//! `tests_anonymize_extended.rs`.
//!
//! # Test Coverage Index (this file)
//! 1. profile_basic_has_patient_name_dummy
//! 2. profile_basic_has_patient_id_dummy
//! 3. profile_basic_has_institution_name_remove
//! 4. profile_basic_has_patient_birth_date_empty
//! 5. profile_basic_minimum_tag_count
//! 6. profile_basic_covers_all_required_demographic_removes
//! 7. profile_basic_replace_uids_includes_sop_instance_uid
//! 8. profile_basic_replace_uids_includes_all_uids
//! 9. profile_basic_replace_uids_retains_basic_actions
//! 10. profile_aggressive_includes_study_date_empty
//! 11. profile_aggressive_includes_all_date_fields
//! 12. profile_aggressive_includes_all_time_fields
//! 13. profile_aggressive_includes_protocol_name_remove
//! 14. profile_enhanced_includes_basic_actions
//! 15. profile_enhanced_has_derivation_description_remove
//! 16. profile_enhanced_removes_private_tags_by_default
//! 17. generate_uid_hash_is_deterministic
//! 18. generate_uid_hash_changes_with_input
//! 19. generate_uid_hash_output_format_is_valid_dicom_uid
//! 20. generate_uid_hash_salt_differentiates_same_original
//! 21. generate_uid_hash_uses_sha256_2_25_root

use super::{generate_uid_from_hash, AnonymizationProfile, TagAction};
use dicom::core::Tag;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Return the action for `tag` within `actions`, or `None` if absent.
fn find_action(actions: &[(Tag, TagAction)], tag: Tag) -> Option<TagAction> {
    actions
        .iter()
        .find(|(t, _)| *t == tag)
        .copied()
        .map(|(_, a)| a)
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
