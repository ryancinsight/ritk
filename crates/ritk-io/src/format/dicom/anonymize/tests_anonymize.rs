//! Unit tests for DICOM anonymization (PS 3.15 Annex E).
//!
//! All tests that verify `anonymize_object` use only in-memory objects
//! constructed with `InMemDicomObject::new_empty()` + `.with_meta()`;
//! no pre-existing DICOM files are required.
//!
//! Profile tests verify the tag-action vectors returned by
//! `AnonymizationProfile::tag_actions` without invoking file I/O.

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
/// - PatientName  (0010,0010) PN "Doe^John"
/// - PatientID    (0010,0020) LO "PAT001"
/// - InstitutionName (0008,0080) LO "General Hospital"
/// - SOPInstanceUID  (0008,0018) UI "1.2.3.4.5"
/// - StudyInstanceUID (0020,000D) UI "1.2.3.4.6"
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
        Tag(0x0008, 0x0080),
        VR::LO,
        PrimitiveValue::from("General Hospital"),
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
        actions.len() >= 18,
        "Basic profile must cover at least 18 tags (PS 3.15 Annex E minimum subset), got {}",
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
        (Tag(0x0008, 0x1048), "PerformingPhysicianName"),
        (Tag(0x0008, 0x1060), "NameOfPhysiciansReadingStudy"),
    ];
    for (tag, name) in remove_tags {
        assert_eq!(
            find_action(&actions, tag),
            Some(TagAction::Remove),
            "{name} ({tag:?}) must map to Remove in Basic profile"
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
fn profile_basic_replace_uids_includes_all_four_uids() {
    let actions = AnonymizationProfile::BasicReplaceUids.tag_actions();
    let uid_tags = [
        (Tag(0x0020, 0x000D), "StudyInstanceUID"),
        (Tag(0x0020, 0x000E), "SeriesInstanceUID"),
        (Tag(0x0008, 0x0018), "SOPInstanceUID"),
        (Tag(0x0020, 0x0052), "FrameOfReferenceUID"),
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

// ─── UID hash ─────────────────────────────────────────────────────────────────

#[test]
fn generate_uid_hash_is_deterministic() {
    let a = generate_uid_from_hash("1.2.3.4.5", "ritk_anon_v1");
    let b = generate_uid_from_hash("1.2.3.4.5", "ritk_anon_v1");
    assert_eq!(a, b, "same (original, salt) must produce identical UID");
}

#[test]
fn generate_uid_hash_changes_with_input() {
    let a = generate_uid_from_hash("1.2.3.4.5", "ritk_anon_v1");
    let b = generate_uid_from_hash("1.2.3.4.6", "ritk_anon_v1");
    assert_ne!(
        a, b,
        "distinct original UIDs must produce distinct hashed UIDs"
    );
}

#[test]
fn generate_uid_hash_output_format_is_valid_dicom_uid() {
    let uid = generate_uid_from_hash("1.2.840.10008.5.1.4.1.1.2", "ritk_anon_v1");
    assert!(
        uid.starts_with("2.999."),
        "UID must start with private OID root 2.999., got: {uid}"
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

// ─── anonymize_object ─────────────────────────────────────────────────────────

#[test]
fn anonymize_object_replaces_patient_name_with_anonymous() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let anon = super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
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
fn anonymize_object_replaces_patient_id_with_anon_id() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let anon = super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    let pid = anon
        .element(Tag(0x0010, 0x0020))
        .expect("PatientID must be present after Dummy action")
        .to_str()
        .expect("PatientID must be readable as string")
        .trim()
        .to_owned();
    assert_eq!(
        pid, "ANON_ID",
        "PatientID must be 'ANON_ID' after Basic anonymization"
    );
}

#[test]
fn anonymize_object_removes_institution_name() {
    let obj = make_test_object();
    let opts = AnonymizeOptions::default();
    let anon = super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    assert!(
        anon.element(Tag(0x0008, 0x0080)).is_err(),
        "InstitutionName must be absent after Remove action in Basic profile"
    );
}

#[test]
fn anonymize_object_replace_uids_changes_sop_uid() {
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::BasicReplaceUids,
        clean_pixel_data: false,
        clean_private_tags: false,
    };
    let anon =
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
        new_uid.starts_with("2.999."),
        "Replaced UID must use RITK private OID root, got: {new_uid}"
    );
}

#[test]
fn anonymize_object_uid_replacement_is_deterministic() {
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::BasicReplaceUids,
        clean_pixel_data: false,
        clean_private_tags: false,
    };
    let uid_a = super::anonymize_object(make_test_object(), &opts)
        .expect("first call must succeed")
        .element(Tag(0x0008, 0x0018))
        .expect("SOPInstanceUID must exist")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    let uid_b = super::anonymize_object(make_test_object(), &opts)
        .expect("second call must succeed")
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
}

// ─── clean_private_tags (GAP-262-IO-08) ──────────────────────────────────────

/// `clean_private_tags = false` preserves private elements in the output.
///
/// Private tag (0009,0010) — odd group — must remain when
/// `clean_private_tags` is `false`.
#[test]
fn clean_private_tags_false_preserves_private_elements() {
    let mut obj = make_test_object();
    // Insert a synthetic private element: group 0x0009 (odd) is private.
    // (0009,0010) UT "vendor_specific_data"
    obj.put(DataElement::new(
        Tag(0x0009, 0x0010),
        VR::UT,
        PrimitiveValue::from("vendor_specific_data"),
    ));
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::Basic,
        clean_pixel_data: false,
        clean_private_tags: false,
    };
    let anon = super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    assert!(
        anon.element(Tag(0x0009, 0x0010)).is_ok(),
        "private tag (0009,0010) must be preserved when clean_private_tags=false"
    );
}

/// `clean_private_tags = true` removes all private elements.
///
/// DICOM PS 3.15: elements with an odd group number are private.
/// Removal is required for full Annex E confidentiality compliance.
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
        clean_pixel_data: false,
        clean_private_tags: true,
    };
    let anon = super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    assert!(
        anon.element(Tag(0x0009, 0x0010)).is_err(),
        "private tag (0009,0010) must be removed when clean_private_tags=true"
    );
    assert!(
        anon.element(Tag(0x0019, 0x0010)).is_err(),
        "private tag (0019,0010) must be removed when clean_private_tags=true"
    );
}

/// Standard (even-group) elements must not be removed by `clean_private_tags`.
///
/// PatientName (0010,0010) has an even group number and must remain.
#[test]
fn clean_private_tags_true_preserves_standard_elements() {
    let obj = make_test_object();
    let opts = AnonymizeOptions {
        profile: AnonymizationProfile::Basic,
        clean_pixel_data: false,
        clean_private_tags: true,
    };
    let anon = super::anonymize_object(obj, &opts).expect("anonymize_object must succeed");
    // PatientName (0010,0010) is an even-group tag, anonymized to ANONYMOUS
    let name = anon
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must exist in output after clean_private_tags")
        .to_str()
        .expect("must be string")
        .trim()
        .to_owned();
    assert_eq!(name, "ANONYMOUS", "PatientName must be anonymized, not removed");
}
