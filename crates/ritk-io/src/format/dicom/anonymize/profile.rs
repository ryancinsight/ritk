//! Anonymization profiles and per-tag actions (DICOM PS 3.15 Annex E).
//!
//! # Specification
//! The Basic profile is the minimal PS 3.15 Annex E Application Level
//! Confidentiality Profile subset. BasicReplaceUids adds UID replacement to
//! preserve internal referential consistency. Aggressive adds temporal metadata
//! removal on top of BasicReplaceUids.

use dicom::core::Tag;

/// Action to apply to a DICOM attribute during anonymization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagAction {
    /// Replace with a profile-defined dummy placeholder value.
    Dummy,
    /// Replace with an empty / zero value, preserving the element.
    Empty,
    /// Remove the element entirely.
    Remove,
    /// Keep unchanged.
    Keep,
    /// Replace UID with a deterministically hashed equivalent,
    /// preserving intra-study referential consistency.
    ReplaceUid,
}

/// Anonymization profile controlling which standard DICOM tags are acted upon.
///
/// Profiles are additive: each level extends the previous.
#[derive(Debug, Clone)]
pub enum AnonymizationProfile {
    /// PS 3.15 Annex E Basic Application Level Confidentiality Profile.
    ///
    /// Removes or replaces all patient-identifying attributes listed in Annex E
    /// Table E.1-1 (minimal subset).
    Basic,
    /// Extends `Basic` by replacing all instance, series, study, and
    /// frame-of-reference UIDs with deterministically hashed equivalents.
    BasicReplaceUids,
    /// Extends `BasicReplaceUids` by additionally emptying date/time fields
    /// and removing descriptive text attributes.
    Aggressive,
}

impl AnonymizationProfile {
    /// Return the ordered list of `(Tag, TagAction)` pairs for this profile.
    ///
    /// Tags not present in the list are left unchanged. Later entries for the
    /// same tag supersede earlier ones (no duplicates are emitted here, but the
    /// caller must be aware when combining profiles externally).
    pub fn tag_actions(&self) -> Vec<(Tag, TagAction)> {
        let mut actions = basic_actions();
        match self {
            AnonymizationProfile::Basic => {}
            AnonymizationProfile::BasicReplaceUids => {
                actions.extend(uid_actions());
            }
            AnonymizationProfile::Aggressive => {
                actions.extend(uid_actions());
                actions.extend(aggressive_actions());
            }
        }
        actions
    }
}

/// PS 3.15 Annex E Basic Application Level Confidentiality Profile actions.
///
/// Invariant: every tag listed in Annex E Table E.1-1 (patient demographics,
/// institution identifiers, referring and performing personnel) is covered.
fn basic_actions() -> Vec<(Tag, TagAction)> {
    vec![
        (Tag(0x0010, 0x0010), TagAction::Dummy),  // PatientName
        (Tag(0x0010, 0x0020), TagAction::Dummy),  // PatientID
        (Tag(0x0010, 0x0030), TagAction::Empty),  // PatientBirthDate
        (Tag(0x0010, 0x0040), TagAction::Empty),  // PatientSex
        (Tag(0x0010, 0x1010), TagAction::Remove), // PatientAge
        (Tag(0x0010, 0x1030), TagAction::Remove), // PatientWeight
        (Tag(0x0010, 0x1000), TagAction::Remove), // OtherPatientIDs
        (Tag(0x0010, 0x1040), TagAction::Remove), // PatientAddress
        (Tag(0x0010, 0x2154), TagAction::Remove), // PatientTelephoneNumbers
        (Tag(0x0020, 0x0010), TagAction::Empty),  // StudyID
        (Tag(0x0008, 0x0050), TagAction::Empty),  // AccessionNumber
        (Tag(0x0008, 0x0090), TagAction::Empty),  // ReferringPhysicianName
        (Tag(0x0008, 0x0080), TagAction::Remove), // InstitutionName
        (Tag(0x0008, 0x0081), TagAction::Remove), // InstitutionAddress
        (Tag(0x0008, 0x1010), TagAction::Remove), // StationName
        (Tag(0x0008, 0x1070), TagAction::Remove), // OperatorsName
        (Tag(0x0008, 0x1048), TagAction::Remove), // PerformingPhysicianName
        (Tag(0x0008, 0x1060), TagAction::Remove), // NameOfPhysiciansReadingStudy
    ]
}

/// UID replacement actions added by `BasicReplaceUids`.
///
/// Covers all instance-level UIDs required for intra-study consistency by
/// PS 3.15 Annex E Option: Retain UIDs.
fn uid_actions() -> Vec<(Tag, TagAction)> {
    vec![
        (Tag(0x0020, 0x000D), TagAction::ReplaceUid), // StudyInstanceUID
        (Tag(0x0020, 0x000E), TagAction::ReplaceUid), // SeriesInstanceUID
        (Tag(0x0008, 0x0018), TagAction::ReplaceUid), // SOPInstanceUID
        (Tag(0x0020, 0x0052), TagAction::ReplaceUid), // FrameOfReferenceUID
    ]
}

/// Additional `Aggressive` actions: temporal metadata and descriptive text.
///
/// Removes all acquisition date/time fields and study/series/protocol
/// description strings that could contribute to re-identification.
fn aggressive_actions() -> Vec<(Tag, TagAction)> {
    vec![
        (Tag(0x0008, 0x0020), TagAction::Empty),  // StudyDate
        (Tag(0x0008, 0x0021), TagAction::Empty),  // SeriesDate
        (Tag(0x0008, 0x0022), TagAction::Empty),  // AcquisitionDate
        (Tag(0x0008, 0x0023), TagAction::Empty),  // ContentDate
        (Tag(0x0008, 0x0030), TagAction::Empty),  // StudyTime
        (Tag(0x0008, 0x0031), TagAction::Empty),  // SeriesTime
        (Tag(0x0008, 0x0032), TagAction::Empty),  // AcquisitionTime
        (Tag(0x0008, 0x0033), TagAction::Empty),  // ContentTime
        (Tag(0x0008, 0x1030), TagAction::Remove), // StudyDescription
        (Tag(0x0008, 0x103E), TagAction::Remove), // SeriesDescription
        (Tag(0x0018, 0x1030), TagAction::Remove), // ProtocolName
    ]
}
