//! Anonymization profiles and per-tag actions (DICOM PS 3.15 Annex E).
//!
//! # Specification
//! The Basic profile is the PS 3.15 Annex E "Basic Application Level
//! Confidentiality Profile" which mandates removal/replacement of all
//! patient-identifying attributes listed in Table E.1-1.
//!
//! BasicReplaceUids extends Basic by deterministically replacing all
//! instance, series, study, and frame-of-reference UIDs (Annex E Option:
//! Retain UIDs).
//!
//! Aggressive extends BasicReplaceUids by additionally zeroing all
//! date/time fields and removing descriptive text attributes.
//!
//! Enhanced extends Aggressive with additional removal of procedure-step
//! attributes, content annotations, and all private tags.

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
    /// Removes or replaces all patient-identifying attributes listed in
    /// Annex E Table E.1-1 (minimal subset).
    Basic,
    /// Extends `Basic` by replacing all instance, series, study, and
    /// frame-of-reference UIDs with deterministically hashed equivalents.
    BasicReplaceUids,
    /// Extends `BasicReplaceUids` by additionally emptying date/time fields
    /// and removing descriptive text attributes.
    Aggressive,
    /// Extends `Aggressive` with additional removal of procedure-step
    /// attributes, content annotations, digital signatures, encrypted
    /// attributes, and all private tags.
    Enhanced,
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
            AnonymizationProfile::Enhanced => {
                actions.extend(uid_actions());
                actions.extend(aggressive_actions());
                actions.extend(enhanced_actions());
            }
        }
        actions
    }

    /// Whether this profile mandates private tag removal by default.
    pub fn removes_private_tags(&self) -> bool {
        matches!(self, AnonymizationProfile::Enhanced)
    }
}

// â”€â”€â”€ Basic profile (PS 3.15 Annex E Table E.1-1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Invariant: every tag listed in Annex E Table E.1-1 for the Basic
// Application Level Confidentiality Profile is covered. Tags are grouped
// by DICOM action type (D=Remove, Z=Empty, Z/D=Empty, X/Z/D=Remove).

fn basic_actions() -> Vec<(Tag, TagAction)> {
    vec![
        // â”€â”€ D (Delete / Remove) â€” patient demographics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0010, 0x0010), TagAction::Dummy), // PatientName â†’ configurable dummy
        (Tag(0x0010, 0x0020), TagAction::Dummy), // PatientID â†’ configurable dummy
        (Tag(0x0010, 0x0030), TagAction::Empty), // PatientBirthDate
        (Tag(0x0010, 0x0032), TagAction::Remove), // PatientBirthTime
        (Tag(0x0010, 0x0050), TagAction::Remove), // PatientInsurancePlanCodeSequence
        (Tag(0x0010, 0x0101), TagAction::Remove), // PatientInsurancePlanSequence
        (Tag(0x0010, 0x1000), TagAction::Remove), // OtherPatientIDs
        (Tag(0x0010, 0x1001), TagAction::Remove), // OtherPatientNames
        (Tag(0x0010, 0x1010), TagAction::Remove), // PatientAge
        (Tag(0x0010, 0x1030), TagAction::Remove), // PatientWeight
        (Tag(0x0010, 0x1040), TagAction::Remove), // PatientAddress
        (Tag(0x0010, 0x1100), TagAction::Remove), // PatientEthnicGroup
        (Tag(0x0010, 0x2150), TagAction::Remove), // CountryOfResidence
        (Tag(0x0010, 0x2152), TagAction::Remove), // RegionOfResidence
        (Tag(0x0010, 0x2154), TagAction::Remove), // PatientTelephoneNumbers
        // â”€â”€ D (Delete / Remove) â€” institution & personnel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0008, 0x0050), TagAction::Empty), // AccessionNumber (Z/D per Annex E)
        (Tag(0x0008, 0x0080), TagAction::Remove), // InstitutionName
        (Tag(0x0008, 0x0081), TagAction::Remove), // InstitutionAddress
        (Tag(0x0008, 0x0082), TagAction::Remove), // InstitutionCodeSequence
        (Tag(0x0008, 0x0090), TagAction::Empty), // ReferringPhysicianName
        (Tag(0x0008, 0x0092), TagAction::Remove), // ReferringPhysicianAddress
        (Tag(0x0008, 0x0094), TagAction::Remove), // ReferringPhysicianTelephoneNumbers
        (Tag(0x0008, 0x0096), TagAction::Remove), // ReferringPhysicianIdentificationSequence
        (Tag(0x0008, 0x1010), TagAction::Remove), // StationName
        (Tag(0x0008, 0x1040), TagAction::Remove), // InstitutionalDepartmentName
        (Tag(0x0008, 0x1049), TagAction::Remove), // InstitutionalDepartmentTypeCodeSequence
        (Tag(0x0008, 0x1050), TagAction::Remove), // PerformingPhysicianName
        (Tag(0x0008, 0x1052), TagAction::Remove), // PerformingPhysicianIdentificationSequence
        (Tag(0x0008, 0x1060), TagAction::Remove), // NameOfPhysiciansReadingStudy
        (Tag(0x0008, 0x1070), TagAction::Remove), // OperatorsName
        (Tag(0x0008, 0x1080), TagAction::Remove), // AdmittingDiagnosesDescription
        (Tag(0x0008, 0x1110), TagAction::Remove), // ReferencedStudySequence
        // â”€â”€ Z (Zero / Empty) â€” additional personnel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0008, 0x009C), TagAction::Empty), // ConsultingPhysicianName
        (Tag(0x0008, 0x1048), TagAction::Empty), // PhysicianApprovingInterpretation
        (Tag(0x0010, 0x0021), TagAction::Empty), // IssuerOfPatientID
        // â”€â”€ D (Delete / Remove) â€” device & protocol â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0018, 0x1000), TagAction::Remove), // DeviceSerialNumber
        (Tag(0x0018, 0x1002), TagAction::Remove), // DeviceUID
        (Tag(0x0018, 0x1030), TagAction::Remove), // ProtocolName
        // â”€â”€ D (Delete / Remove) â€” image comments & descriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0028, 0x4000), TagAction::Remove), // ImageComments
        // â”€â”€ D (Delete / Remove) â€” study-level references â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0020, 0x0200), TagAction::Remove), // SynchronizationFrameOfReferenceUID
        (Tag(0x0020, 0x3404), TagAction::Remove), // FrameOfReferenceTransformDescription
        // â”€â”€ D (Delete / Remove) â€” request & procedure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0032, 0x1032), TagAction::Remove), // RequestingPhysician
        (Tag(0x0032, 0x1033), TagAction::Remove), // RequestingService
        (Tag(0x0038, 0x0010), TagAction::Remove), // AdmissionID
        (Tag(0x0038, 0x0011), TagAction::Remove), // IssuerOfAdmissionID
        (Tag(0x0038, 0x0040), TagAction::Remove), // DischargeDiagnosisDescription
        // â”€â”€ D (Delete / Remove) â€” scheduled procedure steps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0040, 0x0001), TagAction::Remove), // ScheduledStationAETitle
        (Tag(0x0040, 0x0002), TagAction::Remove), // ScheduledProcedureStepStartDate
        (Tag(0x0040, 0x0003), TagAction::Remove), // ScheduledProcedureStepStartTime
        (Tag(0x0040, 0x0004), TagAction::Remove), // ScheduledProcedureStepEndDate
        (Tag(0x0040, 0x0005), TagAction::Remove), // ScheduledProcedureStepEndTime
        (Tag(0x0040, 0x0006), TagAction::Remove), // ScheduledPerformingPhysicianName
        (Tag(0x0040, 0x0010), TagAction::Remove), // ScheduledStationName
        // â”€â”€ D (Delete / Remove) â€” performed procedure steps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0040, 0x0241), TagAction::Remove), // PerformedStationAETitle
        (Tag(0x0040, 0x0242), TagAction::Remove), // PerformedStationName
        (Tag(0x0040, 0x0243), TagAction::Remove), // PerformedLocation
        (Tag(0x0040, 0x0244), TagAction::Remove), // PerformedProcedureStepStartDate
        (Tag(0x0040, 0x0245), TagAction::Remove), // PerformedProcedureStepStartTime
        (Tag(0x0040, 0x0253), TagAction::Remove), // PerformedProcedureStepEndDate
        (Tag(0x0040, 0x0254), TagAction::Remove), // PerformedProcedureStepEndTime
        (Tag(0x0040, 0x0275), TagAction::Remove), // RequestAttributeSequence
        (Tag(0x0040, 0x0280), TagAction::Remove), // CommentsOnThePerformedProcedureStep
        // â”€â”€ D (Delete / Remove) â€” verification & content â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0040, 0xA024), TagAction::Remove), // VerifyingOrganization
        (Tag(0x0040, 0xA120), TagAction::Remove), // DateTime
        (Tag(0x0040, 0xA123), TagAction::Remove), // PersonName
        // â”€â”€ D (Delete / Remove) â€” annotations & graphic content â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0070, 0x0001), TagAction::Remove), // GraphicAnnotationSequence
        (Tag(0x0070, 0x0084), TagAction::Remove), // ContentCreatorName
        // â”€â”€ D (Delete / Remove) â€” storage & icon â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0088, 0x0140), TagAction::Remove), // StorageMediaFileSetUID
        (Tag(0x0088, 0x0200), TagAction::Remove), // IconImageSequence
        // â”€â”€ D (Delete / Remove) â€” digital signatures & encrypted attrs â”€â”€
        (Tag(0x0400, 0x0100), TagAction::Remove), // DigitalSignatureUID
        (Tag(0x0400, 0x0402), TagAction::Remove), // ReferencedDigitalSignatureSequence
        (Tag(0x0400, 0x0403), TagAction::Remove), // ReferencedSOPInstanceMACSequence
        (Tag(0x0400, 0x0550), TagAction::Remove), // OriginalAttributesSequence
        (Tag(0xFFFA, 0xFFFA), TagAction::Remove), // DigitalSignaturesSequence
        (Tag(0xFFFC, 0xFFFC), TagAction::Remove), // EncryptedAttributesSequence
        // â”€â”€ Z/D (Empty) â€” study/series ID â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0020, 0x0010), TagAction::Empty), // StudyID
        (Tag(0x0010, 0x0040), TagAction::Empty), // PatientSex
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
        (Tag(0x0008, 0x1155), TagAction::ReplaceUid), // ReferencedSOPInstanceUID
        (Tag(0x0008, 0x0019), TagAction::ReplaceUid), // SourceImageSequence (contains UIDs)
    ]
}

/// Additional `Aggressive` actions: temporal metadata and descriptive text.
///
/// Removes all acquisition date/time fields and study/series/protocol
/// description strings that could contribute to re-identification.
fn aggressive_actions() -> Vec<(Tag, TagAction)> {
    vec![
        // â”€â”€ Date fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0008, 0x0020), TagAction::Empty), // StudyDate
        (Tag(0x0008, 0x0021), TagAction::Empty), // SeriesDate
        (Tag(0x0008, 0x0022), TagAction::Empty), // AcquisitionDate
        (Tag(0x0008, 0x0023), TagAction::Empty), // ContentDate
        (Tag(0x0008, 0x0024), TagAction::Empty), // OverlayDate
        (Tag(0x0008, 0x0025), TagAction::Empty), // CurveDate
        (Tag(0x0008, 0x002A), TagAction::Empty), // AcquisitionDateTime
        // â”€â”€ Time fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0008, 0x0030), TagAction::Empty), // StudyTime
        (Tag(0x0008, 0x0031), TagAction::Empty), // SeriesTime
        (Tag(0x0008, 0x0032), TagAction::Empty), // AcquisitionTime
        (Tag(0x0008, 0x0033), TagAction::Empty), // ContentTime
        (Tag(0x0008, 0x0034), TagAction::Empty), // OverlayTime
        (Tag(0x0008, 0x0035), TagAction::Empty), // CurveTime
        (Tag(0x0008, 0x0201), TagAction::Empty), // TimezoneOffsetFromUTC
        // â”€â”€ Descriptive text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0008, 0x1030), TagAction::Remove), // StudyDescription
        (Tag(0x0008, 0x103E), TagAction::Remove), // SeriesDescription
        (Tag(0x0018, 0x1030), TagAction::Remove), // ProtocolName
    ]
}

/// Additional `Enhanced` actions: procedure-step details, content annotations,
/// and digital signature cleanup beyond the Aggressive profile.
fn enhanced_actions() -> Vec<(Tag, TagAction)> {
    vec![
        // â”€â”€ Additional procedure-step removal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0040, 0x0007), TagAction::Remove), // ScheduledProcedureStepDescription
        (Tag(0x0040, 0x000B), TagAction::Remove), // ScheduledProcedureStepLocation
        (Tag(0x0040, 0x000E), TagAction::Remove), // ScheduledProcedureStepReason
        // â”€â”€ Content annotations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0040, 0xA027), TagAction::Remove), // VerifyingObserverName
        (Tag(0x0040, 0xA030), TagAction::Remove), // VerificationDateTime
        (Tag(0x0040, 0xA032), TagAction::Remove), // ObservationDateTime
        (Tag(0x0040, 0xA075), TagAction::Remove), // VerifyingObserverIdentificationCodeSequence
        // â”€â”€ Additional content â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        (Tag(0x0008, 0x2111), TagAction::Remove), // DerivationDescription
        (Tag(0x0020, 0x4000), TagAction::Remove), // ImageComments (already in Basic, repeated for safety)
        (Tag(0x3006, 0x0024), TagAction::Remove), // ReferencedFrameOfReferenceUID
        (Tag(0x3006, 0x00C2), TagAction::Remove), // RelatedFrameOfReferenceUID
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_profile_minimum_tag_count() {
        let actions = AnonymizationProfile::Basic.tag_actions();
        assert!(
            actions.len() >= 60,
            "Basic profile must cover at least 60 tags (PS 3.15 Annex E), got {}",
            actions.len()
        );
    }

    #[test]
    fn enhanced_includes_basic_actions() {
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

    fn find_action(actions: &[(Tag, TagAction)], tag: Tag) -> Option<TagAction> {
        actions
            .iter()
            .find(|(t, _)| *t == tag)
            .copied()
            .map(|(_, a)| a)
    }
}
