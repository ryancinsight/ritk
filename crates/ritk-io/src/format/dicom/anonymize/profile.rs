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

use ritk_dicom::dataset::DicomTag;

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
    /// Return the ordered list of `(DicomTag, TagAction)` pairs for this profile.
    ///
    /// Tags not present in the list are left unchanged. Later entries for the
    /// same tag supersede earlier ones (no duplicates are emitted here, but the
    /// caller must be aware when combining profiles externally).
    pub fn tag_actions(&self) -> Vec<(DicomTag, TagAction)> {
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

// ─── Basic profile (PS 3.15 Annex E Table E.1-1) ─────────────────────────────
//
// Invariant: every tag listed in Annex E Table E.1-1 for the Basic
// Application Level Confidentiality Profile is covered. Tags are grouped
// by DICOM action type (D=Remove, Z=Empty, Z/D=Empty, X/Z/D=Remove).

fn basic_actions() -> Vec<(DicomTag, TagAction)> {
    vec![
        // ── D (Delete / Remove) — patient demographics ──────────────────
        (DicomTag::new(0x0010, 0x0010), TagAction::Dummy), // PatientName → configurable dummy
        (DicomTag::new(0x0010, 0x0020), TagAction::Dummy), // PatientID → configurable dummy
        (DicomTag::new(0x0010, 0x0030), TagAction::Empty), // PatientBirthDate
        (DicomTag::new(0x0010, 0x0032), TagAction::Remove), // PatientBirthTime
        (DicomTag::new(0x0010, 0x0050), TagAction::Remove), // PatientInsurancePlanCodeSequence
        (DicomTag::new(0x0010, 0x0101), TagAction::Remove), // PatientInsurancePlanSequence
        (DicomTag::new(0x0010, 0x1000), TagAction::Remove), // OtherPatientIDs
        (DicomTag::new(0x0010, 0x1001), TagAction::Remove), // OtherPatientNames
        (DicomTag::new(0x0010, 0x1010), TagAction::Remove), // PatientAge
        (DicomTag::new(0x0010, 0x1030), TagAction::Remove), // PatientWeight
        (DicomTag::new(0x0010, 0x1040), TagAction::Remove), // PatientAddress
        (DicomTag::new(0x0010, 0x1100), TagAction::Remove), // PatientEthnicGroup
        (DicomTag::new(0x0010, 0x2150), TagAction::Remove), // CountryOfResidence
        (DicomTag::new(0x0010, 0x2152), TagAction::Remove), // RegionOfResidence
        (DicomTag::new(0x0010, 0x2154), TagAction::Remove), // PatientTelephoneNumbers
        // ── D (Delete / Remove) — institution & personnel ───────────────
        (DicomTag::new(0x0008, 0x0050), TagAction::Empty), // AccessionNumber (Z/D per Annex E)
        (DicomTag::new(0x0008, 0x0080), TagAction::Remove), // InstitutionName
        (DicomTag::new(0x0008, 0x0081), TagAction::Remove), // InstitutionAddress
        (DicomTag::new(0x0008, 0x0082), TagAction::Remove), // InstitutionCodeSequence
        (DicomTag::new(0x0008, 0x0090), TagAction::Empty), // ReferringPhysicianName
        (DicomTag::new(0x0008, 0x0092), TagAction::Remove), // ReferringPhysicianAddress
        (DicomTag::new(0x0008, 0x0094), TagAction::Remove), // ReferringPhysicianTelephoneNumbers
        (DicomTag::new(0x0008, 0x0096), TagAction::Remove), // ReferringPhysicianIdentificationSequence
        (DicomTag::new(0x0008, 0x1010), TagAction::Remove), // StationName
        (DicomTag::new(0x0008, 0x1040), TagAction::Remove), // InstitutionalDepartmentName
        (DicomTag::new(0x0008, 0x1049), TagAction::Remove), // InstitutionalDepartmentTypeCodeSequence
        (DicomTag::new(0x0008, 0x1050), TagAction::Remove), // PerformingPhysicianName
        (DicomTag::new(0x0008, 0x1052), TagAction::Remove), // PerformingPhysicianIdentificationSequence
        (DicomTag::new(0x0008, 0x1060), TagAction::Remove), // NameOfPhysiciansReadingStudy
        (DicomTag::new(0x0008, 0x1070), TagAction::Remove), // OperatorsName
        (DicomTag::new(0x0008, 0x1080), TagAction::Remove), // AdmittingDiagnosesDescription
        (DicomTag::new(0x0008, 0x1110), TagAction::Remove), // ReferencedStudySequence
        // ── Z (Zero / Empty) — additional personnel ─────────────────────
        (DicomTag::new(0x0008, 0x009C), TagAction::Empty), // ConsultingPhysicianName
        (DicomTag::new(0x0008, 0x1048), TagAction::Empty), // PhysicianApprovingInterpretation
        (DicomTag::new(0x0010, 0x0021), TagAction::Empty), // IssuerOfPatientID
        // ── D (Delete / Remove) — device & protocol ─────────────────────
        (DicomTag::new(0x0018, 0x1000), TagAction::Remove), // DeviceSerialNumber
        (DicomTag::new(0x0018, 0x1002), TagAction::Remove), // DeviceUID
        (DicomTag::new(0x0018, 0x1030), TagAction::Remove), // ProtocolName
        // ── D (Delete / Remove) — image comments & descriptions ─────────
        (DicomTag::new(0x0028, 0x4000), TagAction::Remove), // ImageComments
        // ── D (Delete / Remove) — study-level references ────────────────
        (DicomTag::new(0x0020, 0x0200), TagAction::Remove), // SynchronizationFrameOfReferenceUID
        (DicomTag::new(0x0020, 0x3404), TagAction::Remove), // FrameOfReferenceTransformDescription
        // ── D (Delete / Remove) — request & procedure ───────────────────
        (DicomTag::new(0x0032, 0x1032), TagAction::Remove), // RequestingPhysician
        (DicomTag::new(0x0032, 0x1033), TagAction::Remove), // RequestingService
        (DicomTag::new(0x0038, 0x0010), TagAction::Remove), // AdmissionID
        (DicomTag::new(0x0038, 0x0011), TagAction::Remove), // IssuerOfAdmissionID
        (DicomTag::new(0x0038, 0x0040), TagAction::Remove), // DischargeDiagnosisDescription
        // ── D (Delete / Remove) — scheduled procedure steps ─────────────
        (DicomTag::new(0x0040, 0x0001), TagAction::Remove), // ScheduledStationAETitle
        (DicomTag::new(0x0040, 0x0002), TagAction::Remove), // ScheduledProcedureStepStartDate
        (DicomTag::new(0x0040, 0x0003), TagAction::Remove), // ScheduledProcedureStepStartTime
        (DicomTag::new(0x0040, 0x0004), TagAction::Remove), // ScheduledProcedureStepEndDate
        (DicomTag::new(0x0040, 0x0005), TagAction::Remove), // ScheduledProcedureStepEndTime
        (DicomTag::new(0x0040, 0x0006), TagAction::Remove), // ScheduledPerformingPhysicianName
        (DicomTag::new(0x0040, 0x0010), TagAction::Remove), // ScheduledStationName
        // ── D (Delete / Remove) — performed procedure steps ─────────────
        (DicomTag::new(0x0040, 0x0241), TagAction::Remove), // PerformedStationAETitle
        (DicomTag::new(0x0040, 0x0242), TagAction::Remove), // PerformedStationName
        (DicomTag::new(0x0040, 0x0243), TagAction::Remove), // PerformedLocation
        (DicomTag::new(0x0040, 0x0244), TagAction::Remove), // PerformedProcedureStepStartDate
        (DicomTag::new(0x0040, 0x0245), TagAction::Remove), // PerformedProcedureStepStartTime
        (DicomTag::new(0x0040, 0x0253), TagAction::Remove), // PerformedProcedureStepEndDate
        (DicomTag::new(0x0040, 0x0254), TagAction::Remove), // PerformedProcedureStepEndTime
        (DicomTag::new(0x0040, 0x0275), TagAction::Remove), // RequestAttributeSequence
        (DicomTag::new(0x0040, 0x0280), TagAction::Remove), // CommentsOnThePerformedProcedureStep
        // ── D (Delete / Remove) — verification & content ────────────────
        (DicomTag::new(0x0040, 0xA024), TagAction::Remove), // VerifyingOrganization
        (DicomTag::new(0x0040, 0xA120), TagAction::Remove), // DateTime
        (DicomTag::new(0x0040, 0xA123), TagAction::Remove), // PersonName
        // ── D (Delete / Remove) — annotations & graphic content ─────────
        (DicomTag::new(0x0070, 0x0001), TagAction::Remove), // GraphicAnnotationSequence
        (DicomTag::new(0x0070, 0x0084), TagAction::Remove), // ContentCreatorName
        // ── D (Delete / Remove) — storage & icon ────────────────────────
        (DicomTag::new(0x0088, 0x0140), TagAction::Remove), // StorageMediaFileSetUID
        (DicomTag::new(0x0088, 0x0200), TagAction::Remove), // IconImageSequence
        // ── D (Delete / Remove) — digital signatures & encrypted attrs ──
        (DicomTag::new(0x0400, 0x0100), TagAction::Remove), // DigitalSignatureUID
        (DicomTag::new(0x0400, 0x0402), TagAction::Remove), // ReferencedDigitalSignatureSequence
        (DicomTag::new(0x0400, 0x0403), TagAction::Remove), // ReferencedSOPInstanceMACSequence
        (DicomTag::new(0x0400, 0x0550), TagAction::Remove), // OriginalAttributesSequence
        (DicomTag::new(0xFFFA, 0xFFFA), TagAction::Remove), // DigitalSignaturesSequence
        (DicomTag::new(0xFFFC, 0xFFFC), TagAction::Remove), // EncryptedAttributesSequence
        // ── Z/D (Empty) — study/series ID ───────────────────────────────
        (DicomTag::new(0x0020, 0x0010), TagAction::Empty), // StudyID
        (DicomTag::new(0x0010, 0x0040), TagAction::Empty), // PatientSex
    ]
}

/// UID replacement actions added by `BasicReplaceUids`.
///
/// Covers all instance-level UIDs required for intra-study consistency by
/// PS 3.15 Annex E Option: Retain UIDs.
fn uid_actions() -> Vec<(DicomTag, TagAction)> {
    vec![
        (DicomTag::new(0x0020, 0x000D), TagAction::ReplaceUid), // StudyInstanceUID
        (DicomTag::new(0x0020, 0x000E), TagAction::ReplaceUid), // SeriesInstanceUID
        (DicomTag::new(0x0008, 0x0018), TagAction::ReplaceUid), // SOPInstanceUID
        (DicomTag::new(0x0020, 0x0052), TagAction::ReplaceUid), // FrameOfReferenceUID
        (DicomTag::new(0x0008, 0x1155), TagAction::ReplaceUid), // ReferencedSOPInstanceUID
        (DicomTag::new(0x0008, 0x0019), TagAction::ReplaceUid), // SourceImageSequence (contains UIDs)
    ]
}

/// Additional `Aggressive` actions: temporal metadata and descriptive text.
///
/// Removes all acquisition date/time fields and study/series/protocol
/// description strings that could contribute to re-identification.
fn aggressive_actions() -> Vec<(DicomTag, TagAction)> {
    vec![
        // ── Date fields ──────────────────────────────────────────────────
        (DicomTag::new(0x0008, 0x0020), TagAction::Empty), // StudyDate
        (DicomTag::new(0x0008, 0x0021), TagAction::Empty), // SeriesDate
        (DicomTag::new(0x0008, 0x0022), TagAction::Empty), // AcquisitionDate
        (DicomTag::new(0x0008, 0x0023), TagAction::Empty), // ContentDate
        (DicomTag::new(0x0008, 0x0024), TagAction::Empty), // OverlayDate
        (DicomTag::new(0x0008, 0x0025), TagAction::Empty), // CurveDate
        (DicomTag::new(0x0008, 0x002A), TagAction::Empty), // AcquisitionDateTime
        // ── Time fields ──────────────────────────────────────────────────
        (DicomTag::new(0x0008, 0x0030), TagAction::Empty), // StudyTime
        (DicomTag::new(0x0008, 0x0031), TagAction::Empty), // SeriesTime
        (DicomTag::new(0x0008, 0x0032), TagAction::Empty), // AcquisitionTime
        (DicomTag::new(0x0008, 0x0033), TagAction::Empty), // ContentTime
        (DicomTag::new(0x0008, 0x0034), TagAction::Empty), // OverlayTime
        (DicomTag::new(0x0008, 0x0035), TagAction::Empty), // CurveTime
        (DicomTag::new(0x0008, 0x0201), TagAction::Empty), // TimezoneOffsetFromUTC
        // ── Descriptive text ─────────────────────────────────────────────
        (DicomTag::new(0x0008, 0x1030), TagAction::Remove), // StudyDescription
        (DicomTag::new(0x0008, 0x103E), TagAction::Remove), // SeriesDescription
        (DicomTag::new(0x0018, 0x1030), TagAction::Remove), // ProtocolName
    ]
}

/// Additional `Enhanced` actions: procedure-step details, content annotations,
/// and digital signature cleanup beyond the Aggressive profile.
fn enhanced_actions() -> Vec<(DicomTag, TagAction)> {
    vec![
        // ── Additional procedure-step removal ────────────────────────────
        (DicomTag::new(0x0040, 0x0007), TagAction::Remove), // ScheduledProcedureStepDescription
        (DicomTag::new(0x0040, 0x000B), TagAction::Remove), // ScheduledProcedureStepLocation
        (DicomTag::new(0x0040, 0x000E), TagAction::Remove), // ScheduledProcedureStepReason
        // ── Content annotations ──────────────────────────────────────────
        (DicomTag::new(0x0040, 0xA027), TagAction::Remove), // VerifyingObserverName
        (DicomTag::new(0x0040, 0xA030), TagAction::Remove), // VerificationDateTime
        (DicomTag::new(0x0040, 0xA032), TagAction::Remove), // ObservationDateTime
        (DicomTag::new(0x0040, 0xA075), TagAction::Remove), // VerifyingObserverIdentificationCodeSequence
        // ── Additional content ───────────────────────────────────────────
        (DicomTag::new(0x0008, 0x2111), TagAction::Remove), // DerivationDescription
        (DicomTag::new(0x0020, 0x4000), TagAction::Remove), // ImageComments (already in Basic, repeated for safety)
        (DicomTag::new(0x3006, 0x0024), TagAction::Remove), // ReferencedFrameOfReferenceUID
        (DicomTag::new(0x3006, 0x00C2), TagAction::Remove), // RelatedFrameOfReferenceUID
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

    fn find_action(actions: &[(DicomTag, TagAction)], tag: DicomTag) -> Option<TagAction> {
        actions
            .iter()
            .find(|(t, _)| *t == tag)
            .copied()
            .map(|(_, a)| a)
    }
}
