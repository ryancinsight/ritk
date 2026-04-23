//! DICOM SOP Class policy -- accept/reject image-bearing SOP classes.
//!
//! # Mathematical Specification / Invariants
//!
//! Let U = set of all DICOM SOP Class UIDs.
//! Partition U into IMAGE_UIDS union NONIMAGE_UIDS union UNKNOWN.
//!
//! **Theorem (policy correctness):** A DICOM file is loadable as a pixel image
//! iff its SOP Class UID is in IMAGE_UIDS.
//!
//! **Rejection invariant:** Any file whose SOP Class UID is classified as
//! non-image MUST be excluded from series assembly with a deterministic error
//! describing the specific class encountered.
//!
//! # References
//! - DICOM PS3.4 SB -- Storage SOP Classes.
//! - DICOM PS3.6 -- Data Dictionary (UID registry).

/// Classification of a DICOM SOP Class UID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SopClassKind {
    // Image-bearing SOP classes
    /// CT Image Storage (1.2.840.10008.5.1.4.1.1.2)
    CtImageStorage,
    /// Enhanced CT Image Storage (1.2.840.10008.5.1.4.1.1.2.1)
    EnhancedCtImageStorage,
    /// Legacy Converted Enhanced CT Image Storage (1.2.840.10008.5.1.4.1.1.2.2)
    LegacyConvertedEnhancedCtImageStorage,
    /// MR Image Storage (1.2.840.10008.5.1.4.1.1.4)
    MrImageStorage,
    /// Enhanced MR Image Storage (1.2.840.10008.5.1.4.1.1.4.1)
    EnhancedMrImageStorage,
    /// Legacy Converted Enhanced MR Image Storage (1.2.840.10008.5.1.4.1.1.4.4)
    LegacyConvertedEnhancedMrImageStorage,
    /// PET Image Storage (1.2.840.10008.5.1.4.1.1.128)
    PetImageStorage,
    /// Enhanced PET Image Storage (1.2.840.10008.5.1.4.1.1.130)
    EnhancedPetImageStorage,
    /// Legacy Converted Enhanced PET Image Storage (1.2.840.10008.5.1.4.1.1.128.1)
    LegacyConvertedEnhancedPetImageStorage,
    /// CR Image Storage (1.2.840.10008.5.1.4.1.1.1)
    CrImageStorage,
    /// Digital X-Ray Image Storage -- For Presentation (1.2.840.10008.5.1.4.1.1.1.1)
    DigitalXRayImageStorageForPresentation,
    /// Digital X-Ray Image Storage -- For Processing (1.2.840.10008.5.1.4.1.1.1.1.1)
    DigitalXRayImageStorageForProcessing,
    /// Digital Mammography X-Ray Image Storage -- For Presentation (1.2.840.10008.5.1.4.1.1.1.2)
    DigitalMammographyXRayImageStorageForPresentation,
    /// Ultrasound Image Storage (1.2.840.10008.5.1.4.1.1.6.1)
    UltrasoundImageStorage,
    /// Ultrasound Multi-frame Image Storage (1.2.840.10008.5.1.4.1.1.3.1)
    UltrasoundMultiFrameImageStorage,
    /// NM Image Storage (1.2.840.10008.5.1.4.1.1.20)
    NuclearMedicineImageStorage,
    /// Secondary Capture Image Storage (1.2.840.10008.5.1.4.1.1.7)
    SecondaryCaptureImageStorage,
    /// Multi-frame Grayscale Byte Secondary Capture Image Storage (1.2.840.10008.5.1.4.1.1.7.2)
    MultiFrameGrayscaleByteSecondaryCaptureImageStorage,
    /// Multi-frame Grayscale Word Secondary Capture Image Storage (1.2.840.10008.5.1.4.1.1.7.3)
    MultiFrameGrayscaleWordSecondaryCaptureImageStorage,
    /// Multi-frame True Color Secondary Capture Image Storage (1.2.840.10008.5.1.4.1.1.7.4)
    MultiFrameTrueColorSecondaryCaptureImageStorage,
    /// X-Ray Angiographic Image Storage (1.2.840.10008.5.1.4.1.1.12.1)
    XRayAngiographicImageStorage,
    /// X-Ray Radiofluoroscopic Image Storage (1.2.840.10008.5.1.4.1.1.12.2)
    XRayRadioFluoroscopicImageStorage,
    /// Breast Tomosynthesis Image Storage (1.2.840.10008.5.1.4.1.1.13.1.3)
    BreastTomosynthesisImageStorage,
    /// VL Endoscopic Image Storage (1.2.840.10008.5.1.4.1.1.77.1.1)
    VlEndoscopicImageStorage,
    /// VL Microscopic Image Storage (1.2.840.10008.5.1.4.1.1.77.1.2)
    VlMicroscopicImageStorage,
    /// VL Slide-Coordinates Microscopic Image Storage (1.2.840.10008.5.1.4.1.1.77.1.3)
    VlSlideCoordinatesMicroscopicImageStorage,
    /// VL Photographic Image Storage (1.2.840.10008.5.1.4.1.1.77.1.4)
    VlPhotographicImageStorage,
    /// Ophthalmic Photography 8-bit Image Storage (1.2.840.10008.5.1.4.1.1.77.1.5.1)
    OphthalmicPhotography8BitImageStorage,
    /// Ophthalmic Photography 16-bit Image Storage (1.2.840.10008.5.1.4.1.1.77.1.5.2)
    OphthalmicPhotography16BitImageStorage,
    /// Segmentation Storage (1.2.840.10008.5.1.4.1.1.66.4)
    SegmentationStorage,
    /// Parametric Map Storage (1.2.840.10008.5.1.4.1.1.30)
    ParametricMapStorage,

    // Non-image SOP classes
    /// RT Structure Set Storage (1.2.840.10008.5.1.4.1.1.481.3)
    RtStructureSetStorage,
    /// RT Plan Storage (1.2.840.10008.5.1.4.1.1.481.5)
    RtPlanStorage,
    /// RT Dose Storage (1.2.840.10008.5.1.4.1.1.481.2)
    RtDoseStorage,
    /// Basic Text SR Storage (1.2.840.10008.5.1.4.1.1.88.11)
    BasicTextSrStorage,
    /// Enhanced SR Storage (1.2.840.10008.5.1.4.1.1.88.22)
    EnhancedSrStorage,
    /// Comprehensive SR Storage (1.2.840.10008.5.1.4.1.1.88.33)
    ComprehensiveSrStorage,
    /// Comprehensive 3D SR Storage (1.2.840.10008.5.1.4.1.1.88.34)
    Comprehensive3DSrStorage,
    /// Key Object Selection Document Storage (1.2.840.10008.5.1.4.1.1.88.59)
    KeyObjectSelectionDocumentStorage,
    /// Grayscale Softcopy Presentation State Storage (1.2.840.10008.5.1.4.1.1.11.1)
    GrayscaleSoftcopyPresentationStateStorage,
    /// Color Softcopy Presentation State Storage (1.2.840.10008.5.1.4.1.1.11.2)
    ColorSoftcopyPresentationStateStorage,
    /// Modality Performed Procedure Step (1.2.840.10008.3.1.2.3.3)
    ModalityPerformedProcedureStep,
    /// 12-lead ECG Waveform Storage (1.2.840.10008.5.1.4.1.1.9.1.1)
    TwelveLeadEcgWaveformStorage,
    /// Ambulatory ECG Waveform Storage (1.2.840.10008.5.1.4.1.1.9.1.2)
    AmbulatoryEcgWaveformStorage,
    /// Hemodynamic Waveform Storage (1.2.840.10008.5.1.4.1.1.9.2.1)
    HemodynamicWaveformStorage,
    /// Encapsulated PDF Storage (1.2.840.10008.5.1.4.1.1.104.1)
    EncapsulatedPdfStorage,
    /// Fiducials Storage (1.2.840.10008.5.1.4.1.1.66.2)
    FiducialsStorage,
    /// Stereometric Relationship Storage (1.2.840.10008.5.1.4.1.1.77.1.5.3)
    StereometricRelationshipStorage,
    /// Macular Grid Thickness and Volume Report Storage (1.2.840.10008.5.1.4.1.1.79.1)
    MacularGridThicknessAndVolumeReportStorage,
    /// Ophthalmic Axial Measurements Storage (1.2.840.10008.5.1.4.1.1.78.7)
    OphthalmicAxialMeasurementsStorage,

    // Unknown
    /// UID not in the RITK classification table.
    Other(String),
}

impl SopClassKind {
    /// Returns `true` iff this SOP class carries loadable 2D/3D pixel data.
    ///
    /// Image-bearing classes have a Pixel Data element (7FE0,0010) with
    /// rows x columns x frames of scalar intensity values decodable as
    /// a 3-D image tensor. Non-image classes MUST NOT be passed to the
    /// pixel-loading path.
    pub fn is_image_storage(&self) -> bool {
        matches!(
            self,
            SopClassKind::CtImageStorage
                | SopClassKind::EnhancedCtImageStorage
                | SopClassKind::LegacyConvertedEnhancedCtImageStorage
                | SopClassKind::MrImageStorage
                | SopClassKind::EnhancedMrImageStorage
                | SopClassKind::LegacyConvertedEnhancedMrImageStorage
                | SopClassKind::PetImageStorage
                | SopClassKind::EnhancedPetImageStorage
                | SopClassKind::LegacyConvertedEnhancedPetImageStorage
                | SopClassKind::CrImageStorage
                | SopClassKind::DigitalXRayImageStorageForPresentation
                | SopClassKind::DigitalXRayImageStorageForProcessing
                | SopClassKind::DigitalMammographyXRayImageStorageForPresentation
                | SopClassKind::UltrasoundImageStorage
                | SopClassKind::UltrasoundMultiFrameImageStorage
                | SopClassKind::NuclearMedicineImageStorage
                | SopClassKind::SecondaryCaptureImageStorage
                | SopClassKind::MultiFrameGrayscaleByteSecondaryCaptureImageStorage
                | SopClassKind::MultiFrameGrayscaleWordSecondaryCaptureImageStorage
                | SopClassKind::MultiFrameTrueColorSecondaryCaptureImageStorage
                | SopClassKind::XRayAngiographicImageStorage
                | SopClassKind::XRayRadioFluoroscopicImageStorage
                | SopClassKind::BreastTomosynthesisImageStorage
                | SopClassKind::VlEndoscopicImageStorage
                | SopClassKind::VlMicroscopicImageStorage
                | SopClassKind::VlSlideCoordinatesMicroscopicImageStorage
                | SopClassKind::VlPhotographicImageStorage
                | SopClassKind::OphthalmicPhotography8BitImageStorage
                | SopClassKind::OphthalmicPhotography16BitImageStorage
                | SopClassKind::SegmentationStorage
                | SopClassKind::ParametricMapStorage
        )
    }
}

/// Classify a DICOM SOP Class UID string into a [`SopClassKind`].
///
/// UIDs are matched against the RITK classification table. Any UID not in
/// the table is classified as [`SopClassKind::Other`] carrying the original
/// string. This is a pure, total function with no side effects.
pub fn classify_sop_class(uid: &str) -> SopClassKind {
    let uid = uid.trim();
    match uid {
        // Image SOP classes
        "1.2.840.10008.5.1.4.1.1.2"     => SopClassKind::CtImageStorage,
        "1.2.840.10008.5.1.4.1.1.2.1"   => SopClassKind::EnhancedCtImageStorage,
        "1.2.840.10008.5.1.4.1.1.2.2"   => SopClassKind::LegacyConvertedEnhancedCtImageStorage,
        "1.2.840.10008.5.1.4.1.1.4"     => SopClassKind::MrImageStorage,
        "1.2.840.10008.5.1.4.1.1.4.1"   => SopClassKind::EnhancedMrImageStorage,
        "1.2.840.10008.5.1.4.1.1.4.4"   => SopClassKind::LegacyConvertedEnhancedMrImageStorage,
        "1.2.840.10008.5.1.4.1.1.128"   => SopClassKind::PetImageStorage,
        "1.2.840.10008.5.1.4.1.1.130"   => SopClassKind::EnhancedPetImageStorage,
        "1.2.840.10008.5.1.4.1.1.128.1" => SopClassKind::LegacyConvertedEnhancedPetImageStorage,
        "1.2.840.10008.5.1.4.1.1.1"     => SopClassKind::CrImageStorage,
        "1.2.840.10008.5.1.4.1.1.1.1"   => SopClassKind::DigitalXRayImageStorageForPresentation,
        "1.2.840.10008.5.1.4.1.1.1.1.1" => SopClassKind::DigitalXRayImageStorageForProcessing,
        "1.2.840.10008.5.1.4.1.1.1.2"   => SopClassKind::DigitalMammographyXRayImageStorageForPresentation,
        "1.2.840.10008.5.1.4.1.1.6.1"   => SopClassKind::UltrasoundImageStorage,
        "1.2.840.10008.5.1.4.1.1.3.1"   => SopClassKind::UltrasoundMultiFrameImageStorage,
        "1.2.840.10008.5.1.4.1.1.20"    => SopClassKind::NuclearMedicineImageStorage,
        "1.2.840.10008.5.1.4.1.1.7"     => SopClassKind::SecondaryCaptureImageStorage,
        "1.2.840.10008.5.1.4.1.1.7.2"   => SopClassKind::MultiFrameGrayscaleByteSecondaryCaptureImageStorage,
        "1.2.840.10008.5.1.4.1.1.7.3"   => SopClassKind::MultiFrameGrayscaleWordSecondaryCaptureImageStorage,
        "1.2.840.10008.5.1.4.1.1.7.4"   => SopClassKind::MultiFrameTrueColorSecondaryCaptureImageStorage,
        "1.2.840.10008.5.1.4.1.1.12.1"  => SopClassKind::XRayAngiographicImageStorage,
        "1.2.840.10008.5.1.4.1.1.12.2"  => SopClassKind::XRayRadioFluoroscopicImageStorage,
        "1.2.840.10008.5.1.4.1.1.13.1.3" => SopClassKind::BreastTomosynthesisImageStorage,
        "1.2.840.10008.5.1.4.1.1.77.1.1" => SopClassKind::VlEndoscopicImageStorage,
        "1.2.840.10008.5.1.4.1.1.77.1.2" => SopClassKind::VlMicroscopicImageStorage,
        "1.2.840.10008.5.1.4.1.1.77.1.3" => SopClassKind::VlSlideCoordinatesMicroscopicImageStorage,
        "1.2.840.10008.5.1.4.1.1.77.1.4" => SopClassKind::VlPhotographicImageStorage,
        "1.2.840.10008.5.1.4.1.1.77.1.5.1" => SopClassKind::OphthalmicPhotography8BitImageStorage,
        "1.2.840.10008.5.1.4.1.1.77.1.5.2" => SopClassKind::OphthalmicPhotography16BitImageStorage,
        "1.2.840.10008.5.1.4.1.1.66.4"  => SopClassKind::SegmentationStorage,
        "1.2.840.10008.5.1.4.1.1.30"    => SopClassKind::ParametricMapStorage,
        // Non-image SOP classes
        "1.2.840.10008.5.1.4.1.1.481.3" => SopClassKind::RtStructureSetStorage,
        "1.2.840.10008.5.1.4.1.1.481.5" => SopClassKind::RtPlanStorage,
        "1.2.840.10008.5.1.4.1.1.481.2" => SopClassKind::RtDoseStorage,
        "1.2.840.10008.5.1.4.1.1.88.11" => SopClassKind::BasicTextSrStorage,
        "1.2.840.10008.5.1.4.1.1.88.22" => SopClassKind::EnhancedSrStorage,
        "1.2.840.10008.5.1.4.1.1.88.33" => SopClassKind::ComprehensiveSrStorage,
        "1.2.840.10008.5.1.4.1.1.88.34" => SopClassKind::Comprehensive3DSrStorage,
        "1.2.840.10008.5.1.4.1.1.88.59" => SopClassKind::KeyObjectSelectionDocumentStorage,
        "1.2.840.10008.5.1.4.1.1.11.1"  => SopClassKind::GrayscaleSoftcopyPresentationStateStorage,
        "1.2.840.10008.5.1.4.1.1.11.2"  => SopClassKind::ColorSoftcopyPresentationStateStorage,
        "1.2.840.10008.3.1.2.3.3"       => SopClassKind::ModalityPerformedProcedureStep,
        "1.2.840.10008.5.1.4.1.1.9.1.1" => SopClassKind::TwelveLeadEcgWaveformStorage,
        "1.2.840.10008.5.1.4.1.1.9.1.2" => SopClassKind::AmbulatoryEcgWaveformStorage,
        "1.2.840.10008.5.1.4.1.1.9.2.1" => SopClassKind::HemodynamicWaveformStorage,
        "1.2.840.10008.5.1.4.1.1.104.1" => SopClassKind::EncapsulatedPdfStorage,
        "1.2.840.10008.5.1.4.1.1.66.2"  => SopClassKind::FiducialsStorage,
        "1.2.840.10008.5.1.4.1.1.77.1.5.3" => SopClassKind::StereometricRelationshipStorage,
        "1.2.840.10008.5.1.4.1.1.79.1"  => SopClassKind::MacularGridThicknessAndVolumeReportStorage,
        "1.2.840.10008.5.1.4.1.1.78.7"  => SopClassKind::OphthalmicAxialMeasurementsStorage,
        other                            => SopClassKind::Other(other.to_string()),
    }
}

/// Returns `true` iff `uid` refers to a pixel-bearing image SOP class.
///
/// Equivalent to `classify_sop_class(uid).is_image_storage()`.
/// Unknown UIDs (classified as `Other`) return `false`.
pub fn is_image_sop_class(uid: &str) -> bool {
    classify_sop_class(uid).is_image_storage()
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    // Positive: image SOP classes

    #[test]
    fn test_ct_is_image_storage() {
        assert_eq!(classify_sop_class("1.2.840.10008.5.1.4.1.1.2"), SopClassKind::CtImageStorage);
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.2"));
    }

    #[test]
    fn test_mr_is_image_storage() {
        assert_eq!(classify_sop_class("1.2.840.10008.5.1.4.1.1.4"), SopClassKind::MrImageStorage);
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.4"));
    }

    #[test]
    fn test_pet_is_image_storage() {
        assert_eq!(classify_sop_class("1.2.840.10008.5.1.4.1.1.128"), SopClassKind::PetImageStorage);
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.128"));
    }

    #[test]
    fn test_enhanced_ct_is_image_storage() {
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.2.1"));
    }

    #[test]
    fn test_enhanced_mr_is_image_storage() {
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.4.1"));
    }

    #[test]
    fn test_secondary_capture_is_image_storage() {
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.7"));
    }

    #[test]
    fn test_ultrasound_is_image_storage() {
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.6.1"));
    }

    #[test]
    fn test_nm_is_image_storage() {
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.20"));
    }

    #[test]
    fn test_segmentation_is_image_storage() {
        assert!(is_image_sop_class("1.2.840.10008.5.1.4.1.1.66.4"));
    }

    // Negative: non-image SOP classes

    #[test]
    fn test_rtstruct_is_not_image_storage() {
        assert_eq!(classify_sop_class("1.2.840.10008.5.1.4.1.1.481.3"), SopClassKind::RtStructureSetStorage);
        assert!(!is_image_sop_class("1.2.840.10008.5.1.4.1.1.481.3"));
    }

    #[test]
    fn test_rtplan_is_not_image_storage() {
        assert_eq!(classify_sop_class("1.2.840.10008.5.1.4.1.1.481.5"), SopClassKind::RtPlanStorage);
        assert!(!is_image_sop_class("1.2.840.10008.5.1.4.1.1.481.5"));
    }

    #[test]
    fn test_rtdose_is_not_image_storage() {
        assert!(!is_image_sop_class("1.2.840.10008.5.1.4.1.1.481.2"));
    }

    #[test]
    fn test_basic_text_sr_is_not_image_storage() {
        assert_eq!(classify_sop_class("1.2.840.10008.5.1.4.1.1.88.11"), SopClassKind::BasicTextSrStorage);
        assert!(!is_image_sop_class("1.2.840.10008.5.1.4.1.1.88.11"));
    }

    #[test]
    fn test_enhanced_sr_is_not_image_storage() {
        assert!(!is_image_sop_class("1.2.840.10008.5.1.4.1.1.88.22"));
    }

    #[test]
    fn test_grayscale_pr_is_not_image_storage() {
        assert!(!is_image_sop_class("1.2.840.10008.5.1.4.1.1.11.1"));
    }

    #[test]
    fn test_waveform_ecg_is_not_image_storage() {
        assert!(!is_image_sop_class("1.2.840.10008.5.1.4.1.1.9.1.1"));
    }

    #[test]
    fn test_encapsulated_pdf_is_not_image_storage() {
        assert!(!is_image_sop_class("1.2.840.10008.5.1.4.1.1.104.1"));
    }

    // Boundary: unknown and empty UIDs

    #[test]
    fn test_unknown_uid_classified_as_other() {
        let kind = classify_sop_class("9.9.9.9.9.9.9");
        assert!(matches!(kind, SopClassKind::Other(_)));
        assert!(!is_image_sop_class("9.9.9.9.9.9.9"));
    }

    #[test]
    fn test_empty_uid_classified_as_other() {
        let kind = classify_sop_class("");
        assert!(matches!(kind, SopClassKind::Other(_)));
        assert!(!is_image_sop_class(""));
    }

    #[test]
    fn test_uid_with_leading_trailing_whitespace() {
        assert!(is_image_sop_class("  1.2.840.10008.5.1.4.1.1.2  "));
        assert!(!is_image_sop_class("  1.2.840.10008.5.1.4.1.1.481.3  "));
    }

    // Exhaustive partition check

    #[test]
    fn test_all_known_image_uids_return_true() {
        let image_uids = [
            "1.2.840.10008.5.1.4.1.1.2",
            "1.2.840.10008.5.1.4.1.1.2.1",
            "1.2.840.10008.5.1.4.1.1.2.2",
            "1.2.840.10008.5.1.4.1.1.4",
            "1.2.840.10008.5.1.4.1.1.4.1",
            "1.2.840.10008.5.1.4.1.1.4.4",
            "1.2.840.10008.5.1.4.1.1.128",
            "1.2.840.10008.5.1.4.1.1.130",
            "1.2.840.10008.5.1.4.1.1.128.1",
            "1.2.840.10008.5.1.4.1.1.1",
            "1.2.840.10008.5.1.4.1.1.6.1",
            "1.2.840.10008.5.1.4.1.1.3.1",
            "1.2.840.10008.5.1.4.1.1.20",
            "1.2.840.10008.5.1.4.1.1.7",
            "1.2.840.10008.5.1.4.1.1.12.1",
            "1.2.840.10008.5.1.4.1.1.12.2",
            "1.2.840.10008.5.1.4.1.1.66.4",
            "1.2.840.10008.5.1.4.1.1.30",
        ];
        for uid in &image_uids {
            assert!(is_image_sop_class(uid), "Expected {uid} to be image storage");
        }
    }

    #[test]
    fn test_all_known_non_image_uids_return_false() {
        let non_image_uids = [
            "1.2.840.10008.5.1.4.1.1.481.3",
            "1.2.840.10008.5.1.4.1.1.481.5",
            "1.2.840.10008.5.1.4.1.1.481.2",
            "1.2.840.10008.5.1.4.1.1.88.11",
            "1.2.840.10008.5.1.4.1.1.88.22",
            "1.2.840.10008.5.1.4.1.1.88.33",
            "1.2.840.10008.5.1.4.1.1.88.34",
            "1.2.840.10008.5.1.4.1.1.88.59",
            "1.2.840.10008.5.1.4.1.1.11.1",
            "1.2.840.10008.5.1.4.1.1.11.2",
            "1.2.840.10008.5.1.4.1.1.9.1.1",
            "1.2.840.10008.5.1.4.1.1.104.1",
            "1.2.840.10008.5.1.4.1.1.66.2",
        ];
        for uid in &non_image_uids {
            assert!(!is_image_sop_class(uid), "Expected {uid} to be non-image");
        }
    }
}
