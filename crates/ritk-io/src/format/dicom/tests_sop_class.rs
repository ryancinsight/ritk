use super::*;

// Positive: image SOP classes

#[test]
fn test_ct_is_image_storage() {
    assert_eq!(
        classify_sop_class("1.2.840.10008.5.1.4.1.1.2"),
        SopClassKind::CtImageStorage
    );
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.2").is_image_storage());
}

#[test]
fn test_mr_is_image_storage() {
    assert_eq!(
        classify_sop_class("1.2.840.10008.5.1.4.1.1.4"),
        SopClassKind::MrImageStorage
    );
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.4").is_image_storage());
}

#[test]
fn test_pet_is_image_storage() {
    assert_eq!(
        classify_sop_class("1.2.840.10008.5.1.4.1.1.128"),
        SopClassKind::PetImageStorage
    );
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.128").is_image_storage());
}

#[test]
fn test_enhanced_ct_is_image_storage() {
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.2.1").is_image_storage());
}

#[test]
fn test_enhanced_mr_is_image_storage() {
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.4.1").is_image_storage());
}

#[test]
fn test_secondary_capture_is_image_storage() {
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.7").is_image_storage());
}

#[test]
fn test_ultrasound_is_image_storage() {
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.6.1").is_image_storage());
}

#[test]
fn test_nm_is_image_storage() {
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.20").is_image_storage());
}

#[test]
fn test_segmentation_is_image_storage() {
    assert!(classify_sop_class("1.2.840.10008.5.1.4.1.1.66.4").is_image_storage());
}

// Negative: non-image SOP classes

#[test]
fn test_rtstruct_is_not_image_storage() {
    assert_eq!(
        classify_sop_class("1.2.840.10008.5.1.4.1.1.481.3"),
        SopClassKind::RtStructureSetStorage
    );
    assert!(!classify_sop_class("1.2.840.10008.5.1.4.1.1.481.3").is_image_storage());
}

#[test]
fn test_rtplan_is_not_image_storage() {
    assert_eq!(
        classify_sop_class("1.2.840.10008.5.1.4.1.1.481.5"),
        SopClassKind::RtPlanStorage
    );
    assert!(!classify_sop_class("1.2.840.10008.5.1.4.1.1.481.5").is_image_storage());
}

#[test]
fn test_rtdose_is_not_image_storage() {
    assert!(!classify_sop_class("1.2.840.10008.5.1.4.1.1.481.2").is_image_storage());
}

#[test]
fn test_basic_text_sr_is_not_image_storage() {
    assert_eq!(
        classify_sop_class("1.2.840.10008.5.1.4.1.1.88.11"),
        SopClassKind::BasicTextSrStorage
    );
    assert!(!classify_sop_class("1.2.840.10008.5.1.4.1.1.88.11").is_image_storage());
}

#[test]
fn test_enhanced_sr_is_not_image_storage() {
    assert!(!classify_sop_class("1.2.840.10008.5.1.4.1.1.88.22").is_image_storage());
}

#[test]
fn test_grayscale_pr_is_not_image_storage() {
    assert!(!classify_sop_class("1.2.840.10008.5.1.4.1.1.11.1").is_image_storage());
}

#[test]
fn test_waveform_ecg_is_not_image_storage() {
    assert!(!classify_sop_class("1.2.840.10008.5.1.4.1.1.9.1.1").is_image_storage());
}

#[test]
fn test_encapsulated_pdf_is_not_image_storage() {
    assert!(!classify_sop_class("1.2.840.10008.5.1.4.1.1.104.1").is_image_storage());
}

// Boundary: unknown and empty UIDs

#[test]
fn test_unknown_uid_classified_as_other() {
    let kind = classify_sop_class("9.9.9.9.9.9.9");
    assert!(matches!(kind, SopClassKind::Other(_)));
    assert!(!classify_sop_class("9.9.9.9.9.9.9").is_image_storage());
}

#[test]
fn test_empty_uid_classified_as_other() {
    let kind = classify_sop_class("");
    assert!(matches!(kind, SopClassKind::Other(_)));
    assert!(!classify_sop_class("").is_image_storage());
}

#[test]
fn test_uid_with_leading_trailing_whitespace() {
    assert!(classify_sop_class("  1.2.840.10008.5.1.4.1.1.2  ").is_image_storage());
    assert!(!classify_sop_class("  1.2.840.10008.5.1.4.1.1.481.3  ").is_image_storage());
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
        assert!(
            classify_sop_class(uid).is_image_storage(),
            "Expected {uid} to be image storage"
        );
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
        assert!(
            !classify_sop_class(uid).is_image_storage(),
            "Expected {uid} to be non-image"
        );
    }
}
