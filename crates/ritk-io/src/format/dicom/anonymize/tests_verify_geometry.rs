//! Geometry coherence in the export gate.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::tests_verify::{build_object, write_to_temp};
use super::verify::{verify_dicom_file, VerifyIssue, VerifyOptions};
use ritk_dicom::dataset::{DataElement, DicomTag, PrimitiveValue, VR};

#[test]
fn geometry_pixel_mismatch_reported() {
    let mut obj = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    // Declare 4x4 but keep 2x2 pixel data → length mismatch.
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(4_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(4_u16),
    ));
    let path = write_to_temp(&obj, "geometry_mismatch.dcm");

    let report = verify_dicom_file(&path, &VerifyOptions::default()).unwrap();
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, VerifyIssue::GeometryMismatch { .. })),
        "geometry mismatch must be reported, got {:?}",
        report.issues
    );
}

#[test]
fn geometry_is_required_of_pixel_data_and_of_nothing_else() {
    let geometry = [
        DicomTag::new(0x0028, 0x0010),
        DicomTag::new(0x0028, 0x0011),
        DicomTag::new(0x0028, 0x0100),
    ];
    let mut report_like = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
    let mut image = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.9");
    for tag in geometry {
        assert!(report_like.remove_element(tag).is_some());
        assert!(image.remove_element(tag).is_some());
    }
    assert!(report_like
        .remove_element(DicomTag::new(0x7FE0, 0x0010))
        .is_some());
    let missing = [VerifyIssue::GeometryMismatch {
        detail: "Rows/Columns/BitsAllocated must all be present".to_owned(),
    }];
    let pixel_elements = [
        (DicomTag::new(0x7FE0, 0x0008), VR::OF, vec![0_u8; 4]),
        (DicomTag::new(0x7FE0, 0x0009), VR::OD, vec![0_u8; 8]),
        (DicomTag::new(0x7FE0, 0x0010), VR::OB, Vec::new()),
    ];
    for (tag, vr, bytes) in pixel_elements {
        let mut object = report_like.clone();
        object.put(DataElement::new(tag, vr, PrimitiveValue::from(bytes)));
        let found = verify_dicom_file(
            write_to_temp(&object, "pixel-element-without-geometry.dcm"),
            &VerifyOptions::default(),
        )
        .unwrap();
        assert_eq!(found.issues, missing, "{tag}");
    }

    let without_pixels = verify_dicom_file(
        write_to_temp(&report_like, "no-pixels.dcm"),
        &VerifyOptions::default(),
    )
    .unwrap();
    let with_pixels = verify_dicom_file(
        write_to_temp(&image, "pixels-without-geometry.dcm"),
        &VerifyOptions::default(),
    )
    .unwrap();

    assert_eq!(without_pixels.issues, []);
    assert_eq!(with_pixels.issues, missing);
}
