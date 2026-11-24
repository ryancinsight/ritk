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

#[test]
fn a_frame_is_sized_in_packed_bits_only_when_one_bit_is_allocated() {
    // A 5 x 5 single-sample frame is 25 bits, 4 bytes packed and 25 bytes at
    // 8 bits; a 4 x 4 one is 16 bits, exactly 2 bytes packed.
    let cases = [
        (1_u16, 5_u16, 4_usize, None),
        (1, 5, 2, Some(4_usize)),
        (1, 4, 2, None),
        (1, 4, 0, Some(2)),
        (8, 5, 26, None),
        (8, 5, 24, Some(25)),
    ];
    for (bits, side, length, expected) in cases {
        let mut object = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
        for (element, value) in [(0x0010, side), (0x0011, side), (0x0100, bits)] {
            object.put(DataElement::new(
                DicomTag::new(0x0028, element),
                VR::US,
                PrimitiveValue::from(value),
            ));
        }
        object.put(DataElement::new(
            DicomTag::new(0x7FE0, 0x0010),
            VR::OB,
            PrimitiveValue::from(vec![0_u8; length]),
        ));
        let report = verify_dicom_file(
            write_to_temp(&object, "packed-bits.dcm"),
            &VerifyOptions::default(),
        )
        .unwrap();
        let issues: Vec<VerifyIssue> = expected
            .map(|expected| VerifyIssue::GeometryMismatch {
                detail: format!(
                    "PixelData length {length} < expected single-frame {expected} \
                     (rows={side} cols={side} samples=1 bits={bits})"
                ),
            })
            .into_iter()
            .collect();
        assert_eq!(
            report.issues, issues,
            "bits={bits} side={side} length={length}"
        );
    }
}

#[test]
fn a_frame_size_past_64_bits_is_a_mismatch() {
    // Rows, Columns and SamplesPerPixel are US in the standard; a writer that
    // stores them as UL can declare 2^32 - 1 of each.
    let cases = [(1_u16, u32::MAX), (16, u32::MAX)];
    for (bits, samples) in cases {
        let mut object = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
        for (element, value) in [(0x0010, u32::MAX), (0x0011, u32::MAX), (0x0002, samples)] {
            object.put(DataElement::new(
                DicomTag::new(0x0028, element),
                VR::UL,
                PrimitiveValue::from(value),
            ));
        }
        object.put(DataElement::new(
            DicomTag::new(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(bits),
        ));
        let report = verify_dicom_file(
            write_to_temp(&object, "oversized-frame.dcm"),
            &VerifyOptions::default(),
        )
        .unwrap();
        assert_eq!(
            report.issues,
            [VerifyIssue::GeometryMismatch {
                detail: format!(
                    "frame size exceeds 64 bits \
                     (rows=4294967295 cols=4294967295 samples=4294967295 bits={bits})"
                ),
            }],
            "bits={bits}"
        );
    }
}

#[test]
fn a_zero_or_unreadable_samples_per_pixel_is_a_mismatch() {
    // The fixture holds a 2 x 2 frame of 16 bits and 8 bytes of Pixel Data.
    let cases = [
        (
            DataElement::new(
                DicomTag::new(0x0028, 0x0002),
                VR::US,
                PrimitiveValue::from(0_u16),
            ),
            "non-positive geometry rows=2 cols=2 samples=0 bits=16",
        ),
        (
            DataElement::new(
                DicomTag::new(0x0028, 0x0002),
                VR::LO,
                PrimitiveValue::from("three"),
            ),
            "SamplesPerPixel is present and is not a number",
        ),
    ];
    for (samples, detail) in cases {
        let mut object = build_object("1.2.3.4.5.6", "1.2.3.4.5.7", "1.2.3.4.5.8");
        object.put(samples);
        let report = verify_dicom_file(
            write_to_temp(&object, "samples.dcm"),
            &VerifyOptions::default(),
        )
        .unwrap();
        assert_eq!(
            report.issues,
            [VerifyIssue::GeometryMismatch {
                detail: detail.to_owned(),
            }]
        );
    }
}
