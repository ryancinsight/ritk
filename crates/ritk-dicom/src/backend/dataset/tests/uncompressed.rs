//! Tests for `NativeBackend`.
//!
//! Extracted to keep the 500-line structural limit.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::super::*;
use crate::backend::parse_budget::{parse_bytes_with_budget, parse_file_with_budget};
use crate::backend::{decode_frame_with, parse_file_with};
use crate::dataset::PixelFragments;
use crate::dataset::SmallVec;
use crate::dataset::{DataElement, PrimitiveValue, VR};
use crate::dataset::{Dataset, FileMetaBuilder};
use crate::pixel::{PixelLayout, PixelSignedness};
use crate::ParseBudget;

#[test]
fn native_backend_parses_file_and_decodes_uncompressed_frame() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("slice.dcm");

    let mut obj = Dataset::new();
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.1001"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(15_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(vec![10_u16, 20, 30, 40])),
    ));

    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.1001")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("DICOM file must be written");

    let mut parsed = parse_file_with::<NativeBackend, _>(&path).expect("parse must succeed");
    let encoded = std::fs::read(&path).expect("written DICOM must be readable");
    let bounded = parse_bytes_with_budget::<NativeBackend>(&encoded, &ParseBudget::DEFAULT)
        .expect("budgeted parse must succeed");
    let bounded_file = parse_file_with_budget::<NativeBackend, _>(&path, &ParseBudget::DEFAULT)
        .expect("budgeted file parse must succeed");
    assert_eq!(
        bounded
            .element(DicomTag::new(0x0008, 0x0018))
            .expect("SOP Instance UID must exist")
            .to_str()
            .expect("SOP Instance UID must be text"),
        "2.25.1001"
    );
    assert_eq!(
        bounded_file
            .element(DicomTag::new(0x0008, 0x0018))
            .expect("SOP Instance UID must exist")
            .to_str()
            .expect("SOP Instance UID must be text"),
        "2.25.1001"
    );
    let budget = ParseBudget::new(encoded.len() - 1, 100, 8);
    let error = parse_file_with_budget::<NativeBackend, _>(&path, &budget)
        .expect_err("file byte budget must reject before parsing");
    assert!(
        format!("{error:#}").contains("DICOM file exceeds parse budget"),
        "unexpected budget error: {error:#}"
    );

    let request = DecodeFrameRequest {
        frame_index: 0,
        transfer_syntax: TransferSyntaxKind::ExplicitVrLittleEndian,
        layout: PixelLayout {
            rows: 2,
            cols: 2,
            samples_per_pixel: 1,
            bits_allocated: 16,
            bits_stored: 16,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 2.0,
            rescale_intercept: -10.0,
        },
    };
    let decoded =
        decode_frame_with::<NativeBackend>(&parsed, request.clone()).expect("decode must succeed");

    assert_eq!(decoded.pixels, vec![10.0, 30.0, 50.0, 70.0]);

    parsed.put(DataElement::new(
        DicomTag::new(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(14_u16),
    ));
    let error = decode_frame_with::<NativeBackend>(&parsed, request)
        .expect_err("HighBit must equal BitsStored minus one");
    assert!(
        error
            .to_string()
            .contains("HighBit=14 does not equal BitsStored-1=15"),
        "expected HighBit alignment error, got {error:#}"
    );
}

#[test]
fn native_backend_decodes_requested_native_multiframe_only() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("multiframe.dcm");

    let mut obj = Dataset::new();
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.1002"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("2"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(vec![1_u16, 2, 100, 200])),
    ));

    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
            .media_storage_sop_instance_uid("2.25.1002")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("DICOM file must be written");

    let parsed = parse_file_with::<NativeBackend, _>(&path).expect("parse must succeed");

    let decoded = decode_frame_with::<NativeBackend>(
        &parsed,
        DecodeFrameRequest {
            frame_index: 1,
            transfer_syntax: TransferSyntaxKind::ExplicitVrLittleEndian,
            layout: PixelLayout {
                rows: 1,
                cols: 2,
                samples_per_pixel: 1,
                bits_allocated: 16,
                bits_stored: 16,
                pixel_representation: PixelSignedness::Unsigned,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        },
    )
    .expect("second native frame decode must succeed");

    assert_eq!(decoded.pixels, vec![100.0, 200.0]);
}

#[test]
fn malformed_jpeg_returns_the_codec_error() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("bad_native_jpeg.dcm");

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![vec![0xFF, 0xD8, 0xFF]]);
    let pixel_sequence: PixelFragments = PixelFragments::from_fragments(fragments);

    let mut obj = Dataset::new();
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.1003"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(8_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OB,
        pixel_sequence,
    ));

    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
            .media_storage_sop_instance_uid("2.25.1003")
            .transfer_syntax("1.2.840.10008.1.2.4.50"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("DICOM file must be written");

    let parsed = parse_file_with::<NativeBackend, _>(&path).expect("parse must succeed");

    let err = decode_frame_with::<NativeBackend>(
        &parsed,
        DecodeFrameRequest {
            frame_index: 0,
            transfer_syntax: TransferSyntaxKind::JpegBaseline,
            layout: PixelLayout {
                rows: 1,
                cols: 1,
                samples_per_pixel: 1,
                bits_allocated: 8,
                bits_stored: 8,
                pixel_representation: PixelSignedness::Unsigned,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        },
    )
    .expect_err("malformed native-owned JPEG fragment must fail");

    let msg = format!("{err:#}");
    assert!(
        msg.contains("JPEG"),
        "error must come from the RITK-native JPEG decoder, got: {msg}"
    );
    assert!(
        !msg.contains("fallback"),
        "expected the native JPEG decoder error, got: {msg}"
    );
}
