//! Tests for `DicomRsBackend`.
//!
//! Extracted to keep the 500-line structural limit.

use super::*;
use crate::backend::{decode_frame_with, parse_bytes_with, parse_file_with};
use crate::pixel::PixelLayout;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};

#[test]
fn dicom_rs_backend_parses_file_and_decodes_uncompressed_frame() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("slice.dcm");

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.1001"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(vec![10_u16, 20, 30, 40])),
    ));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.1001")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("DICOM file must be written");

    let parsed = parse_file_with::<DicomRsBackend, _>(&path).expect("parse must succeed");

    let decoded = decode_frame_with::<DicomRsBackend>(
        &parsed,
        DecodeFrameRequest {
            frame_index: 0,
            transfer_syntax: TransferSyntaxKind::ExplicitVrLittleEndian,
            layout: PixelLayout {
                rows: 2,
                cols: 2,
                samples_per_pixel: 1,
                bits_allocated: 16,
                pixel_representation: 0,
                rescale_slope: 2.0,
                rescale_intercept: -10.0,
            },
        },
    )
    .expect("decode must succeed");

    assert_eq!(decoded.pixels, vec![10.0, 30.0, 50.0, 70.0]);
}

#[test]
fn dicom_rs_backend_decodes_requested_native_multiframe_only() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("multiframe.dcm");

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.1002"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(vec![1_u16, 2, 100, 200])),
    ));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
            .media_storage_sop_instance_uid("2.25.1002")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("DICOM file must be written");

    let parsed = parse_file_with::<DicomRsBackend, _>(&path).expect("parse must succeed");

    let decoded = decode_frame_with::<DicomRsBackend>(
        &parsed,
        DecodeFrameRequest {
            frame_index: 1,
            transfer_syntax: TransferSyntaxKind::ExplicitVrLittleEndian,
            layout: PixelLayout {
                rows: 1,
                cols: 2,
                samples_per_pixel: 1,
                bits_allocated: 16,
                pixel_representation: 0,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        },
    )
    .expect("second native frame decode must succeed");

    assert_eq!(decoded.pixels, vec![100.0, 200.0]);
}

#[test]
fn native_owned_jpeg_errors_do_not_fallback_to_dicom_rs() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("bad_native_jpeg.dcm");

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![vec![0xFF, 0xD8, 0xFF]]);
    let pixel_sequence: PixelFragmentSequence<Vec<u8>> =
        PixelFragmentSequence::new_fragments(fragments);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.1003"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(8_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OB,
        pixel_sequence,
    ));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
            .media_storage_sop_instance_uid("2.25.1003")
            .transfer_syntax("1.2.840.10008.1.2.4.50"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("DICOM file must be written");

    let parsed = parse_file_with::<DicomRsBackend, _>(&path).expect("parse must succeed");

    let err = decode_frame_with::<DicomRsBackend>(
        &parsed,
        DecodeFrameRequest {
            frame_index: 0,
            transfer_syntax: TransferSyntaxKind::JpegBaseline,
            layout: PixelLayout {
                rows: 1,
                cols: 1,
                samples_per_pixel: 1,
                bits_allocated: 8,
                pixel_representation: 0,
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
        "native-owned JPEG syntaxes must not fall back through dicom-rs, got: {msg}"
    );
}

/// `DicomRsBackend::parse_bytes` must round-trip an in-memory DICOM object.
///
/// Analytical basis:
/// - `InMemDicomObject` with File Meta is written to a temp file.
/// - Raw bytes are read back and parsed via `parse_bytes_with::<DicomRsBackend>`.
/// - The parsed object must contain the same PatientName value that was written,
///   proving `parse_bytes` constructs a semantically equivalent object from
///   Part 10 bytes without file I/O.
#[test]
fn dicom_rs_backend_parse_bytes_round_trips_in_memory_object() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("test_parse_bytes.dcm");

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.2001"),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Test^Patient"),
    ));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.2001")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("DICOM file must be written");

    let bytes = std::fs::read(&path).expect("temp file must be readable");
    let parsed = parse_bytes_with::<DicomRsBackend>(&bytes)
        .expect("parse_bytes must succeed on valid Part 10 bytes");

    let patient_name = parsed
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must be present")
        .value()
        .to_str()
        .expect("PatientName must be a string")
        .trim_end_matches(|c: char| c == '\0' || c == ' ')
        .to_owned();

    assert_eq!(patient_name, "Test^Patient");
}

/// `DicomRsBackend::parse_bytes` must reject garbage input.
///
/// Analytical basis: bytes without a valid DICOM Part 10 preamble or
/// DICM magic cannot be parsed. The function must return `Err`, not panic.
#[test]
fn dicom_rs_backend_parse_bytes_rejects_garbage_input() {
    let result = parse_bytes_with::<DicomRsBackend>(&[0xDE, 0xAD, 0xBE, 0xEF]);
    assert!(result.is_err(), "parse_bytes must reject non-DICOM bytes");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.to_lowercase().contains("dicom")
            || msg.to_lowercase().contains("parse")
            || msg.to_lowercase().contains("failed"),
        "error must describe parse failure, got: {msg}"
    );
}
