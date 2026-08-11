//! Tests for `DicomRsBackend`.
//!
//! Extracted to keep the 500-line structural limit.

use super::*;
use crate::backend::{
    decode_frame_with, parse_bytes_with, parse_file_with, write_bytes_with, write_file_with,
};
use crate::pixel::{PixelLayout, PixelSignedness};
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};
use ritk_codecs::encode_rle_lossless_fragment_u16_grayscale;
use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, Jpeg2000Encoding};
use ritk_codecs::jpeg_ls::encoder::encode_grayscale_jpeg_ls;

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
                pixel_representation: PixelSignedness::Unsigned,
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
        .trim_end_matches(['\0', ' '])
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

#[test]
fn dicom_rs_backend_writes_bytes_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("write_bytes_src.dcm");

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.3001"),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Encoder^Roundtrip"),
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
        PrimitiveValue::U16(SmallVec::from_vec(vec![42_u16])),
    ));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.3001")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("source DICOM file must be written");

    let parsed = parse_file_with::<DicomRsBackend, _>(&path).expect("parse must succeed");
    let bytes = write_bytes_with::<DicomRsBackend>(&parsed).expect("write_bytes must succeed");
    let reparsed = parse_bytes_with::<DicomRsBackend>(&bytes).expect("re-parse must succeed");

    let patient_name = reparsed
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must be present")
        .value()
        .to_str()
        .expect("PatientName must be a string")
        .trim_end_matches(['\0', ' '])
        .to_owned();
    assert_eq!(patient_name, "Encoder^Roundtrip");
}

#[test]
fn dicom_rs_backend_writes_file_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let src = dir.path().join("write_file_src.dcm");
    let dst = dir.path().join("write_file_dst.dcm");

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.3002"),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Encoder^File"),
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
        PrimitiveValue::U16(SmallVec::from_vec(vec![7_u16])),
    ));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.3002")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&src)
    .expect("source DICOM file must be written");

    let parsed = parse_file_with::<DicomRsBackend, _>(&src).expect("parse must succeed");
    write_file_with::<DicomRsBackend, _>(&dst, &parsed).expect("write_file must succeed");

    let reparsed = parse_file_with::<DicomRsBackend, _>(&dst).expect("re-parse must succeed");
    let patient_name = reparsed
        .element(Tag(0x0010, 0x0010))
        .expect("PatientName must be present")
        .value()
        .to_str()
        .expect("PatientName must be a string")
        .trim_end_matches(['\0', ' '])
        .to_owned();
    assert_eq!(patient_name, "Encoder^File");
}

fn write_single_frame_compressed_fixture(
    path: &std::path::Path,
    width: u16,
    height: u16,
    transfer_syntax_uid: &str,
    sop_instance_uid: &str,
    fragment: Vec<u8>,
) {
    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![fragment]);
    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Compressed^Roundtrip"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(height),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(width),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(8u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(8u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(7u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1053),
        VR::DS,
        PrimitiveValue::from("1.000000"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1052),
        VR::DS,
        PrimitiveValue::from("0.000000"),
    ));
    obj.put(DataElement::new(Tag(0x7FE0, 0x0010), VR::OB, pfs));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
            .media_storage_sop_instance_uid(sop_instance_uid)
            .transfer_syntax(transfer_syntax_uid),
    )
    .expect("file meta must be valid")
    .write_to_file(path)
    .expect("compressed fixture file write must succeed");
}

#[test]
fn dicom_rs_backend_round_trips_jpegls_pixeldata_via_write_bytes_and_file() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let src = dir.path().join("jls_src.dcm");
    let dst = dir.path().join("jls_dst.dcm");
    let width = 3u16;
    let height = 2u16;

    let original_u8: Vec<u8> = vec![5, 7, 9, 11, 13, 15];
    let original_u16: Vec<u16> = original_u8.iter().map(|&v| u16::from(v)).collect();
    let fragment =
        encode_grayscale_jpeg_ls(&original_u16, u32::from(height), u32::from(width), 8, 0)
            .expect("JPEG-LS fixture encode must succeed");
    write_single_frame_compressed_fixture(
        &src,
        width,
        height,
        "1.2.840.10008.1.2.4.80",
        "2.25.900001",
        fragment,
    );

    let request = DecodeFrameRequest {
        frame_index: 0,
        transfer_syntax: TransferSyntaxKind::JpegLsLossless,
        layout: PixelLayout {
            rows: usize::from(height),
            cols: usize::from(width),
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        },
    };

    let parsed = parse_file_with::<DicomRsBackend, _>(&src).expect("parse source JPEG-LS file");
    let decoded_before = decode_frame_with::<DicomRsBackend>(&parsed, request.clone())
        .expect("decode source JPEG-LS frame")
        .pixels;
    let bytes = write_bytes_with::<DicomRsBackend>(&parsed).expect("write JPEG-LS bytes");
    let reparsed_bytes = parse_bytes_with::<DicomRsBackend>(&bytes).expect("reparse JPEG-LS bytes");
    let decoded_after_bytes = decode_frame_with::<DicomRsBackend>(&reparsed_bytes, request.clone())
        .expect("decode JPEG-LS frame after write_bytes")
        .pixels;

    write_file_with::<DicomRsBackend, _>(&dst, &parsed).expect("write JPEG-LS file");
    let reparsed_file = parse_file_with::<DicomRsBackend, _>(&dst).expect("reparse JPEG-LS file");
    let decoded_after_file = decode_frame_with::<DicomRsBackend>(&reparsed_file, request)
        .expect("decode JPEG-LS frame after write_file")
        .pixels;

    let expected: Vec<f32> = original_u8.iter().map(|&v| f32::from(v)).collect();
    assert_eq!(decoded_before, expected);
    assert_eq!(decoded_after_bytes, expected);
    assert_eq!(decoded_after_file, expected);
}

#[test]
fn dicom_rs_backend_round_trips_j2k_pixeldata_via_write_bytes() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let src = dir.path().join("j2k_src.dcm");
    let width = 2u16;
    let height = 2u16;

    let original_u8: Vec<u8> = vec![1, 2, 3, 4];
    let original_i32: Vec<i32> = original_u8.iter().map(|&v| i32::from(v)).collect();
    let fragment = encode_grayscale_j2k(
        &original_i32,
        u32::from(height),
        u32::from(width),
        8,
        PixelSignedness::Unsigned,
        Jpeg2000Encoding::Lossless {
            decomposition_levels: 1,
        },
    )
    .expect("JPEG 2000 fixture encode must succeed");
    write_single_frame_compressed_fixture(
        &src,
        width,
        height,
        "1.2.840.10008.1.2.4.90",
        "2.25.900002",
        fragment,
    );

    let request = DecodeFrameRequest {
        frame_index: 0,
        transfer_syntax: TransferSyntaxKind::Jpeg2000Lossless,
        layout: PixelLayout {
            rows: usize::from(height),
            cols: usize::from(width),
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        },
    };

    let parsed = parse_file_with::<DicomRsBackend, _>(&src).expect("parse source JPEG 2000 file");
    let decoded_before = decode_frame_with::<DicomRsBackend>(&parsed, request.clone())
        .expect("decode source JPEG 2000 frame")
        .pixels;
    let bytes = write_bytes_with::<DicomRsBackend>(&parsed).expect("write JPEG 2000 bytes");
    let reparsed = parse_bytes_with::<DicomRsBackend>(&bytes).expect("reparse JPEG 2000 bytes");
    let decoded_after = decode_frame_with::<DicomRsBackend>(&reparsed, request)
        .expect("decode JPEG 2000 frame after write_bytes")
        .pixels;

    let expected: Vec<f32> = original_u8.iter().map(|&v| f32::from(v)).collect();
    assert_eq!(decoded_before, expected);
    assert_eq!(decoded_after, expected);
}

#[test]
fn dicom_rs_backend_round_trips_rle_pixeldata_via_write_bytes() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let src = dir.path().join("rle_src.dcm");
    let width = 3u16;
    let height = 2u16;

    let original_u16: Vec<u16> = vec![1, 2, 1024, 2048, 4096, 65535];
    let fragment = encode_rle_lossless_fragment_u16_grayscale(&original_u16);
    write_single_frame_compressed_fixture(
        &src,
        width,
        height,
        "1.2.840.10008.1.2.5",
        "2.25.900003",
        fragment,
    );

    let request = DecodeFrameRequest {
        frame_index: 0,
        transfer_syntax: TransferSyntaxKind::RleLossless,
        layout: PixelLayout {
            rows: usize::from(height),
            cols: usize::from(width),
            samples_per_pixel: 1,
            bits_allocated: 16,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        },
    };

    let parsed = parse_file_with::<DicomRsBackend, _>(&src).expect("parse source RLE file");
    let decoded_before = decode_frame_with::<DicomRsBackend>(&parsed, request.clone())
        .expect("decode source RLE frame")
        .pixels;
    let bytes = write_bytes_with::<DicomRsBackend>(&parsed).expect("write RLE bytes");
    let reparsed = parse_bytes_with::<DicomRsBackend>(&bytes).expect("reparse RLE bytes");
    let decoded_after = decode_frame_with::<DicomRsBackend>(&reparsed, request)
        .expect("decode RLE frame after write_bytes")
        .pixels;

    let expected: Vec<f32> = original_u16.iter().map(|&v| v as f32).collect();
    assert_eq!(decoded_before, expected);
    assert_eq!(decoded_after, expected);
}
