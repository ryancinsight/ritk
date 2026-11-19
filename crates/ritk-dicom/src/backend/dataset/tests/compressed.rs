//! Tests for `NativeBackend`.
//!
//! Extracted to keep the 500-line structural limit.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::super::*;
use crate::backend::{
    decode_frame_with, parse_bytes_with, parse_file_with, write_bytes_with, write_file_with,
};
use crate::dataset::PixelFragments;
use crate::dataset::SmallVec;
use crate::dataset::{DataElement, PrimitiveValue, VR};
use crate::dataset::{Dataset, FileMetaBuilder};
use crate::pixel::{PixelLayout, PixelSignedness};
use ritk_codecs::encode_rle_lossless_fragment_u16_grayscale;
use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, Jpeg2000Encoding};
use ritk_codecs::jpeg_ls::encoder::encode_grayscale_jpeg_ls;

fn write_single_frame_compressed_fixture(
    path: &std::path::Path,
    width: u16,
    height: u16,
    bits_allocated: u16,
    bits_stored: u16,
    transfer_syntax_uid: &str,
    sop_instance_uid: &str,
    fragment: Vec<u8>,
) {
    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![fragment]);
    let pfs: PixelFragments = PixelFragments::from_fragments(fragments);

    let mut obj = Dataset::new();
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Compressed^Roundtrip"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(height),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(width),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(bits_allocated),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(bits_stored),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(
            bits_stored
                .checked_sub(1)
                .expect("fixture BitsStored must be positive"),
        ),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x1053),
        VR::DS,
        PrimitiveValue::from("1.000000"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x1052),
        VR::DS,
        PrimitiveValue::from("0.000000"),
    ));
    obj.put(DataElement::new(DicomTag::new(0x7FE0, 0x0010), VR::OB, pfs));

    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
            .media_storage_sop_instance_uid(sop_instance_uid)
            .transfer_syntax(transfer_syntax_uid),
    )
    .expect("file meta must be valid")
    .write_to_file(path)
    .expect("compressed fixture file write must succeed");
}

#[test]
fn native_backend_round_trips_jpegls_pixeldata_via_write_bytes_and_file() {
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
        8,
        8,
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
            bits_stored: 8,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        },
    };

    let parsed = parse_file_with::<NativeBackend, _>(&src).expect("parse source JPEG-LS file");
    let decoded_before = decode_frame_with::<NativeBackend>(&parsed, request.clone())
        .expect("decode source JPEG-LS frame")
        .pixels;
    let bytes = write_bytes_with::<NativeBackend>(&parsed).expect("write JPEG-LS bytes");
    let reparsed_bytes = parse_bytes_with::<NativeBackend>(&bytes).expect("reparse JPEG-LS bytes");
    let decoded_after_bytes = decode_frame_with::<NativeBackend>(&reparsed_bytes, request.clone())
        .expect("decode JPEG-LS frame after write_bytes")
        .pixels;

    write_file_with::<NativeBackend, _>(&dst, &parsed).expect("write JPEG-LS file");
    let reparsed_file = parse_file_with::<NativeBackend, _>(&dst).expect("reparse JPEG-LS file");
    let decoded_after_file = decode_frame_with::<NativeBackend>(&reparsed_file, request)
        .expect("decode JPEG-LS frame after write_file")
        .pixels;

    let expected: Vec<f32> = original_u8.iter().map(|&v| f32::from(v)).collect();
    assert_eq!(decoded_before, expected);
    assert_eq!(decoded_after_bytes, expected);
    assert_eq!(decoded_after_file, expected);
}

#[test]
fn native_backend_round_trips_j2k_pixeldata_via_write_bytes() {
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
        8,
        8,
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
            bits_stored: 8,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        },
    };

    let parsed = parse_file_with::<NativeBackend, _>(&src).expect("parse source JPEG 2000 file");
    let decoded_before = decode_frame_with::<NativeBackend>(&parsed, request.clone())
        .expect("decode source JPEG 2000 frame")
        .pixels;
    let bytes = write_bytes_with::<NativeBackend>(&parsed).expect("write JPEG 2000 bytes");
    let reparsed = parse_bytes_with::<NativeBackend>(&bytes).expect("reparse JPEG 2000 bytes");
    let decoded_after = decode_frame_with::<NativeBackend>(&reparsed, request)
        .expect("decode JPEG 2000 frame after write_bytes")
        .pixels;

    let expected: Vec<f32> = original_u8.iter().map(|&v| f32::from(v)).collect();
    assert_eq!(decoded_before, expected);
    assert_eq!(decoded_after, expected);
}

#[test]
fn native_backend_round_trips_rle_pixeldata_via_write_bytes() {
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
        16,
        16,
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
            bits_stored: 16,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        },
    };

    let parsed = parse_file_with::<NativeBackend, _>(&src).expect("parse source RLE file");
    let decoded_before = decode_frame_with::<NativeBackend>(&parsed, request.clone())
        .expect("decode source RLE frame")
        .pixels;
    let bytes = write_bytes_with::<NativeBackend>(&parsed).expect("write RLE bytes");
    let reparsed = parse_bytes_with::<NativeBackend>(&bytes).expect("reparse RLE bytes");
    let decoded_after = decode_frame_with::<NativeBackend>(&reparsed, request)
        .expect("decode RLE frame after write_bytes")
        .pixels;

    let expected: Vec<f32> = original_u16.iter().map(|&v| v as f32).collect();
    assert_eq!(decoded_before, expected);
    assert_eq!(decoded_after, expected);
}
