use super::super::decode_compressed_frame;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};
use ritk_codecs::jpeg_ls::encoder::encode_grayscale_jpeg_ls;

/// Build and write a minimal JPEG-LS DICOM Part 10 file.
///
/// Pixel data is encoded with the RITK-native pure-Rust JPEG-LS encoder
/// (ISO 14495-1 / ITU-T T.87). `near = 0` produces TS 1.2.840.10008.1.2.4.80
/// (lossless); `near > 0` produces TS 1.2.840.10008.1.2.4.81 (near-lossless).
/// The bitstream is encapsulated as a single fragment per DICOM PS3.5 §A.4.
fn write_jpegls_dicom_file(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels_u8: &[u8],
    near: u32,
) {
    assert_eq!(
        pixels_u8.len(),
        (width * height) as usize,
        "pixels_u8 length must equal width × height"
    );

    let samples: Vec<u16> = pixels_u8.iter().map(|&v| u16::from(v)).collect();
    let jls_bytes = encode_grayscale_jpeg_ls(&samples, height, width, 8, near);

    let (transfer_syntax, sop_instance) = if near == 0 {
        ("1.2.840.10008.1.2.4.80", "2.25.99999931")
    } else {
        ("1.2.840.10008.1.2.4.81", "2.25.99999941")
    };

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jls_bytes]);
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
        PrimitiveValue::from(sop_instance),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0010, 0x0020),
        VR::LO,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from("2.25.99999932"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from("2.25.99999933"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(height as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(width as u16),
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

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                .media_storage_sop_instance_uid(sop_instance)
                .transfer_syntax(transfer_syntax),
        )
        .expect("FileMetaTableBuilder failed");
    file_obj.write_to_file(path).expect("write_to_file failed");
}

/// JPEG-LS Lossless round-trip: a natively encoded bitstream is decoded by
/// the RITK-native lossless decoder with exact per-sample fidelity.
#[test]
fn test_decode_compressed_frame_jpegls_lossless_round_trip() {
    let width = 4u32;
    let height = 1u32;
    let original: Vec<u8> = vec![0, 42, 85, 127];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpegls_lossless.dcm");
    write_jpegls_dicom_file(&path, width, height, &original, 0);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded =
        decode_compressed_frame(&obj, 0, 8, ritk_dicom::PixelSignedness::Unsigned, 1.0, 0.0)
            .expect("JPEG-LS Lossless decode must succeed");

    let expected = original
        .iter()
        .map(|&sample| f32::from(sample))
        .collect::<Vec<_>>();
    assert_eq!(
        decoded, expected,
        "JPEG-LS Lossless must preserve every sample exactly"
    );
}

/// JPEG-LS Lossless multi-row round-trip: a natively encoded scan containing
/// regular-mode and run-mode segments is decoded without bit drift.
#[test]
fn test_decode_compressed_frame_jpegls_lossless_multirow_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        0, 42, 85, 127, 128, 170, 200, 225, 10, 20, 30, 40, 240, 230, 220, 210,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpegls_lossless_multirow.dcm");
    write_jpegls_dicom_file(&path, width, height, &original, 0);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded =
        decode_compressed_frame(&obj, 0, 8, ritk_dicom::PixelSignedness::Unsigned, 1.0, 0.0)
            .expect("JPEG-LS Lossless decode must succeed for multi-row fixture");

    let expected = original
        .iter()
        .map(|&sample| f32::from(sample))
        .collect::<Vec<_>>();
    assert_eq!(
        decoded, expected,
        "JPEG-LS Lossless multi-row decode must preserve every sample exactly"
    );
}

/// JPEG-LS Near-Lossless round-trip: encode NEAR=2 with the RITK-native
/// encoder, decode through the full DICOM pipeline (TS .81 now routes to the
/// RITK-native decoder), verify per-pixel reconstruction error ≤ NEAR.
///
/// # Mathematical justification
/// JPEG-LS near-lossless (NEAR = 2) guarantees |s' − s| ≤ 2 for all pixels
/// per ISO 14495-1 §A.4.4. Tolerance set to exactly 2.0 (the analytical
/// bound). Cross-implementation conformance of the native NEAR encoder was
/// additionally verified one-time against the CharLS-backed dicom-rs decoder
/// before the charls dependency was removed (Sprint 369 gap_audit).
#[test]
fn test_decode_compressed_frame_jpegls_near_lossless_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        10, 50, 100, 150, 200, 245, 30, 80, 130, 180, 220, 60, 110, 160, 210, 40,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpegls_nearlossless.dcm");
    write_jpegls_dicom_file(&path, width, height, &original, 2);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded =
        decode_compressed_frame(&obj, 0, 8, ritk_dicom::PixelSignedness::Unsigned, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG-LS Near-Lossless");

    assert_eq!(decoded.len(), 16, "decoded pixel count must equal 16");
    let max_error = original
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dec)| (orig as f32 - dec).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 2.0,
        "JPEG-LS Near-Lossless decode error {max_error} exceeds analytical bound of 2 \
         (ISO 14495-1: NEAR=2 ⟹ |s'-s| ≤ 2)"
    );
}
