use super::super::decode_compressed_frame;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};

/// Build and write a minimal JPEG-LS Lossless DICOM Part 10 file.
///
/// Pixel data is encoded losslessly (near-lossless parameter = 0) using the
/// `charls` crate (CharLS C++ JPEG-LS implementation, ISO 14495-1 / ITU-T T.87).
/// The bitstream is encapsulated as a single fragment per DICOM PS3.5 §A.4.
fn write_jpegls_lossless_dicom_file(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels_u8: &[u8],
) {
    assert_eq!(
        pixels_u8.len(),
        (width * height) as usize,
        "pixels_u8 length must equal width × height"
    );

    let frame_info = charls::FrameInfo {
        width,
        height,
        bits_per_sample: 8,
        component_count: 1,
    };
    let mut codec = charls::CharLS::default();
    let jls_bytes = codec
        .encode(frame_info, 0, pixels_u8)
        .expect("CharLS encode failed");
    let charls_decoded = codec
        .decode(&jls_bytes)
        .expect("CharLS self-decode failed for lossless fixture");
    assert_eq!(
        charls_decoded, pixels_u8,
        "CharLS lossless fixture generator must preserve source bytes"
    );

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jls_bytes]);
    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(Tag(0x0008, 0x0016), VR::UI, PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3")));
    obj.put(DataElement::new(Tag(0x0008, 0x0018), VR::UI, PrimitiveValue::from("2.25.99999931")));
    obj.put(DataElement::new(Tag(0x0010, 0x0010), VR::PN, PrimitiveValue::from("")));
    obj.put(DataElement::new(Tag(0x0010, 0x0020), VR::LO, PrimitiveValue::from("")));
    obj.put(DataElement::new(Tag(0x0020, 0x000D), VR::UI, PrimitiveValue::from("2.25.99999932")));
    obj.put(DataElement::new(Tag(0x0020, 0x000E), VR::UI, PrimitiveValue::from("2.25.99999933")));
    obj.put(DataElement::new(Tag(0x0028, 0x0010), VR::US, PrimitiveValue::from(height as u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0011), VR::US, PrimitiveValue::from(width as u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0100), VR::US, PrimitiveValue::from(8u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0101), VR::US, PrimitiveValue::from(8u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0102), VR::US, PrimitiveValue::from(7u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0103), VR::US, PrimitiveValue::from(0u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0002), VR::US, PrimitiveValue::from(1u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0004), VR::CS, PrimitiveValue::from("MONOCHROME2")));
    obj.put(DataElement::new(Tag(0x0028, 0x0008), VR::IS, PrimitiveValue::from("1")));
    obj.put(DataElement::new(Tag(0x0028, 0x1053), VR::DS, PrimitiveValue::from("1.000000")));
    obj.put(DataElement::new(Tag(0x0028, 0x1052), VR::DS, PrimitiveValue::from("0.000000")));
    obj.put(DataElement::new(Tag(0x7FE0, 0x0010), VR::OB, pfs));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                .media_storage_sop_instance_uid("2.25.99999931")
                .transfer_syntax("1.2.840.10008.1.2.4.80"),
        )
        .expect("FileMetaTableBuilder failed");
    file_obj.write_to_file(path).expect("write_to_file failed");
}

/// JPEG-LS Lossless round-trip: a CharLS-produced bitstream is decoded by
/// the RITK-native lossless decoder with exact per-sample fidelity.
#[test]
fn test_decode_compressed_frame_jpegls_lossless_round_trip() {
    let width = 4u32;
    let height = 1u32;
    let original: Vec<u8> = vec![0, 42, 85, 127];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpegls_lossless.dcm");
    write_jpegls_lossless_dicom_file(&path, width, height, &original);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
        .expect("JPEG-LS Lossless decode must succeed for CharLS conformance fixture");

    let expected = original
        .iter()
        .map(|&sample| f32::from(sample))
        .collect::<Vec<_>>();
    assert_eq!(decoded, expected, "JPEG-LS Lossless must preserve every sample exactly");
}

/// JPEG-LS Lossless multi-row round-trip: a CharLS-produced scan containing
/// JPEG-LS stuffed bits is decoded by the native backend without bit drift.
#[test]
fn test_decode_compressed_frame_jpegls_lossless_multirow_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        0, 42, 85, 127, 128, 170, 200, 225, 10, 20, 30, 40, 240, 230, 220, 210,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpegls_lossless_multirow.dcm");
    write_jpegls_lossless_dicom_file(&path, width, height, &original);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
        .expect("JPEG-LS Lossless decode must succeed for multi-row CharLS fixture");

    let expected = original
        .iter()
        .map(|&sample| f32::from(sample))
        .collect::<Vec<_>>();
    assert_eq!(decoded, expected, "JPEG-LS Lossless multi-row decode must preserve every sample exactly");
}

/// JPEG-LS Near-Lossless round-trip: encode known pixel values with NEAR=2, decode via
/// codec, verify per-pixel reconstruction error ≤ NEAR per ISO 14495-1.
///
/// Mathematical justification:
/// JPEG-LS near-lossless (NEAR = 2) guarantees |s' − s| ≤ 2 for all pixels per
/// ISO 14495-1 §A.2. Tolerance set to exactly 2.0 (the analytical bound).
#[test]
fn test_decode_compressed_frame_jpegls_near_lossless_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        10, 50, 100, 150, 200, 245, 30, 80, 130, 180, 220, 60, 110, 160, 210, 40,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpegls_nearlossless.dcm");

    let frame_info = charls::FrameInfo {
        width,
        height,
        bits_per_sample: 8,
        component_count: 1,
    };
    let mut codec = charls::CharLS::default();
    let jls_bytes = codec
        .encode(frame_info, 2, &original)
        .expect("CharLS near-lossless encode failed");

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jls_bytes]);
    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(Tag(0x0008, 0x0016), VR::UI, PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3")));
    obj.put(DataElement::new(Tag(0x0008, 0x0018), VR::UI, PrimitiveValue::from("2.25.99999941")));
    obj.put(DataElement::new(Tag(0x0010, 0x0010), VR::PN, PrimitiveValue::from("")));
    obj.put(DataElement::new(Tag(0x0010, 0x0020), VR::LO, PrimitiveValue::from("")));
    obj.put(DataElement::new(Tag(0x0020, 0x000D), VR::UI, PrimitiveValue::from("2.25.99999942")));
    obj.put(DataElement::new(Tag(0x0020, 0x000E), VR::UI, PrimitiveValue::from("2.25.99999943")));
    obj.put(DataElement::new(Tag(0x0028, 0x0010), VR::US, PrimitiveValue::from(height as u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0011), VR::US, PrimitiveValue::from(width as u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0100), VR::US, PrimitiveValue::from(8u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0101), VR::US, PrimitiveValue::from(8u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0102), VR::US, PrimitiveValue::from(7u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0103), VR::US, PrimitiveValue::from(0u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0002), VR::US, PrimitiveValue::from(1u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0004), VR::CS, PrimitiveValue::from("MONOCHROME2")));
    obj.put(DataElement::new(Tag(0x0028, 0x0008), VR::IS, PrimitiveValue::from("1")));
    obj.put(DataElement::new(Tag(0x0028, 0x1053), VR::DS, PrimitiveValue::from("1.000000")));
    obj.put(DataElement::new(Tag(0x0028, 0x1052), VR::DS, PrimitiveValue::from("0.000000")));
    obj.put(DataElement::new(Tag(0x7FE0, 0x0010), VR::OB, pfs));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                .media_storage_sop_instance_uid("2.25.99999941")
                .transfer_syntax("1.2.840.10008.1.2.4.81"),
        )
        .expect("FileMetaTableBuilder failed");
    file_obj.write_to_file(&path).expect("write_to_file failed");

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
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
