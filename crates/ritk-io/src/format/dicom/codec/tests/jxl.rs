use super::super::decode_compressed_frame;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};

/// Build and write a minimal JPEG XL Lossless DICOM Part 10 file.
///
/// Pixel data is JXL-encoded losslessly using `zune-jpegxl` (ISO 18181-1 modular path)
/// and encapsulated as a single fragment per DICOM PS3.5 §A.4.
fn write_jxl_lossless_dicom_file(
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

    use zune_core::bit_depth::BitDepth;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::EncoderOptions;
    use zune_jpegxl::JxlSimpleEncoder;
    let options = EncoderOptions::new(
        width as usize,
        height as usize,
        ColorSpace::Luma,
        BitDepth::Eight,
    );
    let encoder = JxlSimpleEncoder::new(pixels_u8, options);
    let jxl_bytes = encoder.encode().expect("JXL encode failed");

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jxl_bytes]);
    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(Tag(0x0008, 0x0016), VR::UI, PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3")));
    obj.put(DataElement::new(Tag(0x0008, 0x0018), VR::UI, PrimitiveValue::from("2.25.99999911")));
    obj.put(DataElement::new(Tag(0x0010, 0x0010), VR::PN, PrimitiveValue::from("")));
    obj.put(DataElement::new(Tag(0x0010, 0x0020), VR::LO, PrimitiveValue::from("")));
    obj.put(DataElement::new(Tag(0x0020, 0x000D), VR::UI, PrimitiveValue::from("2.25.99999912")));
    obj.put(DataElement::new(Tag(0x0020, 0x000E), VR::UI, PrimitiveValue::from("2.25.99999913")));
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
                .media_storage_sop_instance_uid("2.25.99999911")
                .transfer_syntax("1.2.840.10008.1.2.4.110"),
        )
        .expect("FileMetaTableBuilder failed");
    file_obj.write_to_file(path).expect("write_to_file failed");
}

/// JPEG XL Lossless round-trip: encode known pixel values, decode via codec, verify
/// exact pixel equality (lossless invariant: no information loss).
///
/// Mathematical justification:
/// JXL Lossless uses the modular codec path (ISO 18181-1 §9) which is provably lossless.
/// Given integer input samples S[i] ∈ [0, 255]:
///   Encode: JXL_Lossless(S) → bitstream B
///   Decode: JXL_Decode(B) → S' where S'[i] = S[i] for all i.
/// Max error = max|S[i] - S'[i]| = 0.
#[test]
fn test_decode_compressed_frame_jxl_lossless_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        50, 100, 150, 200, 75, 125, 175, 225, 30, 80, 130, 180, 20, 60, 100, 140,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jxl_lossless.dcm");
    write_jxl_lossless_dicom_file(&path, width, height, &original);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
        .expect("decode_compressed_frame must succeed for JPEG XL Lossless");

    assert_eq!(decoded.len(), (width * height) as usize, "decoded pixel count must equal width × height");
    for &v in &decoded {
        assert!((0.0..=255.0).contains(&v), "decoded value {v} is outside valid 8-bit range [0, 255]");
    }
    let max_error = original
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dec)| (orig as f32 - dec).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_error, 0.0,
        "JPEG XL Lossless decode error {max_error} must be exactly 0 \
         (JXL modular path preserves integer sample values exactly)"
    );
}
