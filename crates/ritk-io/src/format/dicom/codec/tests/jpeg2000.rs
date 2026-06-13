use super::super::decode_compressed_frame;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};
use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, WaveletTransform};
use ritk_codecs::PixelSignedness;

/// Build and write a minimal JPEG 2000 Lossless DICOM Part 10 file.
///
/// Pixel data is encoded as a bare J2K codestream via the RITK-native pure-Rust
/// encoder (`ritk_codecs::jpeg_2000::encoder`).  The 5/3 reversible wavelet is
/// used with two DWT decomposition levels, producing lossless output.
///
/// This replaces the former `openjp2`/`openjpeg-sys` FFI encoder, eliminating
/// the Windows heap-corruption abort (0xC0000374) that manifested in the C
/// OpenJPEG runtime when encoding small frames.
fn write_jpeg2000_lossless_dicom_file(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels_u16: &[u16],
) {
    assert_eq!(
        pixels_u16.len(),
        (width * height) as usize,
        "pixels_u16 length must equal width × height"
    );

    // Convert u16 pixels to i32 for the encoder (lossless, no narrowing).
    let pixels_i32: Vec<i32> = pixels_u16.iter().map(|&v| v as i32).collect();

    let j2k_bytes =
        encode_grayscale_j2k(&pixels_i32, height, width, 16, PixelSignedness::Unsigned, 2, WaveletTransform::Reversible);

    assert!(
        j2k_bytes.len() >= 4,
        "J2K encoded codestream must be non-trivial"
    );
    assert_eq!(
        j2k_bytes[..2],
        [0xFF, 0x4F],
        "encoded J2K codestream must start with SOC 0xFF4F"
    );

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![j2k_bytes]);
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
        PrimitiveValue::from("2.25.99999951"),
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
        PrimitiveValue::from("2.25.99999952"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from("2.25.99999953"),
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
        PrimitiveValue::from(16u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(16u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(15u16),
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
                .media_storage_sop_instance_uid("2.25.99999951")
                .transfer_syntax("1.2.840.10008.1.2.4.90"),
        )
        .expect("FileMetaTableBuilder failed");
    file_obj.write_to_file(path).expect("write_to_file failed");
}

/// JPEG 2000 Lossless round-trip: encode known 16-bit pixel values, decode via codec,
/// verify exact pixel equality (lossless invariant).
///
/// # Mathematical justification
/// JPEG 2000 Lossless (TS 1.2.840.10008.1.2.4.90) uses the 5/3 reversible integer
/// wavelet transform with lossless coding.  Per ISO 15444-1 §C.5.5.1, irreversible=0
/// ⟹ exact reconstruction: |decoded[i] − original[i]| = 0 for all i.
///
/// Evidence tier: round-trip differential test — encoder and decoder are separate
/// code paths; the lossless invariant is verified by asserting max_error = 0.
#[test]
fn test_decode_compressed_frame_jpeg2000_lossless_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u16> = vec![
        0, 256, 512, 1024, 2048, 3071, 3584, 3840, 100, 200, 400, 800, 1600, 2400, 3000, 4095,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpeg2000_lossless.dcm");
    write_jpeg2000_lossless_dicom_file(&path, width, height, &original);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded =
        decode_compressed_frame(&obj, 0, 16, ritk_dicom::PixelSignedness::Unsigned, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG 2000 Lossless");

    assert_eq!(decoded.len(), 16, "decoded pixel count must equal 16");
    for (i, &v) in decoded.iter().enumerate() {
        assert!(
            (0.0..=4095.0).contains(&v),
            "decoded[{i}] = {v} is outside valid range [0, 4095]"
        );
    }
    let max_error = original
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dec)| (orig as f32 - dec).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_error, 0.0,
        "JPEG 2000 Lossless decode error {max_error} must be exactly 0 \
         (ISO 15444-1 §C.5.5.1: 5/3 reversible wavelet ⟹ |S'[i]−S[i]| = 0)"
    );
    for (i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig as f32,
            dec,
            "pixel[{i}]: expected {orig}, got {dec} — JPEG 2000 lossless must preserve all sample values"
        );
    }
}
