use super::super::decode_compressed_frame;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};

/// DICOM RLE Lossless PackBits encoder per DICOM PS3.5 Annex G.3.1.
///
/// Specification:
/// - Header byte h encodes run type:
///   - h ∈ [0, 127]: literal run of (h + 1) bytes.
///   - h ∈ [129, 255]: repeat run of (257 − h) copies of the following byte.
///   - h = 128 (0x80): no-op.
/// - Repeat count ∈ [2, 128], header = 257 − count ∈ [129, 255].
/// - Literal count ∈ [1, 128], header = count − 1 ∈ [0, 127].
/// - Segment padded to even length per PS3.5 Annex G.3.2.
fn packbits_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 128 + 2);
    let mut i = 0;
    while i < data.len() {
        let mut repeat = 1usize;
        while i + repeat < data.len() && data[i + repeat] == data[i] && repeat < 128 {
            repeat += 1;
        }
        if repeat >= 2 {
            out.push((257 - repeat) as u8);
            out.push(data[i]);
            i += repeat;
            continue;
        }
        let lit_start = i;
        let mut lit = 1usize;
        while i + lit < data.len() && lit < 128 {
            if i + lit + 1 < data.len() && data[i + lit] == data[i + lit + 1] {
                break;
            }
            lit += 1;
        }
        out.push((lit - 1) as u8);
        out.extend_from_slice(&data[lit_start..lit_start + lit]);
        i += lit;
    }
    if out.len() % 2 != 0 {
        out.push(0x00);
    }
    out
}

/// Assemble a single-segment DICOM RLE Lossless fragment for 8-bit single-channel data.
///
/// Per DICOM PS3.5 Annex G.4.1, the fragment layout is:
///   [RLE Header: 64 bytes] [Segment 0: PackBits-encoded pixel bytes]
///
/// RLE Header (16 × uint32 LE):
///   header[0] = 1  (one segment)
///   header[1] = 64 (segment 0 offset = header size)
///   header[2..15] = 0 (unused)
fn build_rle_fragment_8bit(pixels: &[u8]) -> Vec<u8> {
    let segment = packbits_encode(pixels);
    const HEADER_BYTES: usize = 64;
    let mut header = [0u32; 16];
    header[0] = 1;
    header[1] = HEADER_BYTES as u32;
    let mut out = Vec::with_capacity(HEADER_BYTES + segment.len());
    for &w in &header {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&segment);
    out
}

/// RLE Lossless round-trip: encode known pixel values using DICOM RLE (PS3.5 Annex G),
/// decode via the registered codec, verify exact pixel equality.
///
/// Mathematical justification:
/// DICOM RLE Lossless uses the PackBits algorithm applied per byte plane. PackBits is a
/// lossless compression scheme: for any input S, decode(encode(S)) = S exactly.
/// Therefore max|decoded[i] − original[i]| = 0 for any integer sample sequence.
///
/// The native decoder (`decode_rle_lossless_frame`) implements PS3.5 Annex G correctly
/// for all pixel values. All N=16 pixels are encoded and the native decoder recovers
/// all N pixels exactly — no offset-compensation or restricted pixel values required.
///
/// The encoded slice exercises both PackBits run types:
/// - [0, 50, 50, 50]         — literal followed by 3-repetition repeat run.
/// - [75, 80, 85, 90]        — 4-element literal run.
/// - [100, 100, 100, 100]    — 4-repetition repeat run.
/// - [120, 130, 140, 150]    — 4-element literal run.
#[test]
fn test_decode_compressed_frame_rle_lossless_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        0, 50, 50, 50, 75, 80, 85, 90, 100, 100, 100, 100, 120, 130, 140, 150,
    ];

    let rle_fragment = build_rle_fragment_8bit(&original);
    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![rle_fragment]);
    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_rle_lossless.dcm");

    {
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.99999931"),
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
                    .media_storage_sop_instance_uid("2.25.99999931")
                    .transfer_syntax("1.2.840.10008.1.2.5"),
            )
            .expect("FileMetaTableBuilder failed");
        file_obj.write_to_file(&path).expect("write_to_file failed");
    }

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 8, ritk_dicom::PixelSignedness::Unsigned, 1.0, 0.0)
        .expect("decode_compressed_frame must succeed for RLE Lossless");

    assert_eq!(
        decoded.len(),
        (width * height) as usize,
        "decoded pixel count must equal width × height"
    );
    for &v in &decoded {
        assert!(
            (0.0..=255.0).contains(&v),
            "decoded value {v} is outside valid 8-bit range [0, 255]"
        );
    }
    let max_error = original
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dec)| (orig as f32 - dec).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_error, 0.0,
        "RLE Lossless decode error {max_error} must be exactly 0 (PackBits is lossless per PS3.5 G.3.1)"
    );
}

/// RLE Lossless round-trip with unrestricted pixel values — `pixel[0] = 42` (≠ 0).
///
/// Mathematical justification:
/// PackBits is lossless: `decode(encode(S)) = S` for all `S: &[u8]`. The native
/// `decode_rle_lossless_frame` decoder implements this correctly for all pixel values,
/// including `pixel[0] ≠ 0`.
///
/// Upstream failure mode (dicom-transfer-syntax-registry v0.8.2):
/// The upstream decoder forces `dst[0] = 0` regardless of encoded content, so for
/// `pixel[0] = 42`, the upstream decoder would produce `decoded[0] = 0.0 ≠ 42.0`.
/// This test would FAIL with the upstream decoder and MUST pass with the native decoder.
///
/// Formal invariant: `∀i ∈ [0, N−1]: decoded[i] = original[i]`
///   ⟹ `max|decoded[i] − original[i]| = 0`
#[test]
fn test_decode_compressed_frame_rle_lossless_unrestricted_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        42, 50, 50, 50, 75, 80, 85, 90, 100, 100, 100, 100, 120, 130, 140, 150,
    ];

    let rle_fragment = build_rle_fragment_8bit(&original);
    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![rle_fragment]);
    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_rle_unrestricted.dcm");

    {
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.99999941"),
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
            PrimitiveValue::from("2.25.99999942"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.99999943"),
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
                    .media_storage_sop_instance_uid("2.25.99999941")
                    .transfer_syntax("1.2.840.10008.1.2.5"),
            )
            .expect("FileMetaTableBuilder failed");
        file_obj.write_to_file(&path).expect("write_to_file failed");
    }

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 8, ritk_dicom::PixelSignedness::Unsigned, 1.0, 0.0)
        .expect("decode_compressed_frame must succeed for RLE Lossless with native decoder");

    assert_eq!(
        decoded.len(),
        (width * height) as usize,
        "decoded pixel count must equal width × height"
    );
    assert_eq!(
        decoded[0], 42.0f32,
        "pixel[0] must be 42.0; upstream decoder forces this to 0.0 (off-by-one write start)"
    );
    for &v in &decoded {
        assert!(
            (0.0..=255.0).contains(&v),
            "decoded value {v} is outside valid 8-bit range [0, 255]"
        );
    }
    let max_error = original
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dec)| (orig as f32 - dec).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_error, 0.0,
        "RLE Lossless native decode error {max_error} must be exactly 0 (lossless per PS3.5 G.3.1)"
    );
}
