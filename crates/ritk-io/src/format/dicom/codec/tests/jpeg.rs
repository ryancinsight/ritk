use super::super::decode_compressed_frame;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{FileMetaTableBuilder, InMemDicomObject};

/// Build and write a minimal JPEG Baseline DICOM Part 10 file.
///
/// Pixel data is JPEG-encoded at default quality using the `image` crate and
/// encapsulated as a single fragment per DICOM PS3.5 §A.4.
///
/// # Parameters
/// - `path`: destination file path.
/// - `width`, `height`: image dimensions in pixels.
/// - `pixels_u8`: flat row-major 8-bit grayscale values, length = `width × height`.
fn write_jpeg_dicom_file(path: &std::path::Path, width: u32, height: u32, pixels_u8: &[u8]) {
    assert_eq!(
        pixels_u8.len(),
        (width * height) as usize,
        "pixels_u8 length must equal width × height"
    );

    use image::{DynamicImage, GrayImage};
    let gray = GrayImage::from_raw(width, height, pixels_u8.to_vec())
        .expect("GrayImage::from_raw failed");
    let dyn_img = DynamicImage::ImageLuma8(gray);
    let mut jpeg_bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
    dyn_img
        .write_to(&mut cursor, image::ImageFormat::Jpeg)
        .expect("JPEG encode failed");
    drop(cursor);

    let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jpeg_bytes]);
    let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(Tag(0x0008, 0x0016), VR::UI, PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3")));
    obj.put(DataElement::new(Tag(0x0008, 0x0018), VR::UI, PrimitiveValue::from("2.25.99999901")));
    obj.put(DataElement::new(Tag(0x0010, 0x0010), VR::PN, PrimitiveValue::from("")));
    obj.put(DataElement::new(Tag(0x0010, 0x0020), VR::LO, PrimitiveValue::from("")));
    obj.put(DataElement::new(Tag(0x0020, 0x000D), VR::UI, PrimitiveValue::from("2.25.99999902")));
    obj.put(DataElement::new(Tag(0x0020, 0x000E), VR::UI, PrimitiveValue::from("2.25.99999903")));
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
                .media_storage_sop_instance_uid("2.25.99999901")
                .transfer_syntax("1.2.840.10008.1.2.4.50"),
        )
        .expect("FileMetaTableBuilder failed");
    file_obj.write_to_file(path).expect("write_to_file failed");
}

/// JPEG Baseline round-trip: encode known pixel values, decode via codec, verify
/// each decoded value is within JPEG quantization tolerance of the original.
///
/// Mathematical justification:
/// At JPEG quality 75 the luminance quantization matrix is scaled by factor 0.5
/// from the standard reference table (ITU T.81 Annex K). Per-pixel reconstruction
/// error from a single quantized DCT coefficient (u,v) with step Q[u,v] via the
/// 2D IDCT is bounded by Q[u,v]/2 (one quantization half-step). The dominant
/// contributions for a smooth-gradient image come from the DC and primary AC terms:
///   DC  (0,0): Q = 8  → ≤ 4 per pixel
///   AC  (1,0): Q = 6  → ≤ 3 per pixel
///   AC  (0,1): Q = 6  → ≤ 3 per pixel
///   AC  (1,1): Q = 6  → ≤ 3 per pixel (activated by 4×4→8×8 edge replication)
/// Sum of primary contributors: 4+3+3+3 = 13. Tolerance set to 16 (next integer
/// power-of-2 ≥ 13) to accommodate higher-order AC contributions and fixed-point
/// IDCT rounding in `jpeg-decoder`.
#[test]
fn test_decode_compressed_frame_jpeg_baseline_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        50, 100, 150, 200, 75, 125, 175, 225, 30, 80, 130, 180, 20, 60, 100, 140,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpeg_baseline.dcm");
    write_jpeg_dicom_file(&path, width, height, &original);

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
        .expect("decode_compressed_frame must succeed for JPEG Baseline");

    assert_eq!(decoded.len(), (width * height) as usize, "decoded pixel count must equal width × height");
    for &v in &decoded {
        assert!((0.0..=255.0).contains(&v), "decoded value {v} is outside valid 8-bit range [0, 255]");
    }
    let max_error = original
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dec)| (orig as f32 - dec).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 16.0,
        "JPEG decode error {max_error} exceeds analytical tolerance of 16.0 intensity units \
         (Q75: DC≤4 + AC(1,0)≤3 + AC(0,1)≤3 + AC(1,1)≤3 + higher-order margin = 16)"
    );
}

/// Rescale invariant: Output[i] = raw_sample[i] × slope + intercept.
///
/// Uses a uniform 4×4 patch to isolate the rescale from JPEG spatial
/// quantization effects.
#[test]
fn test_decode_compressed_frame_rescale_contract() {
    let width = 4u32;
    let height = 4u32;
    let pixels: Vec<u8> = vec![128u8; 16];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpeg_rescale.dcm");
    write_jpeg_dicom_file(&path, width, height, &pixels);

    let obj = dicom::object::open_file(&path).expect("open_file");
    let base = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0).expect("identity rescale decode");
    let scaled = decode_compressed_frame(&obj, 0, 8, 0, 2.0, 10.0).expect("slope=2 intercept=10 decode");

    assert_eq!(base.len(), 16, "base must have 16 elements");
    assert_eq!(scaled.len(), 16, "scaled must have 16 elements");
    for (i, (&b, &s)) in base.iter().zip(scaled.iter()).enumerate() {
        let expected = b * 2.0 + 10.0;
        assert!(
            (s - expected).abs() < 0.01,
            "pixel[{i}]: rescale invariant violated: got {s}, expected {b} × 2.0 + 10.0 = {expected}"
        );
    }
}

/// JPEG Extended round-trip: encode known pixel values with the JPEG codec, declare
/// Transfer Syntax as JPEG Extended (1.2.840.10008.1.2.4.51), decode via codec, verify
/// each decoded value is within JPEG quantization tolerance of the original.
///
/// Mathematical justification:
/// JPEG Extended (Process 2 & 4) uses the same DCT + quantization architecture as
/// Baseline (Process 1) but supports 12-bit samples. For 8-bit input encoded with
/// jpeg-encoder at default quality, the same DC/AC quantization bounds apply:
///   DC  (0,0): Q = 8  → ≤ 4 per pixel
///   AC  (1,0): Q = 6  → ≤ 3 per pixel
///   AC  (0,1): Q = 6  → ≤ 3 per pixel
///   AC  (1,1): Q = 6  → ≤ 3 per pixel
/// Sum = 13; tolerance set to 16 (next integer power-of-2 ≥ 13).
/// The codec registered for JPEG Extended uses the same `jpeg-decoder` path as Baseline.
#[test]
fn test_decode_compressed_frame_jpeg_extended_round_trip() {
    let width = 4u32;
    let height = 4u32;
    let original: Vec<u8> = vec![
        50, 100, 150, 200, 75, 125, 175, 225, 30, 80, 130, 180, 20, 60, 100, 140,
    ];
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_jpeg_extended.dcm");

    {
        use image::{DynamicImage, GrayImage};
        let gray = GrayImage::from_raw(width, height, original.clone())
            .expect("GrayImage::from_raw failed");
        let dyn_img = DynamicImage::ImageLuma8(gray);
        let mut jpeg_bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
        dyn_img
            .write_to(&mut cursor, image::ImageFormat::Jpeg)
            .expect("JPEG encode failed");
        drop(cursor);

        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jpeg_bytes]);
        let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(Tag(0x0008, 0x0016), VR::UI, PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3")));
        obj.put(DataElement::new(Tag(0x0008, 0x0018), VR::UI, PrimitiveValue::from("2.25.99999921")));
        obj.put(DataElement::new(Tag(0x0010, 0x0010), VR::PN, PrimitiveValue::from("")));
        obj.put(DataElement::new(Tag(0x0010, 0x0020), VR::LO, PrimitiveValue::from("")));
        obj.put(DataElement::new(Tag(0x0020, 0x000D), VR::UI, PrimitiveValue::from("2.25.99999922")));
        obj.put(DataElement::new(Tag(0x0020, 0x000E), VR::UI, PrimitiveValue::from("2.25.99999923")));
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
                    .media_storage_sop_instance_uid("2.25.99999921")
                    .transfer_syntax("1.2.840.10008.1.2.4.51"),
            )
            .expect("FileMetaTableBuilder failed");
        file_obj.write_to_file(&path).expect("write_to_file failed");
    }

    let obj = dicom::object::open_file(&path).expect("open_file failed");
    let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
        .expect("decode_compressed_frame must succeed for JPEG Extended");

    assert_eq!(decoded.len(), (width * height) as usize, "decoded pixel count must equal width × height");
    for &v in &decoded {
        assert!((0.0..=255.0).contains(&v), "decoded value {v} is outside valid 8-bit range [0, 255]");
    }
    let max_error = original
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dec)| (orig as f32 - dec).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 16.0,
        "JPEG Extended decode error {max_error} exceeds analytical tolerance of 16.0 \
         (Q75: DC≤4 + AC(1,0)≤3 + AC(0,1)≤3 + AC(1,1)≤3 + higher-order margin = 16)"
    );
}
