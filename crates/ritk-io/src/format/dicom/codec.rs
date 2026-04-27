//! DICOM pixel data codec integration via `dicom-pixeldata`.
//!
//! # Architecture
//!
//! Provides `decode_compressed_frame`, the single dispatch point for all
//! codec-supported compressed transfer syntaxes. Calls
//! `dicom_pixeldata::PixelDecoder::decode_pixel_data_frame` to recover raw
//! sample bytes, then applies the linear modality LUT via `decode_pixel_bytes`.
//!
//! # Supported codecs (pure Rust, `native` feature of `dicom-pixeldata`)
//!
//! | Transfer Syntax                      | UID                     | Codec        |
//! |--------------------------------------|-------------------------|--------------|
//! | JPEG Baseline                        | 1.2.840.10008.1.2.4.50  | jpeg-decoder |
//! | JPEG Lossless First-Order Prediction | 1.2.840.10008.1.2.4.70  | jpeg-decoder |
//! | RLE Lossless                         | 1.2.840.10008.1.2.5     | dicom-rle    |
//!
//! # Extension points
//!
//! - JPEG-LS (Lossless + Near-Lossless): enable `charls` feature and add
//!   `JpegLsLossless | JpegLsLossy` to `TransferSyntaxKind::is_codec_supported()`.
//! - JPEG 2000 (Lossless + Lossy): enable `openjp2` feature and add
//!   `Jpeg2000Lossless | Jpeg2000Lossy` to `TransferSyntaxKind::is_codec_supported()`.
//!
//! # Mathematical contract
//!
//! `decode_compressed_frame(obj, f, bits, repr, slope, intercept)`:
//!   `Output[i] = codec_sample[i] × slope + intercept`
//!
//! where `codec_sample[i]` is the integer produced by the codec for pixel i.
//! Identical semantics to `decode_pixel_bytes` (DICOM PS3.3 C.7.6.3.1.4).
//!
//! # Invariants
//!
//! - `is_codec_supported() ⟹ is_compressed()`: codec path only for compressed TS.
//! - `is_natively_supported() ⟹ !is_codec_supported()`: disjoint decode paths.
//! - Output length equals `rows × cols` for a single-frame decode.
//! - Rescale is always applied; identity rescale (slope=1, intercept=0) is valid.

use anyhow::{Context, Result};
use dicom::object::DefaultDicomObject;
use dicom_pixeldata::PixelDecoder;

use super::reader::decode_pixel_bytes;

/// Decode one frame from a compressed DICOM object using the registered codec.
///
/// # Arguments
///
/// - `obj`: open Part 10 DICOM object with compressed transfer syntax in file meta.
/// - `frame_idx`: zero-based frame index (0 for single-frame objects).
/// - `bits_allocated`: from (0028,0100); drives byte interpretation in `decode_pixel_bytes`.
/// - `pixel_representation`: from (0028,0103); 0 = unsigned, 1 = signed.
/// - `slope`: RescaleSlope from (0028,1053); absent ⇒ 1.0.
/// - `intercept`: RescaleIntercept from (0028,1052); absent ⇒ 0.0.
///
/// # Returns
///
/// `Vec<f32>` of length `rows × cols` with modality LUT applied.
///
/// # Errors
///
/// Returns `Err` when the codec fails: unsupported TS, malformed compressed data,
/// or missing codec (feature not enabled).
pub(super) fn decode_compressed_frame(
    obj: &DefaultDicomObject,
    frame_idx: u32,
    bits_allocated: u16,
    pixel_representation: u16,
    slope: f32,
    intercept: f32,
) -> Result<Vec<f32>> {
    let decoded = obj
        .decode_pixel_data_frame(frame_idx)
        .with_context(|| format!("codec decode failed for frame {frame_idx}"))?;
    let raw = decoded.data();
    Ok(decode_pixel_bytes(
        raw,
        bits_allocated,
        pixel_representation,
        slope,
        intercept,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
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

        // JPEG-encode via image crate (JFIF SOF0 Baseline).
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

        // Encapsulate as single fragment per DICOM PS3.5 §A.4.
        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jpeg_bytes]);
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
            PrimitiveValue::from("2.25.99999901"),
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
            PrimitiveValue::from("2.25.99999902"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.99999903"),
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
        // Values span [20, 225] to exercise the full 8-bit range.
        let original: Vec<u8> = vec![
            50, 100, 150, 200, 75, 125, 175, 225, 30, 80, 130, 180, 20, 60, 100, 140,
        ];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jpeg_baseline.dcm");
        write_jpeg_dicom_file(&path, width, height, &original);

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG Baseline");

        assert_eq!(
            decoded.len(),
            (width * height) as usize,
            "decoded pixel count must equal width × height"
        );

        // All decoded values must lie in [0, 255].
        for &v in &decoded {
            assert!(
                (0.0..=255.0).contains(&v),
                "decoded value {v} is outside valid 8-bit range [0, 255]"
            );
        }

        // Each decoded value must be within JPEG tolerance of the original.
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
    /// quantization effects. The decoded raw sample for a uniform patch is
    /// constant; the scaled output must equal raw × slope + intercept.
    #[test]
    fn test_decode_compressed_frame_rescale_contract() {
        let width = 4u32;
        let height = 4u32;
        // Uniform value minimises intra-patch JPEG quantization variation.
        let pixels: Vec<u8> = vec![128u8; 16];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jpeg_rescale.dcm");
        write_jpeg_dicom_file(&path, width, height, &pixels);

        let obj = dicom::object::open_file(&path).expect("open_file");
        let base =
            decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0).expect("identity rescale decode");
        let scaled =
            decode_compressed_frame(&obj, 0, 8, 0, 2.0, 10.0).expect("slope=2 intercept=10 decode");

        assert_eq!(base.len(), 16, "base must have 16 elements");
        assert_eq!(scaled.len(), 16, "scaled must have 16 elements");

        // For each pixel: scaled[i] == base[i] * 2.0 + 10.0 (within float epsilon).
        for (i, (&b, &s)) in base.iter().zip(scaled.iter()).enumerate() {
            let expected = b * 2.0 + 10.0;
            assert!(
                (s - expected).abs() < 0.01,
                "pixel[{i}]: rescale invariant violated: got {s}, expected {b} × 2.0 + 10.0 = {expected}"
            );
        }
    }

    /// Build and write a minimal JPEG XL Lossless DICOM Part 10 file.
    ///
    /// Pixel data is JXL-encoded losslessly using `zune-jpegxl` (ISO 18181-1 modular path)
    /// and encapsulated as a single fragment per DICOM PS3.5 §A.4.
    ///
    /// # Parameters
    /// - `path`: destination file path.
    /// - `width`, `height`: image dimensions in pixels.
    /// - `pixels_u8`: flat row-major 8-bit grayscale values, length = `width × height`.
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

        // JXL-encode losslessly via zune-jpegxl.
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

        // Encapsulate as single fragment per DICOM PS3.5 §A.4.
        let fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::from_vec(vec![jxl_bytes]);
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
            PrimitiveValue::from("2.25.99999911"),
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
            PrimitiveValue::from("2.25.99999912"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.99999913"),
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
        // Values span [20, 225] to exercise the full 8-bit range.
        let original: Vec<u8> = vec![
            50, 100, 150, 200, 75, 125, 175, 225, 30, 80, 130, 180, 20, 60, 100, 140,
        ];
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_jxl_lossless.dcm");
        write_jxl_lossless_dicom_file(&path, width, height, &original);

        let obj = dicom::object::open_file(&path).expect("open_file failed");
        let decoded = decode_compressed_frame(&obj, 0, 8, 0, 1.0, 0.0)
            .expect("decode_compressed_frame must succeed for JPEG XL Lossless");

        assert_eq!(
            decoded.len(),
            (width * height) as usize,
            "decoded pixel count must equal width × height"
        );

        // JXL Lossless: every decoded value must lie in [0, 255].
        for &v in &decoded {
            assert!(
                (0.0..=255.0).contains(&v),
                "decoded value {v} is outside valid 8-bit range [0, 255]"
            );
        }

        // JXL Lossless invariant: per-pixel error must be exactly 0.
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
}
