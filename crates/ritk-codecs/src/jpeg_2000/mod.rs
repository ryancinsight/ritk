//! Native JPEG 2000 (ISO 15444-1) decoder for DICOM encapsulated frames.
//!
//! # Architecture
//! - [`jpeg2k`] owns safe codestream decode.
//! - `jpeg2k` is compiled with its `openjp2` backend, a Rust port of OpenJPEG,
//!   not the `openjpeg-sys` C FFI backend.
//! - [`image`] validates decoded component planes against DICOM pixel metadata
//!   and applies the DICOM modality LUT.
//!
//! # Specification (ISO 15444-1 / DICOM PS3.5)
//! DICOM JPEG 2000 encapsulates a raw J2K codestream (not a JP2 file wrapper):
//! - Transfer Syntax 1.2.840.10008.1.2.4.90: JPEG 2000 Lossless Only.
//! - Transfer Syntax 1.2.840.10008.1.2.4.91: JPEG 2000 lossy or lossless.
//!
//! A valid J2K codestream begins with SOC (0xFF4F).  JP2 wrappers are rejected
//! before decode because DICOM transfer syntaxes 1.2.840.10008.1.2.4.90/91 store
//! a bare codestream fragment.
//!
//! # Decode path
//! 1. Validate SOC marker at byte 0.
//! 2. Decode the codestream through `jpeg2k::Image` with full resolution and all
//!    quality layers.
//! 3. Extract raw `i32` component samples via `ImageComponent::data`.
//! 4. Validate component count, dimensions, precision, and signedness against
//!    [`PixelLayout`].
//! 5. Apply DICOM PS3.3 §C.7.6.3.1 modality LUT:
//!    `output = stored_integer × rescale_slope + rescale_intercept`.

mod image;
#[cfg(test)]
mod test_support;

use anyhow::{bail, Context, Result};
use jpeg2k::{DecodeParameters, Image};

use crate::PixelLayout;

/// J2K Start of Codestream marker (ISO 15444-1 §A.3): bytes `0xFF 0x4F`.
pub(crate) const SOC: u16 = 0xFF4F;

/// JPEG / JFIF Start of Image marker, shared with JPEG-LS (0xFFD8).
/// Presence of SOI instead of SOC indicates another JPEG variant, not a bare
/// J2K codestream.
#[allow(dead_code)]
pub(crate) const SOI: u16 = 0xFFD8;

/// Decode a DICOM-encapsulated JPEG 2000 J2K codestream fragment.
///
/// # Arguments
/// - `fragment`: raw bytes of the encapsulated pixel data item.
/// - `layout`: pixel geometry and DICOM rescale parameters.
///
/// # Errors
/// Returns an error if:
/// - `fragment` does not begin with the SOC marker (0xFF4F).
/// - the JPEG 2000 backend fails to parse or decode the codestream.
/// - decoded component metadata does not match `layout`.
pub fn decode_jpeg2000_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    if !is_jpeg2000_codestream(fragment) {
        bail!(
            "JPEG 2000 fragment does not begin with SOC marker 0xFF4F \
             (first 2 bytes: {:02X?})",
            &fragment[..fragment.len().min(2)]
        );
    }

    let params = DecodeParameters::new().reduce(0).layers(0).strict(true);
    let decoded = Image::from_bytes_with(fragment, params)
        .with_context(|| "JPEG 2000 codestream decode failed")?;

    image::extract_pixels(&decoded, &layout).with_context(|| "JPEG 2000 pixel extraction failed")
}

/// Returns `true` if `fragment` begins with the J2K SOC marker (`0xFF 0x4F`).
///
/// A bare DICOM JPEG 2000 codestream always starts with SOC (ISO 15444-1 §A.3).
/// JP2 file wrappers begin with the 12-byte JP2 Signature Box and do not appear
/// in DICOM transfer syntaxes 1.2.840.10008.1.2.4.90/91.
#[inline]
pub(crate) fn is_jpeg2000_codestream(fragment: &[u8]) -> bool {
    fragment.len() >= 2 && fragment[0] == (SOC >> 8) as u8 && fragment[1] == (SOC & 0xFF) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_support::encode_grayscale_j2k;

    fn layout(rows: usize, cols: usize, bits: u16, signed: bool) -> PixelLayout {
        PixelLayout {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated: bits,
            pixel_representation: u16::from(signed),
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        }
    }

    #[test]
    fn soc_marker_constant_matches_iso_15444_1() {
        assert_eq!(SOC, 0xFF4F, "SOC must equal 0xFF4F per ISO 15444-1 §A.3.1");
        assert_eq!(SOC >> 8, 0xFF, "SOC high byte must be 0xFF");
        assert_eq!(SOC & 0xFF, 0x4F, "SOC low byte must be 0x4F");
    }

    #[test]
    fn soi_constant_matches_jpeg_start_of_image() {
        assert_eq!(SOI, 0xFFD8, "SOI must equal 0xFFD8");
        assert_ne!(SOI, SOC, "SOI and SOC must be distinct markers");
    }

    #[test]
    fn is_jpeg2000_codestream_detects_soc_at_byte_0() {
        let codestream = [0xFF_u8, 0x4F, 0x00, 0x00];
        assert!(
            is_jpeg2000_codestream(&codestream),
            "0xFF 0x4F prefix must be recognized as J2K SOC"
        );
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_jpeg_ls_prefix() {
        let jpeg_ls = [0xFF_u8, 0xD8, 0xFF, 0xF7];
        assert!(
            !is_jpeg2000_codestream(&jpeg_ls),
            "JPEG-LS SOI prefix 0xFFD8 must not be recognized as J2K SOC"
        );
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_rle_prefix() {
        let rle = [0x00_u8, 0x00, 0x00, 0x01];
        assert!(
            !is_jpeg2000_codestream(&rle),
            "RLE prefix must not be recognized as J2K SOC"
        );
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_empty_and_single_byte() {
        assert!(
            !is_jpeg2000_codestream(&[]),
            "empty slice must return false"
        );
        assert!(
            !is_jpeg2000_codestream(&[0xFF]),
            "single byte must return false because SOC requires two bytes"
        );
    }

    #[test]
    fn decode_returns_error_for_non_soc_prefix() {
        let fragment = [0xFF_u8, 0xD8, 0xFF, 0xF7, 0x00, 0x0B];
        let err = decode_jpeg2000_fragment(&fragment, layout(2, 2, 8, false)).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("SOC") || msg.contains("0xFF4F") || msg.contains("FF4F"),
            "error must mention SOC marker; got: {}",
            msg
        );
    }

    #[test]
    fn decode_returns_error_for_truncated_codestream() {
        let truncated = [0xFF_u8, 0x4F, 0x00];
        let err = decode_jpeg2000_fragment(&truncated, layout(4, 4, 8, false)).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("decode") || msg.contains("JPEG 2000") || msg.contains("Unknown format"),
            "truncated J2K codestream error must be codec-contextual; got: {msg}"
        );
    }

    #[test]
    fn decode_jpeg2000_lossless_round_trip_4x4_uniform() {
        let rows = 4u32;
        let cols = 4u32;
        let pixel_value = 128i32;
        let pixels: Vec<i32> = vec![pixel_value; (rows * cols) as usize];
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, 8, false);

        assert!(
            is_jpeg2000_codestream(&j2k),
            "encoded output must start with SOC 0xFF4F; got: {:02X?}",
            &j2k[..j2k.len().min(4)]
        );

        let decoded = decode_jpeg2000_fragment(&j2k, layout(4, 4, 8, false))
            .expect("lossless JPEG 2000 round-trip must succeed");

        assert_eq!(decoded.len(), (rows * cols) as usize);
        for (i, &value) in decoded.iter().enumerate() {
            assert_eq!(
                value, pixel_value as f32,
                "pixel[{i}] must round-trip exactly"
            );
        }
    }

    #[test]
    fn decode_jpeg2000_lossless_round_trip_gradient_2x4() {
        let rows = 2u32;
        let cols = 4u32;
        let pixels: Vec<i32> = (0..8).collect();
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, 8, false);

        let decoded = decode_jpeg2000_fragment(&j2k, layout(2, 4, 8, false))
            .expect("gradient round-trip must succeed");

        assert_eq!(decoded.len(), pixels.len());
        for (i, (&raw, &decoded_val)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(decoded_val, raw as f32, "gradient pixel[{i}] must be exact");
        }
    }

    #[test]
    fn decode_jpeg2000_signed_samples_round_trip() {
        let pixels = [-4, -1, 0, 3];
        let j2k = encode_grayscale_j2k(&pixels, 2, 2, 8, true);

        let decoded = decode_jpeg2000_fragment(&j2k, layout(2, 2, 8, true))
            .expect("signed lossless JPEG 2000 round-trip must succeed");

        assert_eq!(decoded, vec![-4.0, -1.0, 0.0, 3.0]);
    }

    #[test]
    fn decode_jpeg2000_lossless_rescale_applied_correctly() {
        let pixels = [100i32];
        let j2k = encode_grayscale_j2k(&pixels, 1, 1, 8, false);
        let mut pixel_layout = layout(1, 1, 8, false);
        pixel_layout.rescale_slope = 2.0;
        pixel_layout.rescale_intercept = -1024.0;

        let decoded = decode_jpeg2000_fragment(&j2k, pixel_layout)
            .expect("single-pixel rescale test must succeed");

        assert_eq!(decoded, vec![-824.0]);
    }

    #[test]
    fn openjp2_backend_version_is_2_5_x() {
        let version_str = unsafe {
            let ptr = openjp2::openjpeg::opj_version();
            assert!(!ptr.is_null(), "opj_version returned null");
            std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
        };
        assert!(
            version_str.starts_with("2.5"),
            "openjp2 Rust backend must report OpenJPEG 2.5.x compatibility; got {version_str}"
        );
    }
}
