//! Native JPEG 2000 (ISO 15444-1) decoder for DICOM encapsulated frames.
//!
//! # Architecture
//! This module provides a **pure-Rust** implementation of JPEG 2000 decoding,
//! eliminating all C/FFI dependencies (`jpeg2k`, `openjp2`, `openjpeg-sys`).
//!
//! Sub-modules:
//! - `marker`     – ISO 15444-1 marker constants and byte-read utilities.
//! - `codestream` – Main-header parser (SIZ, COD, QCD, SOT).
//! - `mq_coder`   – MQ arithmetic coder — encoder and decoder (Annex C).
//! - `ebcot`      – EBCOT tier-1 encoder and decoder (Annex D).
//! - `packet`     – Tier-2 packet encoder and decoder (Annex B).
//! - `wavelet`    – Forward and inverse 5/3 reversible DWT (Annex F).
//! - `subband`    – Mallat subband geometry (Annex B.5).
//! - `tag_tree`   – Quad-tree inclusion/MSB coding (Annex B.10.2).
//! - `image`      – Full codestream decoder and DICOM pixel extractor.
//! - [`encoder`]  – Pure-Rust encoder (produces conformant codestreams).
//!
//! # Specification (ISO 15444-1 / DICOM PS3.5)
//! DICOM JPEG 2000 encapsulates a raw J2K codestream (not a JP2 file wrapper):
//! - Transfer Syntax 1.2.840.10008.1.2.4.90: JPEG 2000 Lossless Only.
//! - Transfer Syntax 1.2.840.10008.1.2.4.91: JPEG 2000 lossy or lossless.
//!
//! # Current limitations
//! - One precinct per resolution/band (no precinct partitioning; code-blocks
//!   are 64×64 within each subband).
//! - Lossy (9/7 irreversible wavelet) decoding is not yet supported — J2K-LOSSY-97.
//! - Interop against externally encoded streams is pending differential
//!   validation — J2K-INTEROP.

pub(crate) mod codestream;
pub(crate) mod ebcot;
pub mod encoder;
pub(crate) mod image;
pub(crate) mod marker;
pub(crate) mod mq_coder;
pub(crate) mod packet;
pub(crate) mod subband;
pub(crate) mod tag_tree;
pub(crate) mod wavelet;

use anyhow::{bail, Result};

use crate::PixelLayout;
use image::{decode_j2k_fragment, is_soc};

/// J2K Start of Codestream marker (ISO 15444-1 §A.3): bytes `0xFF 0x4F`.
#[allow(dead_code)] // Tested via soc_marker_constant_matches_iso_15444_1
pub(crate) const SOC: u16 = 0xFF4F;

/// JPEG / JFIF Start of Image marker (0xFFD8), distinct from SOC.
#[allow(dead_code)] // Tested via soi_constant_matches_jpeg_start_of_image
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
/// - the JPEG 2000 decoder fails to parse or decode the codestream.
/// - decoded component metadata does not match `layout`.
pub fn decode_jpeg2000_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    if !is_jpeg2000_codestream(fragment) {
        bail!(
            "JPEG 2000 fragment does not begin with SOC marker 0xFF4F \
             (first 2 bytes: {:02X?})",
            &fragment[..fragment.len().min(2)]
        );
    }
    decode_j2k_fragment(fragment, layout)
}

/// Returns `true` if `fragment` begins with the J2K SOC marker (`0xFF 0x4F`).
///
/// A bare DICOM JPEG 2000 codestream always starts with SOC (ISO 15444-1 §A.3).
#[inline]
pub(crate) fn is_jpeg2000_codestream(fragment: &[u8]) -> bool {
    is_soc(fragment)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelSignedness;
    use encoder::encode_grayscale_j2k;

    fn layout(rows: usize, cols: usize, bits: u16, signed: PixelSignedness) -> PixelLayout {
        PixelLayout {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated: bits,
            pixel_representation: signed,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        }
    }

    // ── Marker constant tests ────────────────────────────────────────────────

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

    // ── Codestream detection ─────────────────────────────────────────────────

    #[test]
    fn is_jpeg2000_codestream_detects_soc_at_byte_0() {
        assert!(is_jpeg2000_codestream(&[0xFF_u8, 0x4F, 0x00]));
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_jpeg_ls_prefix() {
        assert!(!is_jpeg2000_codestream(&[0xFF_u8, 0xD8, 0xFF, 0xF7]));
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_rle_prefix() {
        assert!(!is_jpeg2000_codestream(&[0x00_u8, 0x00, 0x00, 0x01]));
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_empty_and_single_byte() {
        assert!(!is_jpeg2000_codestream(&[]));
        assert!(!is_jpeg2000_codestream(&[0xFF]));
    }

    // ── Error-path tests ─────────────────────────────────────────────────────

    #[test]
    fn decode_returns_error_for_non_soc_prefix() {
        let fragment = [0xFF_u8, 0xD8, 0xFF, 0xF7, 0x00, 0x0B];
        let err = decode_jpeg2000_fragment(&fragment, layout(2, 2, 8, PixelSignedness::Unsigned))
            .unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("SOC") || msg.contains("0xFF4F") || msg.contains("FF4F"),
            "error must mention SOC marker; got: {msg}"
        );
    }

    #[test]
    fn decode_returns_error_for_truncated_codestream() {
        let truncated = [0xFF_u8, 0x4F, 0x00];
        let err = decode_jpeg2000_fragment(&truncated, layout(4, 4, 8, PixelSignedness::Unsigned))
            .unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("parse")
                || msg.contains("JPEG 2000")
                || msg.contains("J2K")
                || msg.contains("SIZ")
                || msg.contains("SOC"),
            "truncated J2K codestream error must be descriptive; got: {msg}"
        );
    }

    // ── Lossless round-trip tests ────────────────────────────────────────────

    #[test]
    fn decode_jpeg2000_lossless_round_trip_4x4_uniform() {
        let rows = 4u32;
        let cols = 4u32;
        let pixel_value = 128i32;
        let pixels = vec![pixel_value; (rows * cols) as usize];
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, 8, PixelSignedness::Unsigned, 0);

        assert!(
            is_jpeg2000_codestream(&j2k),
            "encoded output must start with SOC 0xFF4F; first bytes: {:02X?}",
            &j2k[..j2k.len().min(4)]
        );

        let decoded = decode_jpeg2000_fragment(&j2k, layout(4, 4, 8, PixelSignedness::Unsigned))
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
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, 8, PixelSignedness::Unsigned, 0);

        let decoded = decode_jpeg2000_fragment(&j2k, layout(2, 4, 8, PixelSignedness::Unsigned))
            .expect("gradient round-trip must succeed");

        assert_eq!(decoded.len(), pixels.len());
        for (i, (&raw, &decoded_val)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(decoded_val, raw as f32, "gradient pixel[{i}] must be exact");
        }
    }

    #[test]
    fn decode_jpeg2000_signed_samples_round_trip() {
        let pixels = [-4i32, -1, 0, 3];
        let j2k = encode_grayscale_j2k(&pixels, 2, 2, 8, PixelSignedness::Signed, 0);

        let decoded = decode_jpeg2000_fragment(&j2k, layout(2, 2, 8, PixelSignedness::Signed))
            .expect("signed lossless JPEG 2000 round-trip must succeed");

        assert_eq!(decoded, vec![-4.0f32, -1.0, 0.0, 3.0]);
    }

    #[test]
    fn decode_jpeg2000_lossless_rescale_applied_correctly() {
        let pixels = [100i32];
        let j2k = encode_grayscale_j2k(&pixels, 1, 1, 8, PixelSignedness::Unsigned, 0);
        let mut pixel_layout = layout(1, 1, 8, PixelSignedness::Unsigned);
        pixel_layout.rescale_slope = 2.0;
        pixel_layout.rescale_intercept = -1024.0;

        let decoded = decode_jpeg2000_fragment(&j2k, pixel_layout)
            .expect("single-pixel rescale test must succeed");

        assert_eq!(decoded, vec![-824.0f32]); // 100 × 2 + (−1024) = −824
    }

    #[test]
    fn decode_jpeg2000_lossless_round_trip_unsigned_16bit() {
        // Regression: 16-bit precision uses ≥ 39 coding passes, exercising the
        // long branch of the pass-count prefix code (ISO 15444-1 Table B.4).
        let pixels: Vec<i32> = vec![
            0, 256, 512, 1024, 2048, 3071, 3584, 3840, 100, 200, 400, 800, 1600, 2400, 3000, 4095,
        ];
        let j2k = encode_grayscale_j2k(&pixels, 4, 4, 16, PixelSignedness::Unsigned, 0);
        let decoded = decode_jpeg2000_fragment(&j2k, layout(4, 4, 16, PixelSignedness::Unsigned))
            .expect("16-bit lossless round-trip must succeed");
        let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
        assert_eq!(decoded, expected, "16-bit samples must round-trip exactly");
    }

    proptest::proptest! {
        /// Lossless invariant (ISO 15444-1, 5/3 reversible, 0 DWT levels):
        /// for any image and precision, |decoded − original| = 0 exactly.
        #[test]
        fn decode_jpeg2000_lossless_round_trip_random(
            rows in 1u32..9,
            cols in 1u32..9,
            precision in proptest::sample::select(vec![8u32, 12, 16]),
            signed in proptest::bool::ANY,
            num_decomp_levels in 0u8..4,
            seed in proptest::num::u64::ANY,
        ) {
            let n = (rows * cols) as usize;
            // Deterministic LCG over the sample domain of the chosen precision.
            let mut state = seed | 1;
            let pixels: Vec<i32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let raw = (state >> 33) as i64;
                    if signed {
                        let half = 1i64 << (precision - 1);
                        ((raw % (2 * half)) - half) as i32
                    } else {
                        (raw % (1i64 << precision)) as i32
                    }
                })
                .collect();
            let signedness = if signed { PixelSignedness::Signed } else { PixelSignedness::Unsigned };
            let j2k = encode_grayscale_j2k(&pixels, rows, cols, precision, signedness, num_decomp_levels);
            let decoded = decode_jpeg2000_fragment(
                &j2k,
                layout(rows as usize, cols as usize, precision as u16, signedness),
            )
            .expect("random lossless round-trip must succeed");
            let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
            proptest::prop_assert_eq!(decoded, expected);
        }
    }

    /// Deterministic CT-like content: gradient + LCG noise.
    fn synthetic(rows: u32, cols: u32, amplitude: i32) -> Vec<i32> {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        (0..rows as usize * cols as usize)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 33) % 32) as i32;
                ((i as i32 * 7) + noise) % amplitude
            })
            .collect()
    }

    #[test]
    fn decode_jpeg2000_multi_codeblock_zero_levels() {
        // 130×70 LL0 → 3×2 code-block grid: exercises multi-code-block packet
        // headers (shared tag trees) without DWT.
        let (rows, cols) = (70u32, 130u32);
        let pixels = synthetic(rows, cols, 256);
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, 8, PixelSignedness::Unsigned, 0);
        let decoded = decode_jpeg2000_fragment(&j2k, layout(70, 130, 8, PixelSignedness::Unsigned))
            .expect("multi-code-block LL0 round-trip must succeed");
        let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
        assert_eq!(decoded, expected, "multi-code-block LL0 must be lossless");
    }

    #[test]
    fn decode_jpeg2000_multi_codeblock_two_levels_16bit() {
        // 150×100, 2 levels: level-1 bands are 75×50 → 2×1 code-block grids;
        // exercises tag-tree coding over non-trivial grids at every resolution.
        let (rows, cols) = (100u32, 150u32);
        let pixels = synthetic(rows, cols, 4096);
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, 16, PixelSignedness::Unsigned, 2);
        let decoded =
            decode_jpeg2000_fragment(&j2k, layout(100, 150, 16, PixelSignedness::Unsigned))
                .expect("multi-code-block 2-level round-trip must succeed");
        let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
        assert_eq!(decoded, expected, "multi-code-block DWT must be lossless");
    }

    #[test]
    fn decode_jpeg2000_lossless_round_trip_two_dwt_levels_16bit() {
        // 5/3 reversible ⟹ exact reconstruction at any decomposition depth.
        let rows = 8u32;
        let cols = 12u32;
        let pixels: Vec<i32> = (0..96).map(|i| (i * 631) % 4096).collect();
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, 16, PixelSignedness::Unsigned, 2);
        let decoded = decode_jpeg2000_fragment(&j2k, layout(8, 12, 16, PixelSignedness::Unsigned))
            .expect("2-level DWT lossless round-trip must succeed");
        let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
        assert_eq!(decoded, expected, "2-level DWT samples must be exact");
    }

    #[test]
    fn decode_jpeg2000_lossless_round_trip_three_dwt_levels_signed_odd_dims() {
        let rows = 7u32;
        let cols = 9u32;
        let pixels: Vec<i32> = (0..63).map(|i| ((i * 37) % 256) - 128).collect();
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, 8, PixelSignedness::Signed, 3);
        let decoded = decode_jpeg2000_fragment(&j2k, layout(7, 9, 8, PixelSignedness::Signed))
            .expect("3-level DWT signed odd-dims round-trip must succeed");
        let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
        assert_eq!(decoded, expected, "3-level DWT samples must be exact");
    }

    #[test]
    fn ritk_native_decoder_replaces_openjp2_backend() {
        // This test was previously `openjp2_backend_version_is_2_5_x`.
        // The RITK-native pure-Rust decoder has no version dependency on
        // OpenJPEG; we verify the codec pipeline is self-consistent by
        // confirming a round-trip produces zero error.
        let pixels: Vec<i32> = (0..16i32).map(|v| v * 10).collect();
        let j2k = encode_grayscale_j2k(&pixels, 4, 4, 8, PixelSignedness::Unsigned, 0);
        let decoded = decode_jpeg2000_fragment(&j2k, layout(4, 4, 8, PixelSignedness::Unsigned))
            .expect("native codec round-trip must succeed");
        let max_err = pixels
            .iter()
            .zip(decoded.iter())
            .map(|(&p, &d)| (p as f32 - d).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            max_err, 0.0,
            "RITK-native J2K round-trip max error must be 0; got {max_err}"
        );
    }
}
