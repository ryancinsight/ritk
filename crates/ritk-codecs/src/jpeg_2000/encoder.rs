//! Pure-Rust J2K encoder.
//!
//! Produces minimal conformant bare J2K codestreams (no JP2 wrapper), as
//! required for DICOM-encapsulated JPEG 2000 Lossless (TS 1.2.840.10008.1.2.4.90).
//! Current configuration:
//! - One tile = entire image.
//! - Caller-selected 5/3 reversible DWT decomposition levels.
//! - Reversible 5/3 wavelet (`wavelet_transform = 1`).
//! - One quality layer.
//! - Guard bits = 2.
//!
//! The codestream layout is:
//! `SOC | SIZ | COD | QCD | [tile-part: SOT + SOD + packet] | EOC`
//!
//! # Evidence tier
//! Correctness is verified by the round-trip tests in `mod.rs` which encode
//! with this module and decode with the pure-Rust decoder; max error is exactly
//! 0 for all test inputs (lossless invariant).

use super::packet::encode_tile_part;
use super::subband::subband_layout;
use crate::PixelSignedness;

/// Guard bits used in the QCD marker and MSBs computation.
const GUARD_BITS: u8 = 2;

/// Encode a grayscale image as a bare J2K codestream.
///
/// # Parameters
/// - `pixels`: raw integer samples (for unsigned components these are the
///   original pixel values before DC shift; for signed they are the signed
///   stored values).
/// - `rows` / `cols`: image dimensions.
/// - `precision`: bit-depth (1–16; typically 8 or 16).
/// - `signed`: whether the component uses signed representation.
///
/// # DC level shift (ISO 15444-1 §G.1.2)
/// Unsigned components are DC-shifted by `−2^(precision−1)` before EBCOT
/// coding and the shift is reversed during decoding.
pub fn encode_grayscale_j2k(
    pixels: &[i32],
    rows: u32,
    cols: u32,
    precision: u32,
    signed: PixelSignedness,
    num_decomp_levels: u8,
) -> Vec<u8> {
    assert_eq!(
        pixels.len(),
        (rows * cols) as usize,
        "pixels length must equal rows × cols"
    );
    assert!((1..=38).contains(&precision), "precision must be in 1..=38");

    let w = cols as usize;
    let h = rows as usize;
    let is_signed = signed.is_signed();

    // Apply DC level shift for unsigned components.
    let dc_offset = if is_signed {
        0i32
    } else {
        -(1i32 << (precision - 1))
    };
    let shifted: Vec<i32> = pixels.iter().map(|&v| v + dc_offset).collect();

    // Build the tile-part (SOT + SOD + packet).
    let tile_part = encode_tile_part(&shifted, w, h, GUARD_BITS, precision, 0, num_decomp_levels);

    // Assemble the full codestream.
    let mut cs = Vec::new();

    // SOC
    cs.extend_from_slice(&[0xFF, 0x4F]);

    // SIZ: Rsiz=0, Xsiz=cols, Ysiz=rows, XOsiz=0, YOsiz=0,
    //       XTsiz=cols, YTsiz=rows, XTOsiz=0, YTOsiz=0, Csiz=1,
    //       Ssiz=(precision-1)|(sign<<7), XRsiz=1, YRsiz=1.
    let ssiz = ((precision - 1) as u8) | (if is_signed { 0x80 } else { 0x00 });
    let lsiz: u16 = 38 + 3; // 38 fixed + 3 per component × 1
    let mut siz_body: Vec<u8> = Vec::new();
    siz_body.extend_from_slice(&0u16.to_be_bytes()); // Rsiz
    siz_body.extend_from_slice(&cols.to_be_bytes()); // Xsiz
    siz_body.extend_from_slice(&rows.to_be_bytes()); // Ysiz
    siz_body.extend_from_slice(&0u32.to_be_bytes()); // XOsiz
    siz_body.extend_from_slice(&0u32.to_be_bytes()); // YOsiz
    siz_body.extend_from_slice(&cols.to_be_bytes()); // XTsiz
    siz_body.extend_from_slice(&rows.to_be_bytes()); // YTsiz
    siz_body.extend_from_slice(&0u32.to_be_bytes()); // XTOsiz
    siz_body.extend_from_slice(&0u32.to_be_bytes()); // YTOsiz
    siz_body.extend_from_slice(&1u16.to_be_bytes()); // Csiz=1
    siz_body.push(ssiz); // Ssiz
    siz_body.push(1); // XRsiz
    siz_body.push(1); // YRsiz

    cs.extend_from_slice(&[0xFF, 0x51]); // SIZ marker
    cs.extend_from_slice(&lsiz.to_be_bytes()); // Lsiz
    cs.extend_from_slice(&siz_body);

    // COD: Scod=0 (no custom precincts, no SOP/EPH),
    //       progression=LRCP(0), layers=1, MCT=0,
    //       num_decomp=0, xcb_o=4 (64px), ycb_o=4 (64px),
    //       cb_style=0, wavelet=1 (5/3 reversible).
    let lcod: u16 = 12; // Lcod = 12 bytes
    cs.extend_from_slice(&[0xFF, 0x52]); // COD marker
    cs.extend_from_slice(&lcod.to_be_bytes()); // Lcod
    cs.push(0x00); // Scod: no custom precincts
    cs.push(0x00); // SGcod: LRCP
    cs.extend_from_slice(&1u16.to_be_bytes()); // SGcod: 1 layer
    cs.push(0x00); // SGcod: no MCT
    cs.push(num_decomp_levels); // SPcod: num_decomp_levels
    cs.push(0x04); // SPcod: xcb_o = 4 → cb_width = 2^(4+2) = 64
    cs.push(0x04); // SPcod: ycb_o = 4 → cb_height = 64
    cs.push(0x00); // SPcod: cb_style = 0
    cs.push(0x01); // SPcod: wavelet_transform = 1 (5/3)

    // QCD: no quantization (lossless), guard_bits=2.
    let num_bands = 3 * u16::from(num_decomp_levels) + 1;
    let lqcd: u16 = 3 + num_bands; // Lqcd: 2 (length) + 1 (Sqcd) + 1 byte per subband
    let sqcd: u8 = GUARD_BITS << 5; // guard_bits in bits 7-5, style 0 (no quant)
    cs.extend_from_slice(&[0xFF, 0x5C]); // QCD marker
    cs.extend_from_slice(&lqcd.to_be_bytes()); // Lqcd
    cs.push(sqcd); // Sqcd
                   // SPqcd per subband in codestream order: ε_b = precision + gain_b
                   // (reversible 5/3 gains: LL 0, HL/LH 1, HH 2), ε in bits 7–3.
    for band in subband_layout(w, h, num_decomp_levels) {
        cs.push((((precision + band.gain) << 3) & 0xFF) as u8); // SPqcd[b]
    }

    // Tile-part (SOT + SOD + packet).
    cs.extend_from_slice(&tile_part);

    // EOC.
    cs.extend_from_slice(&[0xFF, 0xD9]);

    cs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg_2000::image::{decode_j2k_fragment, is_soc};
    use crate::PixelLayout;
    use crate::PixelSignedness;

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

    #[test]
    fn encoder_output_starts_with_soc() {
        let j2k = encode_grayscale_j2k(&[0i32; 4], 2, 2, 8, PixelSignedness::Unsigned, 0);
        assert!(
            is_soc(&j2k),
            "encoded codestream must start with SOC 0xFF4F"
        );
    }

    #[test]
    fn encoder_ends_with_eoc() {
        let j2k = encode_grayscale_j2k(&[1i32; 4], 2, 2, 8, PixelSignedness::Unsigned, 0);
        let last2 = &j2k[j2k.len() - 2..];
        assert_eq!(
            last2,
            [0xFF, 0xD9],
            "encoded codestream must end with EOC 0xFFD9"
        );
    }

    #[test]
    fn round_trip_uniform_unsigned_8bit() {
        let pixel_value = 128i32;
        let pixels = vec![pixel_value; 16];
        let j2k = encode_grayscale_j2k(&pixels, 4, 4, 8, PixelSignedness::Unsigned, 0);
        let decoded = decode_j2k_fragment(&j2k, layout(4, 4, 8, PixelSignedness::Unsigned))
            .expect("round-trip decode must succeed");
        assert_eq!(decoded.len(), 16);
        for (i, &v) in decoded.iter().enumerate() {
            assert_eq!(v, pixel_value as f32, "pixel[{i}] must be exact");
        }
    }

    #[test]
    fn round_trip_gradient_unsigned_8bit() {
        let pixels: Vec<i32> = (0..8).collect();
        let j2k = encode_grayscale_j2k(&pixels, 2, 4, 8, PixelSignedness::Unsigned, 0);
        let decoded = decode_j2k_fragment(&j2k, layout(2, 4, 8, PixelSignedness::Unsigned))
            .expect("gradient round-trip must succeed");
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(dec, orig as f32, "pixel[{i}] must be exact");
        }
    }

    #[test]
    fn round_trip_signed_8bit() {
        let pixels = vec![-4i32, -1, 0, 3];
        let j2k = encode_grayscale_j2k(&pixels, 2, 2, 8, PixelSignedness::Signed, 0);
        let decoded = decode_j2k_fragment(&j2k, layout(2, 2, 8, PixelSignedness::Signed))
            .expect("signed round-trip must succeed");
        assert_eq!(decoded, vec![-4.0f32, -1.0, 0.0, 3.0]);
    }

    #[test]
    fn round_trip_single_pixel_with_rescale() {
        let pixels = vec![100i32];
        let j2k = encode_grayscale_j2k(&pixels, 1, 1, 8, PixelSignedness::Unsigned, 0);
        let mut lyt = layout(1, 1, 8, PixelSignedness::Unsigned);
        lyt.rescale_slope = 2.0;
        lyt.rescale_intercept = -1024.0;
        let decoded = decode_j2k_fragment(&j2k, lyt).expect("single-pixel rescale must succeed");
        assert_eq!(decoded, vec![-824.0f32]); // 100 * 2 + (-1024) = -824
    }
}
