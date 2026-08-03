//! Pure-Rust J2K encoder.
//!
//! Produces minimal conformant bare J2K codestreams (no JP2 wrapper), as
//! required for DICOM-encapsulated JPEG 2000 (TS 1.2.840.10008.1.2.4.90 lossless,
//! .91 lossy).
//! Current configuration:
//! - One tile = entire image.
//! - Caller-selected [`Jpeg2000Encoding`]: 5/3 reversible lossless or 9/7
//!   irreversible with a validated scalar quantization step.
//! - One quality layer.
//! - Guard bits = 2.
//!
//! The codestream layout is:
//! `SOC | SIZ | COD | QCD | [tile-part: SOT + SOD + packet] | EOC`
//!
//! # Evidence tier
//! Correctness is verified by round-trip tests that encode with this module and
//! decode with the pure-Rust decoder. The reversible path reconstructs every
//! sample exactly. Irreversible tests verify the transmitted quantization step,
//! bounded reconstruction error, and the expected size/error ordering between
//! finer and coarser steps.

use super::packet::encode_tile_part;
use super::packet::WaveletTransform;
use super::quantization::ScalarQuantizer;
pub use super::quantization::{QuantizationStep, QuantizationStepError};
use super::subband::subband_layout;
use crate::PixelSignedness;

mod validation;

pub use validation::Jpeg2000EncodeError;
use validation::{validate_geometry, validate_precision};

/// Guard bits used in the QCD marker and MSBs computation.
const GUARD_BITS: u8 = 2;

/// Transform, decomposition depth, and quantization contract for one encoded
/// component.
///
/// The variants make a lossless transform with lossy settings
/// unrepresentable. Use [`QuantizationStep::UNIT`] to retain the previous
/// irreversible encoder behavior.
///
/// # Examples
///
/// ```
/// use ritk_codecs::jpeg_2000::encoder::{Jpeg2000Encoding, QuantizationStep};
///
/// let lossless = Jpeg2000Encoding::Lossless {
///     decomposition_levels: 2,
/// };
/// let lossy = Jpeg2000Encoding::Lossy {
///     decomposition_levels: 2,
///     quantization_step: QuantizationStep::new(4.0)?,
/// };
/// assert_ne!(lossless, lossy);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Jpeg2000Encoding {
    /// Reversible 5/3 transform with no quantization.
    Lossless {
        /// Number of wavelet decomposition levels.
        decomposition_levels: u8,
    },
    /// Irreversible 9/7 transform with scalar dead-zone quantization.
    Lossy {
        /// Number of wavelet decomposition levels.
        decomposition_levels: u8,
        /// Requested positive finite quantization step.
        quantization_step: QuantizationStep,
    },
}

impl Jpeg2000Encoding {
    const fn decomposition_levels(self) -> u8 {
        match self {
            Self::Lossless {
                decomposition_levels,
            }
            | Self::Lossy {
                decomposition_levels,
                ..
            } => decomposition_levels,
        }
    }

    const fn transform(self) -> WaveletTransform {
        match self {
            Self::Lossless { .. } => WaveletTransform::Reversible,
            Self::Lossy { .. } => WaveletTransform::Irreversible,
        }
    }
}

/// Encode a grayscale image as a bare J2K codestream.
///
/// # Parameters
/// - `pixels`: raw integer samples (for unsigned components these are the
///   original pixel values before DC shift; for signed they are the signed
///   stored values).
/// - `rows` / `cols`: image dimensions.
/// - `precision`: bit-depth (1–16).
/// - `signed`: whether the component uses signed representation.
/// - `encoding`: reversible/lossless or irreversible/lossy mode, including
///   decomposition depth and the lossy quantization step.
///
/// # DC level shift (ISO 15444-1 §G.1.2)
/// Unsigned components are DC-shifted by `−2^(precision−1)` before EBCOT
/// coding and the shift is reversed during decoding.
///
/// # Errors
/// Returns an error for zero or overflowing geometry, a mismatched sample
/// count, precision outside 1–16, a decomposition depth larger than the image
/// geometry supports, a sample outside the range declared by `precision` and
/// `signed`, or a lossy quantization step that the QCD exponent cannot
/// represent for every transformed subband.
pub fn encode_grayscale_j2k(
    pixels: &[i32],
    rows: u32,
    cols: u32,
    precision: u32,
    signed: PixelSignedness,
    encoding: Jpeg2000Encoding,
) -> Result<Vec<u8>, Jpeg2000EncodeError> {
    let num_decomp_levels = encoding.decomposition_levels();
    let transform = encoding.transform();
    let (w, h) = validate_geometry(pixels.len(), rows, cols, num_decomp_levels)?;
    let is_signed = signed.is_signed();
    let (minimum, maximum) = validate_precision(precision, is_signed)?;
    if let Some((index, &value)) = pixels
        .iter()
        .enumerate()
        .find(|(_, value)| !(minimum..=maximum).contains(value))
    {
        return Err(Jpeg2000EncodeError::SampleOutOfRange {
            index,
            value,
            minimum,
            maximum,
        });
    }

    // Apply DC level shift for unsigned components.
    let dc_offset = if is_signed {
        0i32
    } else {
        -(1i32 << (precision - 1))
    };

    let bands = subband_layout(w, h, num_decomp_levels);
    let quantizers = match encoding {
        Jpeg2000Encoding::Lossless { .. } => None,
        Jpeg2000Encoding::Lossy {
            quantization_step, ..
        } => Some(
            bands
                .iter()
                .map(|band| {
                    let dynamic_range = precision + band.gain;
                    ScalarQuantizer::from_step(quantization_step, dynamic_range).ok_or(
                        Jpeg2000EncodeError::UnrepresentableQuantizationStep {
                            step: quantization_step,
                            dynamic_range,
                        },
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };

    // Build the tile-part (SOT + SOD + packet). The packet writer quantizes
    // with the exact deltas reconstructed from the QCD values below.
    let tile_part = encode_tile_part(
        pixels,
        w,
        h,
        dc_offset,
        GUARD_BITS,
        precision,
        0,
        num_decomp_levels,
        transform,
        quantizers.as_deref(),
    );

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
    //       caller-selected decomposition depth, xcb_o=4 (64px),
    //       ycb_o=4 (64px), cb_style=0, and caller-selected wavelet.
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
    cs.push(match transform {
        WaveletTransform::Reversible => 0x01,   // 5/3 reversible
        WaveletTransform::Irreversible => 0x00, // 9/7 irreversible
    }); // SPcod: wavelet_transform

    // QCD: reversible → no quantization (style 0, 1-byte ε entries);
    // irreversible → scalar expounded (style 2, 2-byte ε/μ entries) using the
    // exact per-subband representations already supplied to the packet writer.
    let num_bands = 3 * u16::from(num_decomp_levels) + 1;
    cs.extend_from_slice(&[0xFF, 0x5C]); // QCD marker
    match transform {
        WaveletTransform::Reversible => {
            let lqcd: u16 = 3 + num_bands; // 2 (length) + 1 (Sqcd) + 1 byte/subband
            let sqcd: u8 = GUARD_BITS << 5; // guard bits in 7-5, style 0
            cs.extend_from_slice(&lqcd.to_be_bytes());
            cs.push(sqcd);
            for band in &bands {
                cs.push((((precision + band.gain) << 3) & 0xFF) as u8); // ε in 7-3
            }
        }
        WaveletTransform::Irreversible => {
            let lqcd: u16 = 3 + 2 * num_bands; // 2 bytes per subband
            let sqcd: u8 = (GUARD_BITS << 5) | 0x02; // guard bits in 7-5, style 2
            cs.extend_from_slice(&lqcd.to_be_bytes());
            cs.push(sqcd);
            for quantizer in quantizers
                .as_ref()
                .expect("invariant: irreversible encoding has one quantizer per subband")
            {
                cs.extend_from_slice(&quantizer.packed().to_be_bytes());
            }
        }
    }

    // Tile-part (SOT + SOD + packet).
    cs.extend_from_slice(&tile_part);

    // EOC.
    cs.extend_from_slice(&[0xFF, 0xD9]);

    Ok(cs)
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

    const fn lossless(decomposition_levels: u8) -> Jpeg2000Encoding {
        Jpeg2000Encoding::Lossless {
            decomposition_levels,
        }
    }

    #[test]
    fn encoder_output_starts_with_soc() {
        let j2k = encode_grayscale_j2k(&[0i32; 4], 2, 2, 8, PixelSignedness::Unsigned, lossless(0))
            .expect("valid image must encode");
        assert!(
            is_soc(&j2k),
            "encoded codestream must start with SOC 0xFF4F"
        );
    }

    #[test]
    fn encoder_ends_with_eoc() {
        let j2k = encode_grayscale_j2k(&[1i32; 4], 2, 2, 8, PixelSignedness::Unsigned, lossless(0))
            .expect("valid image must encode");
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
        let j2k = encode_grayscale_j2k(&pixels, 4, 4, 8, PixelSignedness::Unsigned, lossless(0))
            .expect("valid image must encode");
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
        let j2k = encode_grayscale_j2k(&pixels, 2, 4, 8, PixelSignedness::Unsigned, lossless(0))
            .expect("valid image must encode");
        let decoded = decode_j2k_fragment(&j2k, layout(2, 4, 8, PixelSignedness::Unsigned))
            .expect("gradient round-trip must succeed");
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(dec, orig as f32, "pixel[{i}] must be exact");
        }
    }

    #[test]
    fn round_trip_signed_8bit() {
        let pixels = vec![-4i32, -1, 0, 3];
        let j2k = encode_grayscale_j2k(&pixels, 2, 2, 8, PixelSignedness::Signed, lossless(0))
            .expect("valid image must encode");
        let decoded = decode_j2k_fragment(&j2k, layout(2, 2, 8, PixelSignedness::Signed))
            .expect("signed round-trip must succeed");
        assert_eq!(decoded, vec![-4.0f32, -1.0, 0.0, 3.0]);
    }

    #[test]
    fn round_trip_single_pixel_with_rescale() {
        let pixels = vec![100i32];
        let j2k = encode_grayscale_j2k(&pixels, 1, 1, 8, PixelSignedness::Unsigned, lossless(0))
            .expect("valid image must encode");
        let mut lyt = layout(1, 1, 8, PixelSignedness::Unsigned);
        lyt.rescale_slope = 2.0;
        lyt.rescale_intercept = -1024.0;
        let decoded = decode_j2k_fragment(&j2k, lyt).expect("single-pixel rescale must succeed");
        assert_eq!(decoded, vec![-824.0f32]); // 100 * 2 + (-1024) = -824
    }

    #[test]
    fn encoder_rejects_zero_geometry() {
        let error = encode_grayscale_j2k(&[], 0, 1, 8, PixelSignedness::Unsigned, lossless(0))
            .expect_err("zero-height image must fail");
        assert_eq!(error, Jpeg2000EncodeError::EmptyImage { rows: 0, cols: 1 });
    }

    #[test]
    fn encoder_rejects_mismatched_sample_count() {
        let error = encode_grayscale_j2k(&[0], 2, 2, 8, PixelSignedness::Unsigned, lossless(0))
            .expect_err("mismatched sample count must fail");
        assert_eq!(
            error,
            Jpeg2000EncodeError::PixelCountMismatch {
                actual: 1,
                expected: 4,
            }
        );
    }

    #[test]
    fn encoder_rejects_unsupported_precision() {
        for precision in [0, 17] {
            let error =
                encode_grayscale_j2k(&[0], 1, 1, precision, PixelSignedness::Signed, lossless(0))
                    .expect_err("unsupported precision must fail");
            assert_eq!(
                error,
                Jpeg2000EncodeError::UnsupportedPrecision {
                    precision,
                    maximum: 16,
                }
            );
        }
    }

    #[test]
    fn encoder_rejects_excessive_decomposition() {
        let error = encode_grayscale_j2k(&[0; 4], 2, 2, 8, PixelSignedness::Unsigned, lossless(2))
            .expect_err("decomposition beyond geometry must fail");
        assert_eq!(
            error,
            Jpeg2000EncodeError::ExcessiveDecomposition {
                requested: 2,
                maximum: 1,
            }
        );
    }

    #[test]
    fn encoder_rejects_samples_outside_declared_range() {
        for (signedness, sample, minimum, maximum) in [
            (PixelSignedness::Unsigned, -1, 0, 255),
            (PixelSignedness::Unsigned, 256, 0, 255),
            (PixelSignedness::Signed, -129, -128, 127),
            (PixelSignedness::Signed, 128, -128, 127),
        ] {
            let error = encode_grayscale_j2k(&[sample], 1, 1, 8, signedness, lossless(0))
                .expect_err("out-of-range sample must fail");
            assert_eq!(
                error,
                Jpeg2000EncodeError::SampleOutOfRange {
                    index: 0,
                    value: sample,
                    minimum,
                    maximum,
                }
            );
        }
    }

    #[test]
    fn encoder_accepts_declared_sample_boundaries() {
        for (signedness, pixels) in [
            (PixelSignedness::Unsigned, [0, 255]),
            (PixelSignedness::Signed, [-128, 127]),
        ] {
            let codestream = encode_grayscale_j2k(&pixels, 1, 2, 8, signedness, lossless(0))
                .expect("inclusive sample boundaries must encode");
            let decoded = decode_j2k_fragment(&codestream, layout(1, 2, 8, signedness))
                .expect("boundary codestream must decode");
            assert_eq!(
                decoded,
                pixels.map(|sample| sample as f32),
                "declared sample boundaries must round-trip exactly"
            );
        }
    }

    #[test]
    fn encoder_rejects_unrepresentable_lossy_step_before_transform() {
        let step = QuantizationStep::new(4.0).expect("positive finite step must be valid");
        let error = encode_grayscale_j2k(
            &[0],
            1,
            1,
            1,
            PixelSignedness::Unsigned,
            Jpeg2000Encoding::Lossy {
                decomposition_levels: 0,
                quantization_step: step,
            },
        )
        .expect_err("QCD exponent underflow must fail");
        assert_eq!(
            error,
            Jpeg2000EncodeError::UnrepresentableQuantizationStep {
                step,
                dynamic_range: 1,
            }
        );
    }

    #[test]
    fn coarser_lossy_step_reduces_documented_fixture_size() {
        let pixels: Vec<i32> = (0i32..96 * 96)
            .map(|index| {
                let x = index % 96;
                let y = index / 96;
                (x * 31 + y * 17 + (x / 8) * 211) % 4096
            })
            .collect();
        let encode = |step| {
            encode_grayscale_j2k(
                &pixels,
                96,
                96,
                12,
                PixelSignedness::Unsigned,
                Jpeg2000Encoding::Lossy {
                    decomposition_levels: 3,
                    quantization_step: QuantizationStep::new(step)
                        .expect("fixture step must be positive and finite"),
                },
            )
            .expect("valid lossy fixture must encode")
        };
        let unit = encode(1.0);
        let coarse = encode(8.0);
        assert!(
            coarse.len() < unit.len(),
            "coarse={} bytes, unit={} bytes",
            coarse.len(),
            unit.len()
        );

        let mean_squared_error = |codestream: &[u8]| {
            let decoded =
                decode_j2k_fragment(codestream, layout(96, 96, 12, PixelSignedness::Unsigned))
                    .expect("lossy codestream must decode");
            let sample_count =
                u32::try_from(pixels.len()).expect("invariant: fixture sample count fits u32");
            pixels
                .iter()
                .zip(&decoded)
                .map(|(&source, &reconstructed)| {
                    let error = f64::from(source) - f64::from(reconstructed);
                    error * error
                })
                .sum::<f64>()
                / f64::from(sample_count)
        };
        let unit_error = mean_squared_error(&unit);
        let coarse_error = mean_squared_error(&coarse);
        assert!(
            coarse_error > unit_error,
            "coarse MSE={coarse_error}, unit MSE={unit_error}"
        );
    }

    #[test]
    fn lossy_codestream_qcd_carries_requested_representable_step() {
        let step = QuantizationStep::new(3.25).expect("positive finite step must be valid");
        let pixels: Vec<i32> = (0..64).map(|value| value * 53 % 4096).collect();
        let codestream = encode_grayscale_j2k(
            &pixels,
            8,
            8,
            12,
            PixelSignedness::Unsigned,
            Jpeg2000Encoding::Lossy {
                decomposition_levels: 2,
                quantization_step: step,
            },
        )
        .expect("representable step must encode");
        let (header, _) = crate::jpeg_2000::codestream::parse_main_header(&codestream)
            .expect("encoder output must have a valid main header");
        let exponents = header.qcd.exponents();
        let mantissas = header.qcd.mantissas();
        let bands = subband_layout(8, 8, 2);
        assert_eq!(exponents.len(), bands.len());
        assert_eq!(mantissas.len(), bands.len());
        for ((band, exponent), mantissa) in bands.iter().zip(exponents).zip(mantissas) {
            assert_eq!(
                crate::jpeg_2000::quantization::step_size(12 + band.gain, exponent, mantissa,),
                step.get()
            );
        }
    }
}
