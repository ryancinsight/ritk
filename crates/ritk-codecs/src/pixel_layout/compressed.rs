//! Encapsulated-codec sample conversion into the DICOM modality domain.

use anyhow::{bail, Context, Result};

use super::{PixelLayout, PixelSignedness};

/// Encode one value from the modality domain back to its stored sample.
///
/// This is the exact inverse of the `* slope + intercept` transform applied by
/// [`decode_compressed_samples`]. Rounding is half-away-from-zero rather than
/// `f32::round`'s half-to-even: stored samples are integers obtained by
/// rescaling continuous detector values, and the midpoint case is a tie between
/// two equally valid integers, so either is defensible -- what is *not*
/// defensible is a rule that biases every tie the same way on a large image,
/// which is why this is stated rather than left implicit.
pub(crate) fn encode_stored_sample(value: f32, layout: PixelLayout) -> Result<i32> {
    layout.validate_rescale_parameters()?;
    if layout.rescale_slope == 0.0 {
        bail!("rescale_slope=0 has no inverse; encoding is undefined");
    }
    if !value.is_finite() {
        bail!("cannot encode a non-finite sample {value}");
    }
    let stored = (value - layout.rescale_intercept) / layout.rescale_slope;
    Ok(if stored < 0.0 {
        (stored - 0.5).round() as i32
    } else {
        (stored + 0.5).round() as i32
    })
}

/// Inverse of [`decode_compressed_samples`] for the eight-bit grayscale case.
///
/// Values outside the representable range are clamped rather than rejected: a
/// lossy codec tolerates a value that saturates, and failing the whole frame
/// over one saturated sample would make a valid image unwritable. The clamp is
/// at the *encoding* boundary on purpose -- the decode path still rejects
/// out-of-range stored samples, so this asymmetry cannot mask a decoder bug.
pub(crate) fn encode_gray_u8_samples<I>(values: I, layout: PixelLayout) -> Result<Vec<u8>>
where
    I: ExactSizeIterator<Item = f32>,
{
    let expected = layout.samples_per_frame()?;
    if values.len() != expected {
        bail!(
            "encoder received {} samples; layout expects {expected}",
            values.len()
        );
    }
    values
        .map(|value| {
            let stored = encode_stored_sample(value, layout)?;
            Ok(u8::try_from(stored.clamp(0, i32::from(u8::MAX))).unwrap_or(u8::MAX))
        })
        .collect()
}

pub(crate) fn decode_compressed_samples<I>(
    samples: I,
    sample_precision: u8,
    layout: PixelLayout,
) -> Result<Vec<f32>>
where
    I: ExactSizeIterator<Item = u16>,
{
    layout.bytes_per_frame()?;
    layout.validate_rescale_parameters()?;
    if !(2..=16).contains(&sample_precision) {
        bail!("compressed sample precision={sample_precision} is outside 2..=16");
    }
    if layout.bits_stored != u16::from(sample_precision) {
        bail!(
            "compressed sample precision {} does not match DICOM BitsStored={}",
            sample_precision,
            layout.bits_stored
        );
    }
    let expected_samples = layout.samples_per_frame()?;
    if samples.len() != expected_samples {
        bail!(
            "compressed decoder returned {} samples; expected {}",
            samples.len(),
            expected_samples
        );
    }

    let maximum = if sample_precision == 16 {
        u16::MAX
    } else {
        (1_u16 << sample_precision) - 1
    };
    let sign_bit = 1_u16 << (sample_precision - 1);
    let modulus = 1_i32 << sample_precision;
    samples
        .map(|raw| {
            if raw > maximum {
                bail!(
                    "compressed sample {} exceeds {}-bit maximum {}",
                    raw,
                    sample_precision,
                    maximum
                );
            }
            let value = match layout.pixel_representation {
                PixelSignedness::Signed if raw & sign_bit != 0 => {
                    let signed = i32::from(raw) - modulus;
                    f32::from(
                        i16::try_from(signed)
                            .context("signed sample must fit the 16-bit precision domain")?,
                    )
                }
                PixelSignedness::Signed | PixelSignedness::Unsigned => f32::from(raw),
            };
            Ok(value * layout.rescale_slope + layout.rescale_intercept)
        })
        .collect()
}
