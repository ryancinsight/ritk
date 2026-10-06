//! Encapsulated-codec sample conversion into the DICOM modality domain.

use anyhow::{bail, Context, Result};
use eunomia::convert::IntegerTarget;

use super::{PixelLayout, PixelSignedness};

/// Inverse of [`decode_compressed_samples`] for the eight-bit grayscale case.
///
/// Values outside the representable range are clamped rather than rejected: a
/// lossy codec tolerates a value that saturates, and failing the whole frame
/// over one saturated sample would make a valid image unwritable. The clamp is
/// at the *encoding* boundary on purpose -- the decode path still rejects
/// out-of-range stored samples, so this asymmetry cannot mask a decoder bug.
pub(crate) fn encode_grayscale_stored_bytes<I>(values: I, layout: PixelLayout) -> Result<Vec<u8>>
where
    I: ExactSizeIterator<Item = f32>,
{
    layout.validate_rescale_parameters()?;
    if layout.rescale_slope == 0.0 {
        bail!("rescale_slope=0 has no inverse; encoding is undefined");
    }
    let expected = layout.samples_per_frame()?;
    if values.len() != expected {
        bail!(
            "encoder received {} samples; layout expects {expected}",
            values.len()
        );
    }
    let maximum = f32::from(u8::MAX);
    values
        .map(|value| {
            if !value.is_finite() {
                bail!("cannot encode a non-finite sample {value}");
            }
            let stored = (value - layout.rescale_intercept) / layout.rescale_slope;
            let rounded = stored.clamp(0.0, maximum).round();
            u8::try_from_rounded(rounded)
                .context("clamped JPEG sample must fit the unsigned eight-bit range")
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

#[cfg(test)]
mod tests {
    use super::encode_grayscale_stored_bytes;
    use crate::pixel_layout::{PixelLayout, PixelSignedness};

    fn layout(slope: f32, intercept: f32) -> PixelLayout {
        PixelLayout {
            rows: 1,
            cols: 3,
            samples_per_pixel: 1,
            bits_allocated: 8,
            bits_stored: 8,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: slope,
            rescale_intercept: intercept,
        }
    }

    #[test]
    fn grayscale_encoding_rounds_and_saturates_the_rescaled_samples() {
        let encoded = encode_grayscale_stored_bytes(
            [-100.0, 0.0, 509.0].into_iter(),
            layout(2.0, -1.0),
        )
        .expect("finite samples matching the layout fit the clamped output range");

        assert_eq!(encoded, [0, 1, 255]);
    }

    #[test]
    fn grayscale_encoding_rejects_non_finite_samples_and_invalid_layouts() {
        let error = encode_grayscale_stored_bytes(
            [f32::NAN, 0.0, 1.0].into_iter(),
            layout(1.0, 0.0),
        )
        .expect_err("NaN cannot produce a stored sample");
        assert!(error.to_string().contains("non-finite sample"));

        let error = encode_grayscale_stored_bytes(
            [0.0, 1.0, 2.0].into_iter(),
            layout(0.0, 0.0),
        )
        .expect_err("zero slope has no inverse mapping");
        assert!(error.to_string().contains("rescale_slope=0"));

        let error = encode_grayscale_stored_bytes([0.0, 1.0].into_iter(), layout(1.0, 0.0))
            .expect_err("the encoded count must match the frame layout");
        assert!(error.to_string().contains("layout expects 3"));
    }
}
