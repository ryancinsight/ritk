//! Encapsulated-codec sample conversion into the DICOM modality domain.

use anyhow::{bail, Context, Result};

use super::{PixelLayout, PixelSignedness};

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
