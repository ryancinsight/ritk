use anyhow::Context;

use super::Jpeg2000DecodeError;
use crate::dimensions::{checked_pixel_count, checked_sample_count};
use crate::{ImageReadBudget, ImageReadResource, PixelLayout};

/// Checks the encoded fragment length and the `f32` output implied by `layout`
/// against `budget`, before the codestream is parsed or output is allocated.
pub(super) fn check_read_budget(
    fragment_len: usize,
    layout: PixelLayout,
    budget: ImageReadBudget,
) -> Result<(), Jpeg2000DecodeError> {
    budget.check_len(ImageReadResource::EncodedBytes, fragment_len)?;
    let pixels = checked_pixel_count(layout.cols, layout.rows).context("J2K image")?;
    let samples =
        checked_sample_count(pixels, layout.samples_per_pixel).context("J2K output samples")?;
    budget.check_len(
        ImageReadResource::DecodedBytes,
        samples.saturating_mul(size_of::<f32>()),
    )?;
    Ok(())
}
