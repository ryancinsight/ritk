use anyhow::Context;

use super::Jpeg2000DecodeError;
use crate::{ImageReadBudget, ImageReadResource, PixelLayout};

/// Checks the encoded fragment length and the `f32` output implied by `layout`
/// against `budget`, before the codestream is parsed or output is allocated.
///
/// The budget is the only size bound on the decoded frame: the fixed pixel cap
/// of the other codecs does not apply, so a caller ceiling above the default is
/// honored.
pub(super) fn check_read_budget(
    fragment_len: usize,
    layout: PixelLayout,
    budget: ImageReadBudget,
    bytes_per_sample: usize,
) -> Result<(), Jpeg2000DecodeError> {
    if !matches!(layout.bits_allocated, 1 | 8 | 16 | 24 | 32 | 40)
        || layout.bits_stored == 0
        || layout.bits_stored > layout.bits_allocated
        || layout.bits_stored > 38
    {
        return Err(anyhow::anyhow!(
            "J2K: invalid DICOM pixel layout: bits_stored={} bits_allocated={}",
            layout.bits_stored,
            layout.bits_allocated
        )
        .into());
    }
    layout
        .validate_rescale_parameters()
        .context("J2K: invalid modality rescale")?;
    let samples = layout
        .samples_per_frame()
        .context("J2K: invalid DICOM pixel layout")?;
    budget.check_len(ImageReadResource::EncodedBytes, fragment_len)?;
    budget.check_scaled(ImageReadResource::DecodedBytes, samples, bytes_per_sample)?;
    Ok(())
}
