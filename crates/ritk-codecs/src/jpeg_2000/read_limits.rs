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
) -> Result<(), Jpeg2000DecodeError> {
    layout
        .bytes_per_frame()
        .context("J2K: invalid DICOM pixel layout")?;
    let samples = layout
        .samples_per_frame()
        .context("J2K: invalid DICOM pixel layout")?;
    budget.check_len(ImageReadResource::EncodedBytes, fragment_len)?;
    budget.check_scaled(ImageReadResource::DecodedBytes, samples, size_of::<f32>())?;
    Ok(())
}
