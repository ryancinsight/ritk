//! JPEG XL codestream adapter. Modality rescale remains in the stored-sample decoder.

use anyhow::{bail, Context, Result};
use jxl_oxide::{AllocTracker, JxlImage};
use ritk_codecs::ImageReadBudget;

use crate::{decode_native_pixel_bytes_checked, PixelLayout};

pub(crate) fn decode(data: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    let budget = ImageReadBudget::DEFAULT;
    budget.check(
        ritk_codecs::ImageReadResource::EncodedBytes,
        u64::try_from(data.len())?,
    )?;
    let allocated = layout.bytes_per_sample()?;
    if !matches!(allocated, 1 | 2) {
        bail!("DICOM JPEG XL requires 8 or 16 allocated bits");
    }
    let samples = layout.samples_per_frame()?;
    let output_bytes = samples
        .checked_mul(size_of::<f32>())
        .context("JPEG XL output byte count overflows")?;
    budget.check(
        ritk_codecs::ImageReadResource::DecodedBytes,
        u64::try_from(output_bytes)?,
    )?;
    let image = JxlImage::builder()
        .alloc_tracker(AllocTracker::with_limit(usize::try_from(
            budget.max_decoded_bytes(),
        )?))
        .read(data)
        .map_err(|error| anyhow::anyhow!("read DICOM JPEG XL frame: {error}"))?;
    if usize::try_from(image.width())? != layout.cols
        || usize::try_from(image.height())? != layout.rows
    {
        bail!("DICOM JPEG XL dimensions disagree with Rows or Columns");
    }
    let stored = image.image_header().metadata.bit_depth.bits_per_sample();
    if stored == 0 || stored > u32::from(layout.bits_allocated) {
        bail!("DICOM JPEG XL precision exceeds BitsAllocated");
    }
    let frame = image
        .render_frame(0)
        .map_err(|error| anyhow::anyhow!("render DICOM JPEG XL frame: {error}"))?;
    let mut stream = frame.stream();
    if usize::try_from(stream.channels())? != layout.samples_per_pixel {
        bail!("DICOM JPEG XL channels disagree with SamplesPerPixel");
    }
    let scale = ((1u32 << stored) - 1) as f32;
    let mut raw = Vec::with_capacity(layout.bytes_per_frame()?);
    let mut batch = [0.0f32; 4096];
    let mut remaining = samples;
    while remaining != 0 {
        let size = remaining.min(batch.len());
        let produced = stream.write_to_buffer(&mut batch[..size]);
        if produced == 0 || produced > remaining {
            bail!("DICOM JPEG XL decoded sample count differs from the declared layout");
        }
        for sample in &batch[..produced] {
            // The renderer supplies normalized values. Rounding to the
            // codestream's integer range recovers all samples through 16 bits.
            let value = (sample.clamp(0.0, 1.0) * scale).round() as u16;
            let word = value.to_le_bytes();
            raw.extend_from_slice(&word[..allocated]);
        }
        remaining -= produced;
    }
    let mut extra = [0f32; 1];
    if stream.write_to_buffer(&mut extra) != 0 {
        bail!("DICOM JPEG XL decoded more samples than declared");
    }
    decode_native_pixel_bytes_checked(&raw, layout)
}
