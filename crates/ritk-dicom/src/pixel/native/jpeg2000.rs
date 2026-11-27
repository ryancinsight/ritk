//! Preserve wide monochrome JPEG 2000 component values before modality rescale.
use super::{NativeScalarPixels, ScalarEncoding};
use crate::backend::EncapsulatedFrameSource;
use crate::dataset::DicomFile;
use crate::{DicomTag, PixelLayout};
use anyhow::{ensure, Context, Result};
use ritk_codecs::jpeg_2000::{decode_jpeg2000_stored_fragment, grayscale_component_precision};
use ritk_codecs::{ImageReadBudget, ImageReadResource};

impl NativeScalarPixels {
    /// Retain exact source samples if any frame's codestream can exceed f32 precision.
    /// DICOM PS3.5 §8.2.4 gives the codestream precedence over the enclosing dataset.
    pub fn from_wide_jpeg2000(obj: &DicomFile, layout: PixelLayout) -> Result<Option<Self>> {
        let frames = obj
            .get(DicomTag::new(0x0028, 0x0008))
            .map_or(Ok(1), |e| e.to_num::<u32>())?;
        ensure!(frames > 0, "empty JPEG 2000 volume");
        for index in 0..frames {
            let frame = obj.encapsulated_frame(index)?;
            if grayscale_component_precision(&frame)? > 24 {
                return Self::collect_jpeg2000(obj, layout, frames).map(Some);
            }
        }
        Ok(None)
    }

    fn collect_jpeg2000(obj: &DicomFile, layout: PixelLayout, frames: u32) -> Result<Self> {
        let count = layout
            .samples_per_frame()?
            .checked_mul(usize::try_from(frames)?)
            .context("JPEG 2000 sample count overflow")?;
        let bytes = count
            .checked_mul(size_of::<i64>())
            .context("JPEG 2000 exact storage overflow")?;
        let budget = ImageReadBudget::DEFAULT;
        budget.check_len(ImageReadResource::DecodedBytes, bytes)?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)?;
        for index in 0..frames {
            let frame = obj.encapsulated_frame(index)?;
            for value in decode_jpeg2000_stored_fragment(&frame, layout, budget)? {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(Self {
            data: data.into(),
            encoding: ScalarEncoding::Integer,
            allocated: 64,
            stored: 64,
            signed: true,
            rescale: Self::modality_rescale(obj),
            frame_rescale: Vec::new(),
        })
    }
}
