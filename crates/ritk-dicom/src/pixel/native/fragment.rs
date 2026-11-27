//! DICOM PS3.5 A.4.11 and A.4.13 encode each frame in its own fragment.
use super::{NativeScalarPixels, ScalarEncoding};
use crate::backend::{parse_budget::inflate_bounded, EncapsulatedFrameSource};
use crate::dataset::DicomFile;
use crate::{tags, DicomTag, PixelLayout, TransferSyntaxKind};
use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use ritk_codecs::{ImageReadBudget, ImageReadResource};
use std::borrow::Cow;

fn frame_bytes(samples: usize, allocated: u16) -> Result<usize> {
    ensure!(
        allocated == 1 || (allocated > 0 && allocated.is_multiple_of(8)),
        "invalid frame allocation"
    );
    Ok(samples
        .checked_mul(usize::from(allocated))
        .context("encapsulated native frame bit count overflow")?
        .div_ceil(8))
}

fn plain<'a>(fragment: &'a [u8], syntax: &TransferSyntaxKind, n: usize) -> Result<Cow<'a, [u8]>> {
    ImageReadBudget::DEFAULT.check(ImageReadResource::DecodedBytes, u64::try_from(n)?)?;
    if *syntax == TransferSyntaxKind::DeflatedImageFrame {
        let data = inflate_bounded(fragment, n)?;
        ensure!(
            data.len() == n,
            "deflated DICOM frame has incorrect decoded length"
        );
        return Ok(Cow::Owned(Vec::from(data)));
    }
    ensure!(
        fragment.len() == n || (n % 2 == 1 && fragment.len() == n + 1),
        "encapsulated uncompressed frame has incorrect length"
    );
    Ok(Cow::Borrowed(&fragment[..n]))
}

impl NativeScalarPixels {
    pub(crate) fn decode_encapsulated(
        fragment: &[u8],
        syntax: &TransferSyntaxKind,
        layout: PixelLayout,
    ) -> Result<Vec<f32>> {
        ensure!(
            layout.bits_stored > 0 && layout.bits_stored <= layout.bits_allocated,
            "invalid stored pixel precision"
        );
        let count = layout.samples_per_frame()?;
        let raw = plain(fragment, syntax, frame_bytes(count, layout.bits_allocated)?)?;
        if matches!(layout.bits_allocated, 8 | 16 | 24 | 32) {
            return crate::decode_native_pixel_bytes_checked(&raw, layout);
        }
        let values = Self {
            data: Bytes::from(raw.into_owned()),
            encoding: ScalarEncoding::Integer,
            allocated: layout.bits_allocated,
            stored: layout.bits_stored,
            signed: layout.pixel_representation.is_signed(),
            rescale: [1., 0.],
            frame_rescale: Vec::new(),
        };
        values.decode(0, 1, layout)
    }
}

/// Retain exact native words for wide scalar readout. Trim the fragment-local
/// padding before joining frames so it cannot become a sample in the next one.
pub(super) fn collect_byte_aligned(
    obj: &DicomFile,
    syntax: &TransferSyntaxKind,
    allocated: u16,
) -> Result<Bytes> {
    ensure!(
        allocated.is_multiple_of(8) && allocated != 0,
        "exact byte-aligned samples required"
    );
    let frames = obj
        .get(DicomTag::new(0x0028, 0x0008))
        .map_or(Ok(1), |e| e.to_num::<usize>())?;
    let rows = obj.element(tags::ROWS)?.to_num::<usize>()?;
    let cols = obj.element(tags::COLUMNS)?.to_num::<usize>()?;
    let n = frame_bytes(
        rows.checked_mul(cols)
            .context("fragment sample count overflow")?,
        allocated,
    )?;
    let total = n
        .checked_mul(frames)
        .context("encapsulated native volume length overflow")?;
    ImageReadBudget::DEFAULT.check(ImageReadResource::DecodedBytes, u64::try_from(total)?)?;
    ensure!(n > 0 && frames > 0, "empty encapsulated native volume");
    let mut out = Vec::new();
    for index in 0..frames {
        let frame = obj.encapsulated_frame(u32::try_from(index)?)?;
        out.extend_from_slice(&plain(&frame, syntax, n)?);
    }
    Ok(out.into())
}
