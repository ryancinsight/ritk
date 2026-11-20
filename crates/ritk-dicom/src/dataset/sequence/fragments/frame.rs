//! The frame offset origin is the first fragment item tag after the BOT.

use std::borrow::Cow;
use std::ops::Range;

use anyhow::{bail, Context, Result};
use bytes::Bytes;

use super::PixelFragments;

impl PixelFragments {
    pub(crate) fn frame(
        &self,
        index: usize,
        count: usize,
        extended: Option<(&[u64], &[u64])>,
    ) -> Result<Cow<'_, [u8]>> {
        if index >= count || self.fragments.is_empty() {
            bail!(
                "DICOM frame {index} out of range for {count} frames and {} fragments",
                self.fragments.len()
            );
        }
        let (range, declared_length) = if let Some((offsets, lengths)) = extended {
            if !self.offsets.is_empty() || offsets.len() != count || lengths.len() != count {
                bail!("DICOM Extended Offset Table disagrees with NumberOfFrames or the Basic Offset Table");
            }
            let range = bounds(offsets, &self.fragments, index)?;
            // An Extended Offset Table is only legal for exactly one fragment
            // per frame. Its length excludes that fragment's even-byte pad.
            if range.len() != 1 {
                bail!("DICOM Extended Offset Table frame {index} spans multiple fragments");
            }
            (range, Some(usize::try_from(lengths[index])?))
        } else if !self.offsets.is_empty() {
            if self.offsets.len() != count {
                bail!("DICOM Basic Offset Table count differs from NumberOfFrames");
            }
            (bounds(&self.offsets, &self.fragments, index)?, None)
        } else if count == 1 {
            (0..self.fragments.len(), None)
        } else if self.fragments.len() == count {
            (index..index + 1, None)
        } else {
            bail!("DICOM frame boundary is ambiguous without an offset table");
        };

        let pieces = &self.fragments[range];
        if let [only] = pieces {
            let length = declared_length.unwrap_or(only.len());
            if length > only.len() || only.len() - length > 1 {
                bail!("DICOM Extended Offset Table length disagrees with frame {index}");
            }
            return Ok(Cow::Borrowed(&only[..length]));
        }
        let size = pieces.iter().try_fold(0usize, |sum, part| {
            sum.checked_add(part.len())
                .context("DICOM encoded frame size overflows")
        })?;
        let budget = ritk_codecs::ImageReadBudget::DEFAULT;
        budget.check(
            ritk_codecs::ImageReadResource::EncodedBytes,
            u64::try_from(size)?,
        )?;
        let mut joined = Vec::with_capacity(size);
        for part in pieces {
            joined.extend_from_slice(part);
        }
        Ok(Cow::Owned(joined))
    }
}

fn bounds<T: Copy + Into<u64>>(
    offsets: &[T],
    fragments: &[Bytes],
    frame: usize,
) -> Result<Range<usize>> {
    let mut offsets = offsets
        .iter()
        .copied()
        .map(Into::into)
        .enumerate()
        .peekable();
    if offsets.peek().map(|&(_, position)| position) != Some(0) {
        bail!("the first DICOM frame offset must be zero");
    }
    let (mut offset, mut start, mut end) = (0u64, 0, fragments.len());
    for (index, fragment) in fragments.iter().enumerate() {
        if let Some(&(number, next)) = offsets.peek() {
            if next < offset {
                bail!("DICOM frame offset falls inside a fragment");
            }
            if next == offset {
                offsets.next();
                if number == frame {
                    start = index;
                }
                if number == frame + 1 {
                    end = index;
                }
            }
        }
        let size = u64::try_from(fragment.len())?;
        offset = offset
            .checked_add(size)
            .and_then(|n| n.checked_add(size % 2 + 8))
            .context("DICOM fragment position overflow")?;
    }
    if offsets.peek().is_some() {
        bail!("DICOM frame offset does not name a fragment");
    }
    Ok(start..end)
}
