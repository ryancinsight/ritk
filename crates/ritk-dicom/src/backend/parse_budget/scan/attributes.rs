//! Bounded selection of top-level DICOM values by tag.

use anyhow::{bail, Result};
use bytes::Bytes;
use consus_core::ParseBudget;
use std::ops::Range;

use crate::{DicomTag, TransferSyntaxKind};

use super::{Event, Scanner};

#[derive(Debug)]
struct RootValue {
    range: Range<usize>,
    compound: bool,
}

/// Select top-level values from a structurally validated Part 10 file.
///
/// Selection stores N tag slots and borrows existing value spans. Deflated
/// input owns one bounded inflated buffer that supplies those spans.
///
/// # Errors
/// Returns the same framing, transfer syntax, and budget failures as
/// [super::validate_part10]. Selected values must be decoded using their
/// respective DICOM representations.
pub fn scan_part10_attributes<'a, const N: usize>(
    data: &'a [u8],
    budget: &ParseBudget,
    tags: [DicomTag; N],
) -> Result<Part10Attributes<'a, N>> {
    let mut values: [Option<RootValue>; N] = [const { None }; N];
    let mut inflated = None;
    let summary = Scanner::new(data, budget, |event| {
        let (tag, range, compound) = match event {
            Event::Primitive {
                header,
                range,
                depth: 0,
                ..
            } => (header.tag, range, false),
            Event::SequenceEnd {
                tag,
                range,
                depth: 0,
            } => (tag, range, true),
            Event::Inflated(bytes) => {
                inflated = Some(bytes);
                return Ok(());
            }
            _ => return Ok(()),
        };
        for (index, wanted) in tags.iter().enumerate() {
            if *wanted == tag {
                values[index] = Some(RootValue {
                    range: range.clone(),
                    compound,
                });
            }
        }
        Ok(())
    })?
    .scan()?;
    Ok(Part10Attributes {
        data,
        inflated,
        tags,
        values,
        transfer_syntax: summary.transfer_syntax,
    })
}

/// A fixed-size selection of top-level attributes from a Part 10 dataset.
#[derive(Debug)]
pub struct Part10Attributes<'a, const N: usize> {
    data: &'a [u8],
    inflated: Option<Bytes>,
    tags: [DicomTag; N],
    values: [Option<RootValue>; N],
    transfer_syntax: TransferSyntaxKind,
}

impl<'a, const N: usize> Part10Attributes<'a, N> {
    /// Dataset transfer syntax, declared in the file meta.
    #[must_use]
    pub const fn transfer_syntax(&self) -> &TransferSyntaxKind {
        &self.transfer_syntax
    }

    /// Borrow the original encoded bytes of a selected primitive.
    ///
    /// Returns None for an absent or unselected tag. A malformed compound
    /// representation for a selected primitive gives a separate error.
    ///
    /// # Errors
    /// Fails for sequences and encapsulated pixel fragments.
    pub fn primitive(&self, tag: DicomTag) -> Result<Option<&[u8]>> {
        let selected = self
            .tags
            .iter()
            .position(|wanted| *wanted == tag)
            .and_then(|position| self.values[position].as_ref());
        let Some(value) = selected else {
            return Ok(None);
        };
        if value.compound {
            bail!(
                "DICOM element ({:04X},{:04X}) is a compound value",
                tag.group,
                tag.element
            );
        }
        let data = self.inflated.as_deref().unwrap_or(self.data);
        Ok(Some(&data[value.range.clone()]))
    }
}
