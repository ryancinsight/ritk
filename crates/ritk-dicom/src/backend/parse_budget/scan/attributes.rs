//! Bounded selection of top-level DICOM values by tag.

use anyhow::{bail, Result};
use consus_core::ParseBudget;
use dicom::core::Tag;

use crate::{DicomTag, TransferSyntaxKind};

use super::{RootValue, Scanner};

/// Select top-level values from a structurally validated Part 10 file.
///
/// The records borrow the original value bytes. The selection stores N tag
/// slots, so its storage depends on the requested tags and remains independent
/// of the pixel size, total tag count, and number of nested items.
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
    let summary = Scanner::new(data, budget, |tag: Tag, value| {
        let key = DicomTag::new(tag.group(), tag.element());
        if let Some(position) = tags.iter().position(|wanted| *wanted == key) {
            values[position] = Some(value);
        }
    })?
    .scan()?;
    Ok(Part10Attributes {
        data,
        tags,
        values,
        transfer_syntax: summary.transfer_syntax,
    })
}

/// A fixed-size selection of top-level attributes from a Part 10 dataset.
#[derive(Debug)]
pub struct Part10Attributes<'a, const N: usize> {
    data: &'a [u8],
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
    pub fn primitive(&self, tag: DicomTag) -> Result<Option<&'a [u8]>> {
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
        Ok(Some(&self.data[value.range.clone()]))
    }
}
