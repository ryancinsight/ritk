//! Structural Part 10 scanner used by the budget-aware backend.

use std::str;

use anyhow::{bail, Context, Result};
use consus_core::ParseBudget;
use dicom::core::{Tag, VR};

use crate::syntax::TransferSyntaxKind;

const DICM_MAGIC: &[u8; 4] = b"DICM";
const UNDEFINED_LENGTH: u32 = u32::MAX;
const FILE_META_GROUP: u16 = 0x0002;
const ITEM_GROUP: u16 = 0xFFFE;
const ITEM_ELEMENT: u16 = 0xE000;
const ITEM_DELIMITER_ELEMENT: u16 = 0xE00D;
const SEQUENCE_DELIMITER_ELEMENT: u16 = 0xE0DD;
const PIXEL_DATA: Tag = Tag(0x7FE0, 0x0010);
const FILE_META_GROUP_LENGTH: Tag = Tag(FILE_META_GROUP, 0x0000);
const TRANSFER_SYNTAX_UID: Tag = Tag(FILE_META_GROUP, 0x0010);

mod sequence;

#[derive(Debug, Clone, Copy)]
struct SyntaxEncoding {
    implicit_vr: bool,
    little_endian: bool,
}

impl SyntaxEncoding {
    fn from_transfer_syntax(syntax: &TransferSyntaxKind) -> Result<Self> {
        match syntax {
            TransferSyntaxKind::ImplicitVrLittleEndian => Ok(Self {
                implicit_vr: true,
                little_endian: true,
            }),
            TransferSyntaxKind::ExplicitVrLittleEndian => Ok(Self {
                implicit_vr: false,
                little_endian: true,
            }),
            TransferSyntaxKind::ExplicitVrBigEndian => Ok(Self {
                implicit_vr: false,
                little_endian: false,
            }),
            TransferSyntaxKind::JpegBaseline
            | TransferSyntaxKind::JpegExtended
            | TransferSyntaxKind::JpegLosslessNonHierarchical
            | TransferSyntaxKind::JpegLosslessFirstOrderPrediction
            | TransferSyntaxKind::JpegLsLossless
            | TransferSyntaxKind::JpegLsLossy
            | TransferSyntaxKind::Jpeg2000Lossless
            | TransferSyntaxKind::Jpeg2000Lossy
            | TransferSyntaxKind::RleLossless
            | TransferSyntaxKind::JpegXlLossless
            | TransferSyntaxKind::JpegXlJpegRecompression
            | TransferSyntaxKind::JpegXl => Ok(Self {
                implicit_vr: false,
                little_endian: true,
            }),
            TransferSyntaxKind::DeflatedExplicitVrLittleEndian => {
                bail!(
                    "DICOM transfer syntax {} requires a bounded deflate adapter",
                    syntax.uid()
                )
            }
            TransferSyntaxKind::Unknown(_) => bail!(
                "DICOM transfer syntax {} is not a dataset syntax supported by the bounded scanner",
                syntax.uid()
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ElementHeader {
    tag: Tag,
    vr: VR,
    length: u32,
    bytes: usize,
}

/// Result of structural validation before object materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DicomParseSummary {
    /// Transfer syntax selected by the Part 10 file meta information.
    pub transfer_syntax: TransferSyntaxKind,
    /// Number of data elements, sequence items, and delimiters observed.
    pub elements: usize,
    /// Deepest sequence/item nesting encountered.
    pub max_depth: u16,
    /// Encoded headers and values accounted against the byte ceiling.
    pub encoded_bytes: usize,
}

struct Scanner<'input> {
    data: &'input [u8],
    budget: &'input ParseBudget,
    accounted_bytes: usize,
    elements: usize,
    max_depth: u16,
}

/// Validate a DICOM Part 10 byte stream before object construction.
///
/// The scanner never copies a declared value. It only reads headers and
/// checks each declared span against the available input and the supplied
/// [`ParseBudget`]. Sequence and item bodies are traversed so a malformed
/// nested length cannot reach the allocating backend.
///
/// # Errors
///
/// Returns an error for missing Part 10 framing, unsupported dataset transfer
/// syntax, malformed headers or delimiters, truncated values, and exceeded
/// byte, element, or depth ceilings.
pub fn validate_part10(data: &[u8], budget: &ParseBudget) -> Result<DicomParseSummary> {
    let mut scanner = Scanner::new(data, budget)?;
    scanner.scan()
}

impl<'input> Scanner<'input> {
    fn new(data: &'input [u8], budget: &'input ParseBudget) -> Result<Self> {
        let length = u64::try_from(data.len()).context("DICOM input length does not fit u64")?;
        budget
            .checked_bytes(length, "DICOM encoded input")
            .map_err(|error| anyhow::anyhow!("DICOM input exceeds parse budget: {error}"))?;
        Ok(Self {
            data,
            budget,
            accounted_bytes: 0,
            elements: 0,
            max_depth: 0,
        })
    }

    fn scan(&mut self) -> Result<DicomParseSummary> {
        let dataset_start = if self.data.get(128..132) == Some(DICM_MAGIC.as_slice()) {
            132
        } else if self.data.get(..4) == Some(DICM_MAGIC.as_slice()) {
            4
        } else {
            bail!("DICOM Part 10 preamble and DICM marker are missing")
        };

        let (dataset_start, transfer_syntax) = self.scan_file_meta(dataset_start)?;
        let encoding = SyntaxEncoding::from_transfer_syntax(&transfer_syntax)?;
        let end = self.scan_dataset(dataset_start, self.data.len(), encoding, 0)?;
        if end != self.data.len() {
            bail!(
                "DICOM data set ended at {end} bytes, input has {}",
                self.data.len()
            )
        }
        Ok(DicomParseSummary {
            transfer_syntax,
            elements: self.elements,
            max_depth: self.max_depth,
            encoded_bytes: self.accounted_bytes,
        })
    }

    fn scan_file_meta(&mut self, start: usize) -> Result<(usize, TransferSyntaxKind)> {
        let encoding = SyntaxEncoding {
            implicit_vr: false,
            little_endian: true,
        };
        let group_length_header = self.read_element_header(start, encoding)?;
        if group_length_header.tag != FILE_META_GROUP_LENGTH
            || group_length_header.vr != VR::UL
            || group_length_header.length != 4
        {
            bail!("DICOM file meta must begin with (0002,0000) UL length 4")
        }
        self.account_element(group_length_header)?;
        let group_length_start = start
            .checked_add(group_length_header.bytes)
            .context("DICOM file meta group length offset overflow")?;
        let group_length_end = self.span_end(group_length_start, 4, self.data.len())?;
        let group_length = self.read_u32(group_length_start, true)?;
        self.account_value(4, "DICOM file meta group length")?;
        let group_end = group_length_end
            .checked_add(
                usize::try_from(group_length).context("DICOM meta length does not fit usize")?,
            )
            .context("DICOM file meta group end overflow")?;
        if group_end > self.data.len() {
            bail!("DICOM file meta group extends beyond input")
        }

        let mut cursor = group_length_end;
        let mut transfer_syntax = None;
        while cursor < group_end {
            let header = self.read_element_header(cursor, encoding)?;
            if header.tag.group() != FILE_META_GROUP || header.length == UNDEFINED_LENGTH {
                bail!("invalid DICOM file meta element at byte {cursor}")
            }
            self.account_element(header)?;
            let value_start = cursor
                .checked_add(header.bytes)
                .context("DICOM file meta value offset overflow")?;
            let value_end = self.span_end(value_start, header.length, group_end)?;
            if header.tag == TRANSFER_SYNTAX_UID {
                let value = self
                    .data
                    .get(value_start..value_end)
                    .context("DICOM transfer syntax value disappeared after span validation")?;
                let value =
                    str::from_utf8(value).context("DICOM transfer syntax UID is not UTF-8")?;
                transfer_syntax = Some(TransferSyntaxKind::from_uid(value));
            }
            self.account_value(
                usize::try_from(header.length)
                    .context("DICOM meta value length does not fit usize")?,
                "DICOM file meta value",
            )?;
            cursor = value_end;
        }
        if cursor != group_end {
            bail!("DICOM file meta group length does not match its elements")
        }
        let transfer_syntax =
            transfer_syntax.context("DICOM TransferSyntaxUID (0002,0010) is missing")?;
        Ok((group_end, transfer_syntax))
    }

    fn scan_dataset(
        &mut self,
        start: usize,
        end: usize,
        encoding: SyntaxEncoding,
        depth: u16,
    ) -> Result<usize> {
        let mut cursor = start;
        while cursor < end {
            let header = self.read_element_header(cursor, encoding)?;
            if header.tag.group() == ITEM_GROUP {
                bail!("unexpected sequence item marker at byte {cursor}")
            }
            cursor = self.scan_element(cursor, end, encoding, depth, header)?;
        }
        if cursor != end {
            bail!("DICOM data set ended at {cursor} bytes, expected {end}")
        }
        Ok(cursor)
    }

    fn read_element_header(
        &self,
        cursor: usize,
        encoding: SyntaxEncoding,
    ) -> Result<ElementHeader> {
        let tag = self.read_tag(cursor, encoding.little_endian)?;
        if tag.group() == ITEM_GROUP {
            bail!("sequence item marker cannot be read as a data element")
        }
        if encoding.implicit_vr {
            let length_start = cursor
                .checked_add(4)
                .context("DICOM element length offset overflow")?;
            let length = self.read_u32(length_start, encoding.little_endian)?;
            return Ok(ElementHeader {
                tag,
                vr: VR::UN,
                length,
                bytes: 8,
            });
        }
        let vr_start = cursor
            .checked_add(4)
            .context("DICOM value representation offset overflow")?;
        let vr_bytes = self.read_array::<2>(vr_start)?;
        let vr = VR::from_binary(vr_bytes).unwrap_or(VR::UN);
        let bytes = if uses_short_length(vr) { 8 } else { 12 };
        let length = if bytes == 8 {
            let length_start = cursor
                .checked_add(6)
                .context("DICOM short length offset overflow")?;
            u32::from(self.read_u16(length_start, encoding.little_endian)?)
        } else {
            let length_start = cursor
                .checked_add(8)
                .context("DICOM long length offset overflow")?;
            self.read_u32(length_start, encoding.little_endian)?
        };
        Ok(ElementHeader {
            tag,
            vr,
            length,
            bytes,
        })
    }

    fn next_tag_is_item(&self, cursor: usize) -> bool {
        self.read_tag(cursor, true)
            .is_ok_and(|tag| tag == Tag(ITEM_GROUP, ITEM_ELEMENT))
    }

    fn read_tag(&self, cursor: usize, little_endian: bool) -> Result<Tag> {
        let group = self.read_u16(cursor, little_endian)?;
        let element = self.read_u16(
            cursor
                .checked_add(2)
                .context("DICOM tag element offset overflow")?,
            little_endian,
        )?;
        Ok(Tag(group, element))
    }

    fn read_u16(&self, cursor: usize, little_endian: bool) -> Result<u16> {
        let bytes = self.read_array::<2>(cursor)?;
        Ok(if little_endian {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    }

    fn read_u32(&self, cursor: usize, little_endian: bool) -> Result<u32> {
        let bytes = self.read_array::<4>(cursor)?;
        Ok(if little_endian {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    }

    fn read_array<const N: usize>(&self, cursor: usize) -> Result<[u8; N]> {
        let end = cursor
            .checked_add(N)
            .context("DICOM header offset overflow")?;
        let bytes = self
            .data
            .get(cursor..end)
            .context("truncated DICOM header")?;
        bytes
            .try_into()
            .map_err(|_| anyhow::anyhow!("DICOM header has an invalid byte width"))
    }

    fn span_end(&self, start: usize, length: u32, parent_end: usize) -> Result<usize> {
        let length = usize::try_from(length).context("DICOM declared length does not fit usize")?;
        let end = start
            .checked_add(length)
            .context("DICOM declared span overflows usize")?;
        if end > parent_end || end > self.data.len() {
            bail!("DICOM declared span [{start}, {end}) exceeds available input")
        }
        Ok(end)
    }

    fn account_element(&mut self, header: ElementHeader) -> Result<()> {
        self.account_header(header.bytes, "DICOM element header")?;
        self.count_element("DICOM element")
    }

    fn account_header(&mut self, bytes: usize, what: &'static str) -> Result<()> {
        self.account(bytes, what)
    }

    fn account_value(&mut self, bytes: usize, what: &'static str) -> Result<()> {
        self.account(bytes, what)
    }

    fn account(&mut self, bytes: usize, what: &'static str) -> Result<()> {
        let declared = u64::try_from(bytes).context("DICOM accounted bytes do not fit u64")?;
        self.budget
            .checked_bytes(declared, what)
            .map_err(|error| anyhow::anyhow!("DICOM parse budget exceeded: {error}"))?;
        let next = self
            .accounted_bytes
            .checked_add(bytes)
            .context("DICOM accounted byte total overflow")?;
        if next > self.budget.max_alloc_bytes {
            bail!(
                "DICOM cumulative parse bytes {next} exceed budget {}",
                self.budget.max_alloc_bytes
            )
        }
        self.accounted_bytes = next;
        Ok(())
    }

    fn count_element(&mut self, what: &'static str) -> Result<()> {
        let next = self
            .elements
            .checked_add(1)
            .context("DICOM element count overflow")?;
        if next > self.budget.max_elements {
            bail!(
                "DICOM {what} count {next} exceeds budget {}",
                self.budget.max_elements
            )
        }
        self.elements = next;
        Ok(())
    }

    fn enter_depth(&mut self, depth: u16) -> Result<u16> {
        let nested = self
            .budget
            .descend(depth, "DICOM sequence nesting")
            .map_err(|error| anyhow::anyhow!("DICOM parse depth budget exceeded: {error}"))?;
        self.max_depth = self.max_depth.max(nested);
        Ok(nested)
    }
}

fn uses_short_length(vr: VR) -> bool {
    matches!(
        vr,
        VR::AE
            | VR::AS
            | VR::AT
            | VR::CS
            | VR::DA
            | VR::DS
            | VR::DT
            | VR::FL
            | VR::FD
            | VR::IS
            | VR::LO
            | VR::LT
            | VR::PN
            | VR::SH
            | VR::SL
            | VR::SS
            | VR::ST
            | VR::TM
            | VR::UI
            | VR::UL
            | VR::US
    )
}

#[cfg(test)]
#[path = "tests_scan.rs"]
mod tests;
