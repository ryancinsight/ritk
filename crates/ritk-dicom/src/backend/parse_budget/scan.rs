//! Structural Part 10 scanner used by the budget-aware backend.

use std::{ops::Range, str};

use crate::dataset::wire::{dictionary, syntax::Encoding};
use crate::dataset::{Length, VR};
use crate::DicomTag;
use anyhow::{bail, Context, Result};
use bytes::Bytes;
use consus_core::{read_integer, ByteOrder, EndianScalar, ParseBudget};

use crate::syntax::TransferSyntaxKind;

const DICM_MAGIC: &[u8; 4] = b"DICM";
const UNDEFINED_LENGTH: u32 = u32::MAX;
const FILE_META_GROUP: u16 = 0x0002;
const ITEM_GROUP: u16 = 0xFFFE;
const ITEM_ELEMENT: u16 = 0xE000;
const ITEM_DELIMITER_ELEMENT: u16 = 0xE00D;
const SEQUENCE_DELIMITER_ELEMENT: u16 = 0xE0DD;
const PIXEL_DATA: DicomTag = DicomTag::new(0x7FE0, 0x0010);
const FILE_META_GROUP_LENGTH: DicomTag = DicomTag::new(FILE_META_GROUP, 0x0000);
const TRANSFER_SYNTAX_UID: DicomTag = DicomTag::new(FILE_META_GROUP, 0x0010);

pub(super) mod attributes;
mod inflate;
pub(super) mod object;
mod sequence;

#[derive(Debug, Clone, Copy)]
struct ElementHeader {
    tag: DicomTag,
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

#[derive(Debug)]
enum Event {
    Meta {
        header: ElementHeader,
        range: Range<usize>,
    },
    Inflated(Bytes),
    Primitive {
        header: ElementHeader,
        range: Range<usize>,
        encoding: Encoding,
        depth: u16,
    },
    SequenceStart {
        header: ElementHeader,
        pixel: bool,
    },
    SequenceEnd {
        tag: DicomTag,
        range: Range<usize>,
        depth: u16,
    },
    ItemStart(Length),
    ItemEnd,
    Fragment(Range<usize>),
}

struct Scanner<'input, Visit> {
    data: &'input [u8],
    budget: &'input ParseBudget,
    visit: Visit,
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
    Scanner::new(data, budget, |_| Ok(()))?.scan()
}

impl<'input, Visit: FnMut(Event) -> Result<()>> Scanner<'input, Visit> {
    fn new(data: &'input [u8], budget: &'input ParseBudget, visit: Visit) -> Result<Self> {
        super::check_input_bytes(data, budget)?;
        Ok(Self {
            data,
            budget,
            visit,
            accounted_bytes: 0,
            elements: 0,
            max_depth: 0,
        })
    }

    fn scan(mut self) -> Result<DicomParseSummary> {
        let dataset_start = if self.data.get(128..132) == Some(DICM_MAGIC.as_slice()) {
            132
        } else if self.data.get(..4) == Some(DICM_MAGIC.as_slice()) {
            4
        } else {
            bail!("DICOM Part 10 preamble and DICM marker are missing")
        };

        let (dataset_start, transfer_syntax) = self.scan_file_meta(dataset_start)?;
        if transfer_syntax == TransferSyntaxKind::DeflatedExplicitVrLittleEndian {
            let available = self.budget.max_alloc_bytes - self.accounted_bytes;
            let plain = inflate::dataset(&self.data[dataset_start..], available)?;
            (self.visit)(Event::Inflated(plain.clone()))?;
            let mut scanner = Scanner {
                data: &plain,
                budget: self.budget,
                visit: self.visit,
                accounted_bytes: self.accounted_bytes,
                elements: self.elements,
                max_depth: self.max_depth,
            };
            return scanner.finish(0, transfer_syntax);
        }
        self.finish(dataset_start, transfer_syntax)
    }

    fn finish(
        &mut self,
        dataset_start: usize,
        transfer_syntax: TransferSyntaxKind,
    ) -> Result<DicomParseSummary> {
        let encoding = Encoding::for_syntax(&transfer_syntax)?;
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
        let encoding = Encoding::EXPLICIT_LE;
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
        let group_length = self.read_field::<u32>(group_length_start, ByteOrder::LittleEndian)?;
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
            if header.tag.group != FILE_META_GROUP || header.length == UNDEFINED_LENGTH {
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
            (self.visit)(Event::Meta {
                header,
                range: value_start..value_end,
            })?;
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
        encoding: Encoding,
        depth: u16,
    ) -> Result<usize> {
        let mut cursor = start;
        while cursor < end {
            let header = self.read_element_header(cursor, encoding)?;
            if header.tag.group == ITEM_GROUP {
                bail!("unexpected sequence item marker at byte {cursor}")
            }
            cursor = self.scan_element(cursor, end, encoding, depth, header)?;
        }
        if cursor != end {
            bail!("DICOM data set ended at {cursor} bytes, expected {end}")
        }
        Ok(cursor)
    }

    fn read_element_header(&self, cursor: usize, encoding: Encoding) -> Result<ElementHeader> {
        let tag = self.read_tag(cursor, encoding.byte_order)?;
        if tag.group == ITEM_GROUP {
            bail!("sequence item marker cannot be read as a data element")
        }
        if encoding.implicit_vr {
            let length_start = cursor
                .checked_add(4)
                .context("DICOM element length offset overflow")?;
            let length = self.read_field::<u32>(length_start, encoding.byte_order)?;
            return Ok(ElementHeader {
                tag,
                vr: dictionary::vr(tag, false),
                length,
                bytes: 8,
            });
        }
        let vr_start = cursor
            .checked_add(4)
            .context("DICOM value representation offset overflow")?;
        let vr_bytes = self.read_array::<2>(vr_start)?;
        if !vr_bytes.iter().all(u8::is_ascii_uppercase) {
            bail!("invalid DICOM VR at byte {vr_start}");
        }
        let vr = std::str::from_utf8(&vr_bytes)?.parse().unwrap_or(VR::UN);
        let bytes = if vr.long_length() { 12 } else { 8 };
        let length = if bytes == 8 {
            let length_start = cursor
                .checked_add(6)
                .context("DICOM short length offset overflow")?;
            u32::from(self.read_field::<u16>(length_start, encoding.byte_order)?)
        } else {
            let length_start = cursor
                .checked_add(8)
                .context("DICOM long length offset overflow")?;
            self.read_field::<u32>(length_start, encoding.byte_order)?
        };
        Ok(ElementHeader {
            tag,
            vr,
            length,
            bytes,
        })
    }

    fn next_tag_is_item(&self, cursor: usize) -> bool {
        self.read_tag(cursor, ByteOrder::LittleEndian)
            .is_ok_and(|tag| tag == DicomTag::new(ITEM_GROUP, ITEM_ELEMENT))
    }

    fn read_tag(&self, cursor: usize, order: ByteOrder) -> Result<DicomTag> {
        let group = self.read_field::<u16>(cursor, order)?;
        let element = self.read_field::<u16>(
            cursor
                .checked_add(2)
                .context("DICOM tag element offset overflow")?,
            order,
        )?;
        Ok(DicomTag::new(group, element))
    }

    fn read_field<T: EndianScalar>(&self, cursor: usize, order: ByteOrder) -> Result<T> {
        let end = cursor
            .checked_add(T::BYTE_WIDTH)
            .context("DICOM header offset overflow")?;
        let bytes = self
            .data
            .get(cursor..end)
            .context("truncated DICOM header")?;
        read_integer(bytes, order).context("DICOM header has an invalid byte width")
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

#[cfg(test)]
#[path = "tests_scan.rs"]
mod tests_scan;
