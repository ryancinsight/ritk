use anyhow::{bail, Context, Result};

use super::{
    DicomTag, ElementHeader, Encoding, Event, Length, Scanner, ITEM_DELIMITER_ELEMENT,
    ITEM_ELEMENT, ITEM_GROUP, PIXEL_DATA, SEQUENCE_DELIMITER_ELEMENT, UNDEFINED_LENGTH, VR,
};

impl<Visit: FnMut(Event) -> Result<()>> Scanner<'_, Visit> {
    pub(super) fn scan_dataset_until_item_delimiter(
        &mut self,
        start: usize,
        end: usize,
        encoding: Encoding,
        depth: u16,
    ) -> Result<usize> {
        let mut cursor = start;
        loop {
            let tag = self.read_tag(cursor, encoding.byte_order)?;
            if tag == DicomTag::new(ITEM_GROUP, ITEM_DELIMITER_ELEMENT) {
                let length_start = cursor
                    .checked_add(4)
                    .context("DICOM item delimiter length offset overflow")?;
                let length = self.read_field::<u32>(length_start, encoding.byte_order)?;
                if length != 0 {
                    bail!("DICOM item delimiter length must be zero")
                }
                self.account_header(8, "DICOM item delimiter")?;
                self.count_element("DICOM item delimiter")?;
                return self.span_end(cursor, 8, end);
            }
            if tag == DicomTag::new(ITEM_GROUP, SEQUENCE_DELIMITER_ELEMENT) {
                bail!("DICOM sequence delimiter encountered inside an item")
            }
            let header = self.read_element_header(cursor, encoding)?;
            if header.tag.group == ITEM_GROUP {
                bail!("unexpected item marker at byte {cursor}")
            }
            cursor = self.scan_element(cursor, end, encoding, depth, header)?;
        }
    }

    pub(super) fn scan_element(
        &mut self,
        cursor: usize,
        parent_end: usize,
        encoding: Encoding,
        depth: u16,
        header: ElementHeader,
    ) -> Result<usize> {
        self.account_element(header)?;
        let value_start = cursor
            .checked_add(header.bytes)
            .context("DICOM value offset overflow")?;
        let pixel = header.tag == PIXEL_DATA && header.length == UNDEFINED_LENGTH;
        let unknown_sequence = header.vr == VR::UN
            && header.length == UNDEFINED_LENGTH
            && self.next_tag_is_item(value_start);
        let value_end = if header.vr == VR::SQ || pixel || unknown_sequence {
            (self.visit)(Event::SequenceStart { header, pixel })?;
            let inner = if unknown_sequence && !pixel {
                Encoding::IMPLICIT_LE
            } else {
                encoding
            };
            let value_end =
                self.scan_sequence(value_start, parent_end, header.length, inner, depth, pixel)?;
            (self.visit)(Event::SequenceEnd {
                tag: header.tag,
                range: value_start..value_end,
                depth,
            })?;
            value_end
        } else {
            if header.length == UNDEFINED_LENGTH {
                bail!(
                    "DICOM element {} has undefined length without a sequence or encapsulated pixel value",
                    header.tag
                );
            }
            let end = self.span_end(value_start, header.length, parent_end)?;
            self.account_value(
                usize::try_from(header.length).context("DICOM value length does not fit usize")?,
                "DICOM element value",
            )?;
            (self.visit)(Event::Primitive {
                header,
                range: value_start..end,
                encoding,
                depth,
            })?;
            end
        };
        Ok(value_end)
    }

    pub(super) fn scan_sequence(
        &mut self,
        start: usize,
        parent_end: usize,
        length: u32,
        encoding: Encoding,
        depth: u16,
        pixel_data: bool,
    ) -> Result<usize> {
        let nested_depth = self.enter_depth(depth)?;
        let sequence_end = if length == UNDEFINED_LENGTH {
            parent_end
        } else {
            self.span_end(start, length, parent_end)?
        };
        if length != UNDEFINED_LENGTH && start == sequence_end {
            return Ok(sequence_end);
        }
        let mut cursor = start;
        loop {
            let tag = self.read_tag(cursor, encoding.byte_order)?;
            if tag == DicomTag::new(ITEM_GROUP, SEQUENCE_DELIMITER_ELEMENT) {
                if length != UNDEFINED_LENGTH {
                    bail!("defined-length DICOM sequence contains a delimiter")
                }
                let length_start = cursor
                    .checked_add(4)
                    .context("DICOM sequence delimiter length offset overflow")?;
                let delimiter_length = self.read_field::<u32>(length_start, encoding.byte_order)?;
                if delimiter_length != 0 {
                    bail!("DICOM sequence delimiter length must be zero")
                }
                self.account_header(8, "DICOM sequence delimiter")?;
                self.count_element("DICOM sequence delimiter")?;
                return self.span_end(cursor, 8, parent_end);
            }
            if tag != DicomTag::new(ITEM_GROUP, ITEM_ELEMENT) {
                bail!("DICOM sequence expects an item at byte {cursor}")
            }
            let length_start = cursor
                .checked_add(4)
                .context("DICOM sequence item length offset overflow")?;
            let item_length = self.read_field::<u32>(length_start, encoding.byte_order)?;
            self.account_header(8, "DICOM sequence item")?;
            self.count_element("DICOM sequence item")?;
            let item_start = self.span_end(cursor, 8, sequence_end)?;
            cursor = if pixel_data {
                if item_length == UNDEFINED_LENGTH {
                    bail!("encapsulated DICOM pixel fragments require defined lengths")
                }
                let item_end = self.span_end(item_start, item_length, sequence_end)?;
                self.account_value(
                    usize::try_from(item_length)
                        .context("DICOM fragment length does not fit usize")?,
                    "DICOM encapsulated fragment",
                )?;
                (self.visit)(Event::Fragment(item_start..item_end))?;
                item_end
            } else {
                let style = if item_length == UNDEFINED_LENGTH {
                    Length::Undefined
                } else {
                    Length::Defined
                };
                (self.visit)(Event::ItemStart(style))?;
                let after = if style == Length::Undefined {
                    self.scan_dataset_until_item_delimiter(
                        item_start,
                        sequence_end,
                        encoding,
                        nested_depth,
                    )?
                } else {
                    let item_end = self.span_end(item_start, item_length, sequence_end)?;
                    self.scan_dataset(item_start, item_end, encoding, nested_depth)?
                };
                (self.visit)(Event::ItemEnd)?;
                after
            };
            if length != UNDEFINED_LENGTH && cursor == sequence_end {
                return Ok(cursor);
            }
            if cursor >= sequence_end {
                if length == UNDEFINED_LENGTH {
                    bail!("undefined-length DICOM sequence is missing its delimiter")
                }
                return Ok(cursor);
            }
        }
    }
}
