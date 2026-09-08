//! Byte-level traversal of an Explicit VR Little Endian DICOMDIR record sequence.

use anyhow::{bail, Context, Result};
use dicom::core::{Tag, VR};

const RECORD_SEQUENCE: Tag = Tag(0x0004, 0x1220);
const ITEM_TAG: Tag = Tag(0xFFFE, 0xE000);
const ITEM_DELIMITER: Tag = Tag(0xFFFE, 0xE00D);
const SEQUENCE_DELIMITER: Tag = Tag(0xFFFE, 0xE0DD);
const UNDEFINED_LENGTH: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
struct RawHeader {
    tag: Tag,
    vr: VR,
    length: u32,
    bytes: usize,
}

pub(super) fn directory_record_offsets(data: &[u8]) -> Result<Vec<u32>> {
    let dataset_start = if data.get(128..132) == Some(b"DICM") {
        132
    } else {
        bail!("DICOMDIR preamble and DICM marker are missing")
    };
    let group_length = read_header(data, dataset_start)?;
    if group_length.tag != Tag(0x0002, 0x0000)
        || group_length.vr != VR::UL
        || group_length.length != 4
    {
        bail!("DICOMDIR file meta must begin with (0002,0000) UL length 4");
    }
    let group_length_start = checked_end(dataset_start, group_length.bytes, data.len())?;
    let group_length_value = read_u32(data, group_length_start)?;
    let group_length_end = checked_end(group_length_start, 4, data.len())?;
    let group_end = checked_end(
        group_length_end,
        usize::try_from(group_length_value).context("DICOMDIR meta length does not fit usize")?,
        data.len(),
    )?;

    let mut cursor = group_end;
    while cursor < data.len() {
        let header = read_header(data, cursor)?;
        let value_start = checked_end(cursor, header.bytes, data.len())?;
        if header.tag == RECORD_SEQUENCE {
            if header.vr != VR::SQ {
                bail!("DICOMDIR DirectoryRecordSequence has a non-sequence value");
            }
            return record_sequence_offsets(data, value_start, header.length);
        }
        cursor = skip_element(data, cursor, data.len(), header)?;
    }
    bail!("DICOMDIR DirectoryRecordSequence is missing")
}

fn record_sequence_offsets(data: &[u8], start: usize, length: u32) -> Result<Vec<u32>> {
    let sequence_end = sequence_end(start, length, data.len())?;
    let mut cursor = start;
    let mut offsets = Vec::new();
    loop {
        if cursor == sequence_end {
            if length == UNDEFINED_LENGTH {
                bail!("DICOMDIR DirectoryRecordSequence is missing its delimiter");
            }
            return Ok(offsets);
        }
        let tag = read_tag(data, cursor)?;
        if tag == SEQUENCE_DELIMITER {
            if length != UNDEFINED_LENGTH {
                bail!("defined-length DICOMDIR sequence contains a delimiter");
            }
            let delimiter_length = read_u32(data, checked_end(cursor, 4, data.len())?)?;
            if delimiter_length != 0 {
                bail!("DICOMDIR sequence delimiter length must be zero");
            }
            return Ok(offsets);
        }
        if tag != ITEM_TAG {
            bail!("DICOMDIR DirectoryRecordSequence expects an item at byte {cursor}");
        }
        offsets.push(u32::try_from(cursor).context("DICOMDIR record offset exceeds u32")?);
        let item_length = read_u32(data, checked_end(cursor, 4, data.len())?)?;
        cursor = skip_item(data, cursor, sequence_end, item_length)?;
    }
}

fn skip_item(data: &[u8], cursor: usize, parent_end: usize, length: u32) -> Result<usize> {
    let item_start = checked_end(cursor, 8, parent_end)?;
    if length == UNDEFINED_LENGTH {
        skip_dataset_until_item_delimiter(data, item_start, parent_end)
    } else {
        let item_end = checked_end(
            item_start,
            usize::try_from(length).context("DICOMDIR item length does not fit usize")?,
            parent_end,
        )?;
        skip_dataset(data, item_start, item_end)?;
        Ok(item_end)
    }
}

fn skip_dataset(data: &[u8], mut cursor: usize, end: usize) -> Result<usize> {
    while cursor < end {
        let header = read_header(data, cursor)?;
        cursor = skip_element(data, cursor, end, header)?;
    }
    if cursor != end {
        bail!("DICOMDIR item ended at byte {cursor}, expected {end}");
    }
    Ok(cursor)
}

fn skip_dataset_until_item_delimiter(data: &[u8], mut cursor: usize, end: usize) -> Result<usize> {
    while cursor < end {
        let tag = read_tag(data, cursor)?;
        if tag == ITEM_DELIMITER {
            let delimiter_length = read_u32(data, checked_end(cursor, 4, end)?)?;
            if delimiter_length != 0 {
                bail!("DICOMDIR item delimiter length must be zero");
            }
            return checked_end(cursor, 8, end);
        }
        if tag == SEQUENCE_DELIMITER {
            bail!("DICOMDIR sequence delimiter encountered inside an item");
        }
        let header = read_header(data, cursor)?;
        cursor = skip_element(data, cursor, end, header)?;
    }
    bail!("DICOMDIR undefined-length item is missing its delimiter")
}

fn skip_element(data: &[u8], cursor: usize, parent_end: usize, header: RawHeader) -> Result<usize> {
    let value_start = checked_end(cursor, header.bytes, parent_end)?;
    if header.vr == VR::SQ {
        return skip_sequence(data, value_start, header.length, parent_end);
    }
    if header.length == UNDEFINED_LENGTH {
        bail!(
            "DICOMDIR primitive element {} has undefined length",
            header.tag
        );
    }
    checked_end(
        value_start,
        usize::try_from(header.length).context("DICOMDIR element length does not fit usize")?,
        parent_end,
    )
}

fn skip_sequence(data: &[u8], start: usize, length: u32, parent_end: usize) -> Result<usize> {
    let sequence_end = sequence_end(start, length, parent_end)?;
    let mut cursor = start;
    loop {
        if cursor == sequence_end {
            if length == UNDEFINED_LENGTH {
                bail!("DICOMDIR undefined-length sequence is missing its delimiter");
            }
            return Ok(cursor);
        }
        let tag = read_tag(data, cursor)?;
        if tag == SEQUENCE_DELIMITER {
            if length != UNDEFINED_LENGTH {
                bail!("defined-length DICOMDIR sequence contains a delimiter");
            }
            let delimiter_length = read_u32(data, checked_end(cursor, 4, parent_end)?)?;
            if delimiter_length != 0 {
                bail!("DICOMDIR sequence delimiter length must be zero");
            }
            return checked_end(cursor, 8, parent_end);
        }
        if tag != ITEM_TAG {
            bail!("DICOMDIR sequence expects an item at byte {cursor}");
        }
        let item_length = read_u32(data, checked_end(cursor, 4, sequence_end)?)?;
        cursor = skip_item(data, cursor, sequence_end, item_length)?;
    }
}

fn sequence_end(start: usize, length: u32, parent_end: usize) -> Result<usize> {
    if length == UNDEFINED_LENGTH {
        return Ok(parent_end);
    }
    checked_end(
        start,
        usize::try_from(length).context("DICOMDIR sequence length does not fit usize")?,
        parent_end,
    )
}

fn read_header(data: &[u8], cursor: usize) -> Result<RawHeader> {
    let tag = read_tag(data, cursor)?;
    let vr_bytes = read_array::<2>(data, checked_end(cursor, 4, data.len())?)?;
    let vr = VR::from_binary(vr_bytes)
        .with_context(|| format!("DICOMDIR element {tag} has an invalid value representation"))?;
    let bytes = if uses_short_length(vr) { 8 } else { 12 };
    let length_offset = checked_end(cursor, if bytes == 8 { 6 } else { 8 }, data.len())?;
    let length = if bytes == 8 {
        u32::from(u16::from_le_bytes(read_array::<2>(data, length_offset)?))
    } else {
        read_u32(data, length_offset)?
    };
    checked_end(cursor, bytes, data.len())?;
    Ok(RawHeader {
        tag,
        vr,
        length,
        bytes,
    })
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

fn read_tag(data: &[u8], cursor: usize) -> Result<Tag> {
    let group = u16::from_le_bytes(read_array::<2>(data, cursor)?);
    let element = u16::from_le_bytes(read_array::<2>(data, checked_end(cursor, 2, data.len())?)?);
    Ok(Tag(group, element))
}

fn read_u32(data: &[u8], cursor: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(read_array::<4>(data, cursor)?))
}

fn read_array<const N: usize>(data: &[u8], cursor: usize) -> Result<[u8; N]> {
    let end = checked_end(cursor, N, data.len())?;
    data.get(cursor..end)
        .context("truncated DICOMDIR byte span")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("DICOMDIR byte span has an invalid width"))
}

fn checked_end(start: usize, length: usize, parent_end: usize) -> Result<usize> {
    let end = start
        .checked_add(length)
        .context("DICOMDIR byte offset overflow")?;
    if end > parent_end {
        bail!("DICOMDIR declared span exceeds its containing value");
    }
    Ok(end)
}
