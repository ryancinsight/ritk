use std::collections::VecDeque;
use std::io::Write;

use anyhow::{bail, Context, Result};
use consus_core::ByteOrder;

use crate::dataset::wire::{charset::CharacterSet, syntax::Encoding};
use crate::dataset::{Dataset, Length, PixelFragments, Value, VR};
use crate::{DicomTag, ParseBudget, TransferSyntaxKind};

use super::fields::{self, attribute_header, item_header, primitive};

const UNDEFINED: u32 = u32::MAX;
const ITEM: u16 = 0xe000;
const ITEM_END: u16 = 0xe00d;
const SEQUENCE_END: u16 = 0xe0dd;
const PIXEL_DATA: DicomTag = DicomTag::new(0x7fe0, 0x0010);

pub(crate) fn dataset(file: &Dataset, out: impl Write, ts: &TransferSyntaxKind) -> Result<()> {
    let (encoding, mut lengths) = plan_dataset(file, ts)?;
    encoded_dataset(file, out, ts, encoding, &mut lengths)
}

pub(super) fn plan_dataset(
    dataset: &Dataset,
    syntax: &TransferSyntaxKind,
) -> Result<(Encoding, VecDeque<u32>)> {
    let encoding = Encoding::for_syntax(syntax)?;
    if let Some(pixel) = dataset.get(PIXEL_DATA)
        && matches!(pixel.value(), Value::PixelSequence(_)) != syntax.is_encapsulated()
    {
        bail!(
            "DICOM Pixel Data representation disagrees with transfer syntax {}",
            syntax.uid()
        );
    }
    let mut lengths = VecDeque::new();
    measure(dataset, encoding, &CharacterSet::default(), &mut lengths, 0)?;
    Ok((encoding, lengths))
}

pub(super) fn encoded_dataset(
    dataset: &Dataset,
    mut out: impl Write,
    ts: &TransferSyntaxKind,
    encoding: Encoding,
    lengths: &mut VecDeque<u32>,
) -> Result<()> {
    let charset = CharacterSet::default();
    if *ts == TransferSyntaxKind::DeflatedExplicitVrLittleEndian {
        let destination = fields::Counted::new(&mut out);
        let mut deflater =
            flate2::write::DeflateEncoder::new(destination, flate2::Compression::default());
        emit(dataset, encoding, &charset, lengths, &mut deflater)?;
        let destination = deflater.finish()?;
        if !destination.bytes.is_multiple_of(2) {
            out.write_all(&[0])?;
        }
    } else {
        emit(dataset, encoding, &charset, lengths, &mut out)?;
    }
    debug_assert!(lengths.is_empty());
    Ok(())
}

pub(super) fn defined_length(bytes: usize) -> Result<u32> {
    let size = u32::try_from(bytes).context("DICOM defined length exceeds 32 bits")?;
    if size == UNDEFINED {
        bail!("DICOM defined length equals the reserved undefined length");
    }
    Ok(size)
}

fn rounded(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(bytes % 2)
        .context("DICOM padding length overflows")
}

fn sum(count: usize, bytes: usize) -> Result<usize> {
    count
        .checked_add(bytes)
        .context("DICOM encoded length overflows")
}

fn charset(dataset: &Dataset, inherited: &CharacterSet) -> Result<CharacterSet> {
    dataset
        .get(crate::tags::SPECIFIC_CHARACTER_SET)
        .map_or_else(
            || Ok(inherited.clone()),
            |element| CharacterSet::from_declaration(&element.to_str()?),
        )
}

fn sequence_encoding(encoding: Encoding, vr: VR) -> Result<Encoding> {
    match vr {
        VR::SQ => Ok(encoding),
        VR::UN => Ok(Encoding::IMPLICIT_LE),
        _ => bail!("DICOM VR {vr} cannot encode a nested dataset"),
    }
}

fn measure(
    dataset: &Dataset,
    encoding: Encoding,
    inherited: &CharacterSet,
    lengths: &mut VecDeque<u32>,
    depth: u16,
) -> Result<usize> {
    let charset = charset(dataset, inherited)?;
    let mut total = 0;
    for element in dataset {
        let vr = element.vr();
        let header = if encoding.implicit_vr || !vr.long_length() {
            8
        } else {
            12
        };
        let size = match element.value() {
            Value::Primitive(_) => {
                let length = rounded(primitive(element, encoding, &charset)?.len())?;
                let max = if header == 8 && !encoding.implicit_vr {
                    u16::MAX as usize
                } else {
                    (UNDEFINED - 1) as usize
                };
                if length > max {
                    bail!(
                        "DICOM element {} VR {vr} value length {length} exceeds {max}",
                        element.tag()
                    );
                }
                length
            }
            Value::Sequence(sequence) => {
                let next_depth = ParseBudget::DEFAULT
                    .descend(depth, "DICOM writer sequence")
                    .map_err(anyhow::Error::from)?;
                let inner = sequence_encoding(encoding, vr)?;
                let mark = reserve(lengths, sequence.length);
                let mut length = 0;
                for item in sequence.items() {
                    let item_mark = reserve(lengths, item.item_length);
                    let item_body = measure(item, inner, &charset, lengths, next_depth)?;
                    fill(lengths, item_mark, item_body)?;
                    length = sum(sum(length, 8)?, item_body)?;
                    if item.item_length == Length::Undefined {
                        length = sum(length, 8)?;
                    }
                }
                fill(lengths, mark, length)?;
                sum(
                    length,
                    if sequence.length == Length::Undefined {
                        8
                    } else {
                        0
                    },
                )?
            }
            Value::PixelSequence(pixel) => {
                if element.tag() != PIXEL_DATA
                    || vr != VR::OB
                    || encoding.implicit_vr
                    || encoding.byte_order == ByteOrder::BigEndian
                {
                    bail!("encapsulated DICOM Pixel Data requires OB and explicit little-endian framing");
                }
                let offsets_size = pixel
                    .offsets
                    .len()
                    .checked_mul(4)
                    .context("DICOM pixel offset table length overflows")?;
                defined_length(offsets_size)?;
                let mut length = sum(16, offsets_size)?;
                for fragment in pixel.fragments() {
                    let size = rounded(fragment.len())?;
                    defined_length(size)?;
                    length = sum(sum(length, size)?, 8)?;
                }
                length
            }
        };
        total = sum(sum(total, header)?, size)?;
    }
    Ok(total)
}

fn reserve(lengths: &mut VecDeque<u32>, length: Length) -> Option<usize> {
    (length == Length::Defined).then(|| {
        let index = lengths.len();
        lengths.push_back(0);
        index
    })
}

fn fill(lengths: &mut VecDeque<u32>, index: Option<usize>, count: usize) -> Result<()> {
    if let Some(index) = index {
        lengths[index] = defined_length(count)?;
    }
    Ok(())
}

fn planned_length(lengths: &mut VecDeque<u32>, style: Length) -> Result<u32> {
    match style {
        Length::Defined => lengths
            .pop_front()
            .context("DICOM writer lost a defined container length"),
        Length::Undefined => Ok(UNDEFINED),
    }
}

pub(super) fn emit(
    dataset: &Dataset,
    encoding: Encoding,
    inherited: &CharacterSet,
    lengths: &mut VecDeque<u32>,
    out: &mut impl Write,
) -> Result<()> {
    let charset = charset(dataset, inherited)?;
    for element in dataset {
        let tag = element.tag();
        let vr = element.vr();
        match element.value() {
            Value::Primitive(_) => {
                let payload = primitive(element, encoding, &charset)?;
                attribute_header(
                    out,
                    encoding,
                    tag,
                    vr,
                    defined_length(rounded(payload.len())?)?,
                )?;
                payload.write(out)?;
                if !payload.len().is_multiple_of(2) {
                    out.write_all(&[vr.padding()])?;
                }
            }
            Value::Sequence(sequence) => {
                attribute_header(
                    out,
                    encoding,
                    tag,
                    vr,
                    planned_length(lengths, sequence.length)?,
                )?;
                let inner = sequence_encoding(encoding, vr)?;
                for item in sequence.items() {
                    item_header(
                        out,
                        inner.byte_order,
                        ITEM,
                        planned_length(lengths, item.item_length)?,
                    )?;
                    emit(item, inner, &charset, lengths, out)?;
                    if item.item_length == Length::Undefined {
                        item_header(out, inner.byte_order, ITEM_END, 0)?;
                    }
                }
                if sequence.length == Length::Undefined {
                    item_header(out, inner.byte_order, SEQUENCE_END, 0)?;
                }
            }
            Value::PixelSequence(pixel) => {
                attribute_header(out, encoding, tag, vr, UNDEFINED)?;
                emit_pixels(pixel, out)?;
            }
        }
    }
    Ok(())
}

fn emit_pixels(pixel: &PixelFragments, out: &mut impl Write) -> Result<()> {
    let order = ByteOrder::LittleEndian;
    item_header(out, order, ITEM, defined_length(pixel.offsets.len() * 4)?)?;
    for offset in &pixel.offsets {
        out.write_all(&offset.to_le_bytes())?;
    }
    for fragment in pixel.fragments() {
        item_header(out, order, ITEM, defined_length(rounded(fragment.len())?)?)?;
        out.write_all(fragment)?;
        if !fragment.len().is_multiple_of(2) {
            out.write_all(&[0])?;
        }
    }
    item_header(out, order, SEQUENCE_END, 0)
}
