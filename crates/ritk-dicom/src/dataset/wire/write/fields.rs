use std::borrow::Cow;
use std::io::{self, Write};

use anyhow::{bail, Context, Result};
use consus_core::ByteOrder;

use crate::dataset::wire::{charset::CharacterSet, syntax::Encoding};
use crate::dataset::{DataElement, PrimitiveValue, VR};
use crate::DicomTag;

fn word(out: &mut impl Write, order: ByteOrder, value: u16) -> io::Result<()> {
    out.write_all(&match order {
        ByteOrder::LittleEndian => value.to_le_bytes(),
        ByteOrder::BigEndian => value.to_be_bytes(),
    })
}

fn double_word(out: &mut impl Write, order: ByteOrder, value: u32) -> io::Result<()> {
    out.write_all(&match order {
        ByteOrder::LittleEndian => value.to_le_bytes(),
        ByteOrder::BigEndian => value.to_be_bytes(),
    })
}

pub(super) fn attribute_header(
    out: &mut impl Write,
    encoding: Encoding,
    tag: DicomTag,
    vr: VR,
    length: u32,
) -> Result<()> {
    let order = encoding.byte_order;
    word(out, order, tag.group)?;
    word(out, order, tag.element)?;
    if encoding.implicit_vr {
        double_word(out, order, length)?;
    } else {
        out.write_all(vr.as_str().as_bytes())?;
        if vr.long_length() {
            out.write_all(&[0, 0])?;
            double_word(out, order, length)?;
        } else {
            word(
                out,
                order,
                u16::try_from(length).context("DICOM short VR length exceeds 16 bits")?,
            )?;
        }
    }
    Ok(())
}

pub(super) fn item_header(
    out: &mut impl Write,
    order: ByteOrder,
    tag: u16,
    length: u32,
) -> Result<()> {
    word(out, order, 0xfffe)?;
    word(out, order, tag)?;
    double_word(out, order, length)?;
    Ok(())
}

pub(super) enum Payload<'a> {
    Bytes(Cow<'a, [u8]>),
    BigEndianWords { raw: &'a [u8], width: usize },
}

impl Payload<'_> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::BigEndianWords { raw, .. } => raw.len(),
        }
    }

    pub(super) fn write(&self, out: &mut impl Write) -> io::Result<()> {
        let (raw, width) = match self {
            Self::Bytes(bytes) => return out.write_all(bytes),
            Self::BigEndianWords { raw, width } => (*raw, *width),
        };
        // OW, OL, OF, OD, and OV word widths divide the buffer size. Each
        // chunk therefore starts at a word boundary without copying a frame.
        let mut buffer = [0u8; 8192];
        for block in raw.chunks(buffer.len()) {
            for (target, word) in buffer
                .chunks_exact_mut(width)
                .zip(block.chunks_exact(width))
            {
                for (byte, source) in target.iter_mut().zip(word.iter().rev()) {
                    *byte = *source;
                }
            }
            out.write_all(&buffer[..block.len()])?;
        }
        Ok(())
    }
}

pub(super) fn primitive<'a>(
    element: &'a DataElement,
    encoding: Encoding,
    charset: &CharacterSet,
) -> Result<Payload<'a>> {
    let value = element
        .value()
        .primitive()
        .context("expected a DICOM primitive")?;
    let vr = element.vr();
    if vr.is_text() {
        let ascii = CharacterSet::default();
        let selected = if vr.uses_character_set() {
            charset
        } else {
            &ascii
        };
        return match value {
            PrimitiveValue::Str(text) => selected.encode(text.trim_end_matches('\0')),
            PrimitiveValue::Strs(text) => {
                Ok(Cow::Owned(selected.encode(&text.join("\\"))?.into_owned()))
            }
            _ => match value.to_str() {
                Cow::Borrowed(text) => selected.encode(text),
                Cow::Owned(text) => Ok(Cow::Owned(selected.encode(&text)?.into_owned())),
            },
        }
        .map(Payload::Bytes);
    }
    if matches!(vr, VR::SQ) && !matches!(value, PrimitiveValue::Empty) {
        bail!("DICOM SQ value requires a sequence");
    }
    let width = vr.byte_word();
    let order = encoding.byte_order;
    if let PrimitiveValue::Bytes(raw) = value
        && width > 1
    {
        if !raw.len().is_multiple_of(width) {
            bail!("DICOM VR {vr} value length is not a multiple of its word width {width}");
        }
        if order == ByteOrder::BigEndian {
            return Ok(Payload::BigEndianWords { raw, width });
        }
    }
    Ok(Payload::Bytes(value.encoded_bytes(order)))
}

pub(super) struct Counted<W> {
    sink: W,
    pub(super) bytes: usize,
}

impl<W> Counted<W> {
    pub(super) fn new(sink: W) -> Self {
        Self { sink, bytes: 0 }
    }
}

impl<W: Write> Write for Counted<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.sink.write(bytes)?;
        self.bytes += written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sink.flush()
    }
}
