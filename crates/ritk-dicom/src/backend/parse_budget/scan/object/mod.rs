//! Materialization sink for the budgeted scanner.

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use consus_core::{ByteOrder, ParseBudget};

use super::{ElementHeader, Event, Scanner, UNDEFINED_LENGTH};
use crate::dataset::wire::dictionary;
use crate::dataset::{
    DataElement, Dataset, DicomFile, FileMeta, Length, PixelFragments, Sequence, Value,
};
use crate::{tags, CharacterSet};

mod primitive;
use primitive::{decode, numbers};

pub(crate) fn materialize(input: Bytes, budget: &ParseBudget) -> Result<DicomFile> {
    let mut builder = Builder {
        input: input.clone(),
        meta: Dataset::new(),
        levels: vec![Level::default()],
        sequences: Vec::new(),
    };
    Scanner::new(&input, budget, |event| builder.visit(event))?.scan()?;
    let meta = FileMeta::from_dataset(builder.meta)?;
    let dataset = builder
        .levels
        .pop()
        .context("missing root DICOM dataset")?
        .dataset;
    let mut file = DicomFile::new(meta, dataset);
    file.preamble = match input.get(128..132) {
        Some(b"DICM") => Some(Box::new(input[..128].try_into()?)),
        _ => None,
    };
    Ok(file)
}

#[derive(Default)]
struct Level {
    dataset: Dataset,
    charset: CharacterSet,
}

struct Builder {
    input: Bytes,
    meta: Dataset,
    levels: Vec<Level>,
    sequences: Vec<Container>,
}

enum Children {
    Items(Vec<Dataset>),
    Fragments(Vec<Bytes>),
}

struct Container {
    header: ElementHeader,
    children: Children,
}

impl Builder {
    fn level(&mut self) -> Result<&mut Level> {
        self.levels
            .last_mut()
            .context("unexpected DICOM item completion")
    }

    fn visit(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Meta { header, range } => {
                let value = decode(
                    header.vr,
                    self.input.slice(range),
                    ByteOrder::LittleEndian,
                    &CharacterSet::default(),
                )?;
                self.meta
                    .put(DataElement::new(header.tag, header.vr, value));
            }
            Event::Inflated(input) => self.input = input,
            Event::Primitive {
                header,
                range,
                encoding,
                ..
            } => {
                let raw = self.input.slice(range);
                let vr = if encoding.implicit_vr {
                    let signed = self
                        .levels
                        .iter()
                        .rev()
                        .find_map(|level| level.dataset.get(tags::PIXEL_REPRESENTATION))
                        .and_then(|element| element.to_num::<u16>().ok())
                        == Some(1);
                    dictionary::vr(header.tag, signed)
                } else {
                    header.vr
                };
                let level = self.level()?;
                let value = decode(vr, raw, encoding.byte_order, &level.charset)
                    .with_context(|| format!("DICOM element {} VR {vr}", header.tag))?;
                if header.tag == tags::SPECIFIC_CHARACTER_SET {
                    level.charset = CharacterSet::from_declaration(&value.to_str())?;
                }
                level.dataset.put(DataElement::new(header.tag, vr, value));
            }
            Event::SequenceStart { header, pixel } => {
                let children = if pixel {
                    Children::Fragments(Vec::new())
                } else {
                    Children::Items(Vec::new())
                };
                self.sequences.push(Container { header, children });
            }
            Event::SequenceEnd { .. } => {
                let Container { header, children } = self
                    .sequences
                    .pop()
                    .context("unexpected DICOM sequence completion")?;
                let (vr, value) = match children {
                    Children::Items(items) => {
                        let length = if header.length == UNDEFINED_LENGTH {
                            Length::Undefined
                        } else {
                            Length::Defined
                        };
                        (header.vr, Value::Sequence(Sequence::new(items, length)))
                    }
                    Children::Fragments(fragments) => {
                        let mut items = fragments.into_iter();
                        let table = items
                            .next()
                            .context("missing Basic Offset Table in DICOM Pixel Data")?;
                        let offsets = numbers::<u32>(&table, ByteOrder::LittleEndian)?.into_vec();
                        (
                            crate::dataset::VR::OB,
                            Value::PixelSequence(PixelFragments::from_spans(
                                offsets,
                                items.collect(),
                            )),
                        )
                    }
                };
                self.level()?
                    .dataset
                    .put(DataElement::new(header.tag, vr, value));
            }
            Event::ItemStart(style) => {
                let charset = self.level()?.charset.clone();
                let mut dataset = Dataset::new();
                dataset.item_length = style;
                self.levels.push(Level { dataset, charset });
            }
            Event::ItemEnd => {
                let item = self
                    .levels
                    .pop()
                    .context("unexpected DICOM item completion")?;
                let Some(Container {
                    children: Children::Items(items),
                    ..
                }) = self.sequences.last_mut()
                else {
                    bail!("DICOM item does not belong to a dataset sequence");
                };
                items.push(item.dataset);
            }
            Event::Fragment(range) => {
                let Some(Container {
                    children: Children::Fragments(items),
                    ..
                }) = self.sequences.last_mut()
                else {
                    bail!("DICOM fragment does not belong to Pixel Data");
                };
                items.push(self.input.slice(range));
            }
        }
        Ok(())
    }
}
