//! Parsing a GIFTI document (GIFTI 1.0, sections 2 and 8.1.2).
//!
//! A streaming pass over the XML events builds the model directly, as the
//! specification recommends for large files (section 10.0). Element nesting is
//! checked against the DTD (section 8.1.2); elements the DTD does not define
//! are skipped with their content, and nesting depth is bounded so a
//! pathological document cannot grow the element stack without limit.

use std::io::Read;

use quick_xml::Reader;
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesStart, Event};

mod element;
mod fields;

use element::{Element, MAX_DEPTH};
use fields::{Attributes, parse_label, parse_matrix};

use crate::decode::{self, DataType, Endian, Payload};
use crate::model::{MAX_DIMENSIONS, element_count};
use crate::{
    CoordinateTransform, DataArray, GiftiError, GiftiImage, GiftiLabel, IndexingOrder, Intent,
    MetaData,
};

/// A `DataArray` between its start tag and its end tag.
struct OpenArray {
    intent: Intent,
    data_type: DataType,
    order: IndexingOrder,
    dims: Vec<usize>,
    count: usize,
    payload: Payload,
    endian: Option<Endian>,
    metadata: MetaData,
    transforms: Vec<CoordinateTransform>,
    data: Option<crate::ArrayData>,
}

#[derive(Default)]
struct OpenTransform {
    data_space: Option<String>,
    transformed_space: Option<String>,
    matrix: Option<[[f64; 4]; 4]>,
}

#[derive(Default)]
struct Parser {
    path: Vec<Element>,
    text: String,
    declared_arrays: Option<usize>,
    closed: bool,
    metadata: MetaData,
    labels: Vec<GiftiLabel>,
    arrays: Vec<DataArray>,
    array: Option<OpenArray>,
    entry: (Option<String>, Option<String>),
    label: Option<(u32, Option<[f32; 4]>)>,
    transform: Option<OpenTransform>,
}

/// Read a GIFTI document.
pub(crate) fn read(mut reader: impl Read) -> Result<GiftiImage, GiftiError> {
    let mut document = String::new();
    reader.read_to_string(&mut document)?;
    let mut xml = Reader::from_str(&document);
    let mut parser = Parser::default();
    loop {
        let event = xml.read_event().map_err(|error| GiftiError::Xml {
            position: xml.error_position(),
            reason: error.to_string(),
        })?;
        match event {
            Event::Start(start) => parser.open(&start)?,
            Event::Empty(start) => {
                parser.open(&start)?;
                parser.close()?;
            }
            Event::End(_) => parser.close()?,
            Event::Text(text) => {
                let text = text
                    .xml10_content()
                    .map_err(|error| xml_error(&xml, &error))?;
                parser.characters(&text);
            }
            Event::CData(data) => {
                let data = data.decode().map_err(|error| xml_error(&xml, &error))?;
                parser.characters(&data);
            }
            Event::GeneralRef(reference) => {
                let resolved = match reference
                    .resolve_char_ref()
                    .map_err(|error| xml_error(&xml, &error))?
                {
                    Some(character) => character.to_string(),
                    None => {
                        let name = reference
                            .decode()
                            .map_err(|error| xml_error(&xml, &error))?;
                        resolve_predefined_entity(&name)
                            .ok_or_else(|| GiftiError::Xml {
                                position: xml.buffer_position(),
                                reason: format!("undefined entity &{name};"),
                            })?
                            .to_owned()
                    }
                };
                parser.characters(&resolved);
            }
            Event::Eof => break,
            Event::Comment(_) | Event::Decl(_) | Event::PI(_) | Event::DocType(_) => {}
        }
    }
    parser.finish()
}

fn xml_error(xml: &Reader<&[u8]>, error: &impl std::fmt::Display) -> GiftiError {
    GiftiError::Xml {
        position: xml.buffer_position(),
        reason: error.to_string(),
    }
}

impl Parser {
    fn open(&mut self, start: &BytesStart<'_>) -> Result<(), GiftiError> {
        let element = Element::from_name(start.local_name().as_ref());
        let parent = self.path.last().copied();
        if self.closed {
            return Err(GiftiError::structure(
                "GIFTI",
                "content after the root element",
            ));
        }
        if parent == Some(Element::Unknown) {
            return self.push(Element::Unknown);
        }
        if !element.allowed_in(parent) {
            return Err(GiftiError::structure(
                element.tag(),
                format!(
                    "not allowed inside {}",
                    parent.map_or("the document", Element::tag)
                ),
            ));
        }
        let attributes = Attributes::of(start, element)?;
        match element {
            Element::Gifti => self.open_root(&attributes)?,
            Element::DataArray => self.open_array(&attributes)?,
            Element::Label => self.label = Some(parse_label(&attributes)?),
            Element::Md => self.entry = (None, None),
            Element::Transform => self.transform = Some(OpenTransform::default()),
            _ => {}
        }
        if element.captures_text() {
            self.text.clear();
        }
        self.push(element)
    }

    fn push(&mut self, element: Element) -> Result<(), GiftiError> {
        if self.path.len() == MAX_DEPTH {
            return Err(GiftiError::structure(
                element.tag(),
                format!("nesting deeper than {MAX_DEPTH}"),
            ));
        }
        self.path.push(element);
        Ok(())
    }

    fn characters(&mut self, text: &str) {
        if self
            .path
            .last()
            .is_some_and(|element| element.captures_text())
        {
            self.text.push_str(text);
        }
    }

    fn open_root(&mut self, attributes: &Attributes) -> Result<(), GiftiError> {
        let version = attributes.required("GIFTI", "Version")?;
        if version.parse::<f64>().ok() != Some(1.0) {
            return Err(GiftiError::Unsupported(format!(
                "GIFTI version {version:?}"
            )));
        }
        let count = attributes.required("GIFTI", "NumberOfDataArrays")?;
        self.declared_arrays = Some(count.parse().map_err(|_| {
            GiftiError::structure("GIFTI", format!("NumberOfDataArrays {count:?}"))
        })?);
        Ok(())
    }

    fn open_array(&mut self, attributes: &Attributes) -> Result<(), GiftiError> {
        const TAG: &str = "DataArray";
        let data_type = attributes.required(TAG, "DataType")?;
        let data_type = DataType::from_name(data_type)
            .ok_or_else(|| GiftiError::structure(TAG, format!("DataType {data_type:?}")))?;
        let order = match attributes.required(TAG, "ArrayIndexingOrder")? {
            "RowMajorOrder" => IndexingOrder::RowMajor,
            "ColumnMajorOrder" => IndexingOrder::ColumnMajor,
            other => {
                return Err(GiftiError::structure(
                    TAG,
                    format!("ArrayIndexingOrder {other:?}"),
                ));
            }
        };
        let payload = match attributes.required(TAG, "Encoding")? {
            "ASCII" => Payload::Ascii,
            "Base64Binary" => Payload::Base64,
            "GZipBase64Binary" => Payload::ZlibBase64,
            "ExternalFileBinary" => {
                return Err(GiftiError::Unsupported(
                    "ExternalFileBinary data needs the path of the XML file".to_owned(),
                ));
            }
            other => return Err(GiftiError::structure(TAG, format!("Encoding {other:?}"))),
        };
        let endian = attributes
            .optional("Endian")
            .map(|name| {
                Endian::from_name(name)
                    .ok_or_else(|| GiftiError::structure(TAG, format!("Endian {name:?}")))
            })
            .transpose()?;
        let rank_text = attributes.required(TAG, "Dimensionality")?;
        let rank = rank_text
            .parse::<usize>()
            .ok()
            .filter(|rank| (1..=MAX_DIMENSIONS).contains(rank))
            .ok_or_else(|| GiftiError::structure(TAG, format!("Dimensionality {rank_text:?}")))?;
        let mut dims = Vec::with_capacity(rank);
        for axis in 0..MAX_DIMENSIONS {
            let name = format!("Dim{axis}");
            match (attributes.optional(&name), axis < rank) {
                (Some(extent), true) => dims.push(
                    extent
                        .parse::<usize>()
                        .map_err(|_| GiftiError::structure(TAG, format!("{name} {extent:?}")))?,
                ),
                (None, true) => return Err(GiftiError::structure(TAG, format!("missing {name}"))),
                (Some(_), false) => {
                    return Err(GiftiError::structure(
                        TAG,
                        format!("{name} beyond Dimensionality {rank}"),
                    ));
                }
                (None, false) => {}
            }
        }
        let count = element_count(&dims)?;
        self.array = Some(OpenArray {
            intent: Intent::from_name(attributes.required(TAG, "Intent")?),
            data_type,
            order,
            dims,
            count,
            payload,
            endian,
            metadata: MetaData::default(),
            transforms: Vec::new(),
            data: None,
        });
        Ok(())
    }

    fn close(&mut self) -> Result<(), GiftiError> {
        let element = self
            .path
            .pop()
            .ok_or_else(|| GiftiError::structure("GIFTI", "end tag without a start tag"))?;
        let text = std::mem::take(&mut self.text);
        match element {
            Element::Gifti => self.closed = true,
            Element::Name => self.entry.0 = Some(text.trim().to_owned()),
            Element::Value => self.entry.1 = Some(text),
            Element::Md => {
                let (Some(name), Some(value)) = std::mem::take(&mut self.entry) else {
                    return Err(GiftiError::structure("MD", "needs one Name and one Value"));
                };
                match self.array.as_mut() {
                    Some(array) => array.metadata.push(name, value),
                    None => self.metadata.push(name, value),
                }
            }
            Element::Label => {
                let (key, rgba) = self
                    .label
                    .take()
                    .ok_or_else(|| GiftiError::structure("Label", "unopened"))?;
                self.labels
                    .push(GiftiLabel::new(key, text.trim().to_owned(), rgba)?);
            }
            Element::DataSpace | Element::TransformedSpace | Element::MatrixData => {
                self.close_transform_part(element, &text)?;
            }
            Element::Transform => {
                let open = self.transform.take().ok_or_else(|| {
                    GiftiError::structure("CoordinateSystemTransformMatrix", "unopened")
                })?;
                let (Some(data_space), Some(transformed_space), Some(matrix)) =
                    (open.data_space, open.transformed_space, open.matrix)
                else {
                    return Err(GiftiError::structure(
                        "CoordinateSystemTransformMatrix",
                        "needs DataSpace, TransformedSpace, and MatrixData",
                    ));
                };
                self.open_array_mut()?.transforms.push(CoordinateTransform {
                    data_space,
                    transformed_space,
                    matrix,
                });
            }
            Element::Data => {
                let index = self.arrays.len();
                let array = self.open_array_mut()?;
                if array.data.is_some() {
                    return Err(GiftiError::structure(
                        "DataArray",
                        "more than one Data element",
                    ));
                }
                array.data = Some(decode::decode(
                    &text,
                    array.payload,
                    array.data_type,
                    array.endian,
                    array.count,
                    index,
                )?);
            }
            Element::DataArray => self.close_array()?,
            Element::MetaData | Element::LabelTable | Element::Unknown => {}
        }
        Ok(())
    }

    fn close_transform_part(&mut self, element: Element, text: &str) -> Result<(), GiftiError> {
        let transform = self
            .transform
            .as_mut()
            .ok_or_else(|| GiftiError::structure(element.tag(), "outside a transform"))?;
        match element {
            Element::DataSpace => transform.data_space = Some(text.trim().to_owned()),
            Element::TransformedSpace => {
                transform.transformed_space = Some(text.trim().to_owned());
            }
            _ => transform.matrix = Some(parse_matrix(text)?),
        }
        Ok(())
    }

    fn open_array_mut(&mut self) -> Result<&mut OpenArray, GiftiError> {
        self.array
            .as_mut()
            .ok_or_else(|| GiftiError::structure("DataArray", "element outside a data array"))
    }

    fn close_array(&mut self) -> Result<(), GiftiError> {
        let index = self.arrays.len();
        let open = self
            .array
            .take()
            .ok_or_else(|| GiftiError::structure("DataArray", "unopened"))?;
        let data = open
            .data
            .ok_or_else(|| GiftiError::data(index, "no Data element"))?;
        let array = DataArray::with_order(open.intent, open.dims, open.order, data)?
            .with_metadata(open.metadata)
            .with_transforms(open.transforms);
        self.arrays.push(array);
        Ok(())
    }

    fn finish(self) -> Result<GiftiImage, GiftiError> {
        if !self.closed || !self.path.is_empty() {
            return Err(GiftiError::structure(
                "GIFTI",
                "document ends before </GIFTI>",
            ));
        }
        if self.declared_arrays != Some(self.arrays.len()) {
            return Err(GiftiError::structure(
                "GIFTI",
                format!(
                    "NumberOfDataArrays {:?} but {} DataArray elements",
                    self.declared_arrays,
                    self.arrays.len()
                ),
            ));
        }
        GiftiImage::new(self.metadata, self.labels, self.arrays)
    }
}
