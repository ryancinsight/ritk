//! Decoding and encoding `Data` element payloads (GIFTI 1.0, sections 4–5).
//!
//! `ASCII` is whitespace-separated decimal text. `Base64Binary` is base64 (RFC
//! 3548) over the values' bytes in the array's `Endian` order.
//! `GZipBase64Binary` is, despite its name, base64 over a *zlib* stream
//! (section 5.0: "compressed using ZLIB"), which is also what nibabel's
//! `gifti/parse_gifti_fast.py` inflates (`zlib.decompress`). Every decoded payload must hold
//! exactly the number of values its shape declares.

use std::io::{Read, Write};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

use crate::{ArrayData, DataEncoding, GiftiError};

/// The `DataType` attribute (section 2.3.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataType {
    UInt8,
    Int32,
    Float32,
}

impl DataType {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "NIFTI_TYPE_UINT8" => Some(Self::UInt8),
            "NIFTI_TYPE_INT32" => Some(Self::Int32),
            "NIFTI_TYPE_FLOAT32" => Some(Self::Float32),
            _ => None,
        }
    }

    const fn width(self) -> usize {
        match self {
            Self::UInt8 => 1,
            Self::Int32 | Self::Float32 => 4,
        }
    }
}

/// The `Endian` attribute (section 2.3.4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Endian {
    Big,
    Little,
}

impl Endian {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "BigEndian" => Some(Self::Big),
            "LittleEndian" => Some(Self::Little),
            _ => None,
        }
    }
}

/// How a payload is read back: the in-document encodings of section 2.3.4.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Payload {
    Ascii,
    Base64,
    ZlibBase64,
}

/// Decode `text` into `count` values of `data_type`.
pub(crate) fn decode(
    text: &str,
    payload: Payload,
    data_type: DataType,
    endian: Option<Endian>,
    count: usize,
    array: usize,
) -> Result<ArrayData, GiftiError> {
    match payload {
        Payload::Ascii => decode_ascii(text, data_type, count, array),
        Payload::Base64 | Payload::ZlibBase64 => {
            let compact: Vec<u8> = text
                .bytes()
                .filter(|byte| !byte.is_ascii_whitespace())
                .collect();
            let decoded = STANDARD
                .decode(compact)
                .map_err(|error| GiftiError::data(array, format!("invalid base64: {error}")))?;
            let expected = count * data_type.width();
            let bytes = if payload == Payload::ZlibBase64 {
                inflate(&decoded, expected, array)?
            } else {
                decoded
            };
            if bytes.len() != expected {
                return Err(GiftiError::data(
                    array,
                    format!("{} bytes decoded, shape needs {expected}", bytes.len()),
                ));
            }
            from_bytes(&bytes, data_type, endian, array)
        }
    }
}

/// Inflate a zlib stream, reading at most one byte past `expected` so that a
/// stream expanding beyond its declared shape is caught without inflating it.
fn inflate(compressed: &[u8], expected: usize, array: usize) -> Result<Vec<u8>, GiftiError> {
    let mut bytes = Vec::new();
    ZlibDecoder::new(compressed)
        .take(
            u64::try_from(expected)
                .unwrap_or(u64::MAX)
                .saturating_add(1),
        )
        .read_to_end(&mut bytes)
        .map_err(|error| GiftiError::data(array, format!("invalid zlib stream: {error}")))?;
    Ok(bytes)
}

fn decode_ascii(
    text: &str,
    data_type: DataType,
    count: usize,
    array: usize,
) -> Result<ArrayData, GiftiError> {
    fn parse<T: std::str::FromStr>(
        text: &str,
        count: usize,
        array: usize,
    ) -> Result<Box<[T]>, GiftiError> {
        let mut values = Vec::with_capacity(count.min(1 << 16));
        for token in text.split_ascii_whitespace() {
            if values.len() == count {
                return Err(GiftiError::data(
                    array,
                    format!("more than the {count} values the shape declares"),
                ));
            }
            values.push(
                token.parse::<T>().map_err(|_| {
                    GiftiError::data(array, format!("value {token:?} does not parse"))
                })?,
            );
        }
        if values.len() != count {
            return Err(GiftiError::data(
                array,
                format!("{} values, shape declares {count}", values.len()),
            ));
        }
        Ok(values.into_boxed_slice())
    }
    Ok(match data_type {
        DataType::UInt8 => ArrayData::UInt8(parse(text, count, array)?),
        DataType::Int32 => ArrayData::Int32(parse(text, count, array)?),
        DataType::Float32 => ArrayData::Float32(parse(text, count, array)?),
    })
}

fn from_bytes(
    bytes: &[u8],
    data_type: DataType,
    endian: Option<Endian>,
    array: usize,
) -> Result<ArrayData, GiftiError> {
    if data_type == DataType::UInt8 {
        return Ok(ArrayData::UInt8(bytes.into()));
    }
    let endian = endian.ok_or_else(|| {
        GiftiError::data(array, "binary data without an Endian attribute".to_owned())
    })?;
    // The caller checked the length is a whole number of four-byte values.
    let words = bytes.chunks_exact(4).map(|chunk| {
        <[u8; 4]>::try_from(chunk).expect("invariant: chunks_exact(4) yields four-byte chunks")
    });
    Ok(match (data_type, endian) {
        (DataType::Int32, Endian::Big) => ArrayData::Int32(words.map(i32::from_be_bytes).collect()),
        (DataType::Int32, Endian::Little) => {
            ArrayData::Int32(words.map(i32::from_le_bytes).collect())
        }
        (DataType::Float32, Endian::Big) => {
            ArrayData::Float32(words.map(f32::from_be_bytes).collect())
        }
        (DataType::Float32, Endian::Little) => {
            ArrayData::Float32(words.map(f32::from_le_bytes).collect())
        }
        (DataType::UInt8, _) => ArrayData::UInt8(bytes.into()),
    })
}

/// Encode `data` as the text of a `Data` element; binary is little-endian.
pub(crate) fn encode(data: &ArrayData, encoding: DataEncoding) -> Result<String, GiftiError> {
    if encoding == DataEncoding::Ascii {
        return Ok(match data {
            ArrayData::UInt8(values) => join(values),
            ArrayData::Int32(values) => join(values),
            ArrayData::Float32(values) => join(values),
        });
    }
    let bytes: Vec<u8> = match data {
        ArrayData::UInt8(values) => values.to_vec(),
        ArrayData::Int32(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        ArrayData::Float32(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    };
    let bytes = if encoding == DataEncoding::GZipBase64Binary {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&bytes)?;
        encoder.finish()?
    } else {
        bytes
    };
    Ok(STANDARD.encode(bytes))
}

/// Space-separated values in Rust's shortest round-tripping decimal form.
fn join<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}
