use ritk_codecs::{parse_usize_vec, ByteOrder, SampleType};
use ritk_image_io::{ImageReadBudget, ImageReadResource};
use std::fs::File;
use std::io::{BufReader, Seek};
use std::path::{Path, PathBuf};

use crate::reader::header::{parse_nrrd_header_from_reader, NrrdHeader};
use crate::reader::stored::NrrdStoredReadError;

use super::super::super::decode::{element_type_spec, sample_type};
use super::super::NrrdReadPurpose;
use super::NrrdEncoding;
use super::{input, source};

/// Header and payload facts shared by image and document reads.
pub(in crate::reader) struct NrrdPayloadPlan {
    path: PathBuf,
    reader: BufReader<File>,
    header_data_start: u64,
    header: NrrdHeader,
    dimension: usize,
    sizes: Vec<usize>,
    element_type: String,
    element_size: usize,
    sample_type: SampleType,
    encoding: NrrdEncoding,
    read_purpose: NrrdReadPurpose,
    line_skip: i32,
    byte_skip: i32,
}

impl NrrdPayloadPlan {
    pub(in crate::reader) fn open<P: AsRef<Path>>(
        path: P,
        read_purpose: NrrdReadPurpose,
    ) -> Result<Self, NrrdStoredReadError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| NrrdStoredReadError::OpenHeader {
            path: path.to_path_buf(),
            source,
        })?;
        let mut reader = BufReader::new(file);
        let header = parse_nrrd_header_from_reader(&mut reader)
            .map_err(|source| NrrdStoredReadError::HeaderParse { source })?;
        let header_data_start = reader
            .stream_position()
            .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;
        let headers = &header.fields;

        let element_type = headers
            .get("type")
            .ok_or(NrrdStoredReadError::MissingHeaderField { field: "type" })?
            .clone();
        let dimension_text = headers
            .get("dimension")
            .ok_or(NrrdStoredReadError::MissingHeaderField { field: "dimension" })?;
        let dimension = dimension_text.parse::<usize>().map_err(|source| {
            NrrdStoredReadError::InvalidDimension {
                value: dimension_text.clone(),
                source,
            }
        })?;
        if !(2..=4).contains(&dimension) {
            return Err(NrrdStoredReadError::UnsupportedDimension { dimension });
        }

        let sizes_text = headers
            .get("sizes")
            .ok_or(NrrdStoredReadError::MissingHeaderField { field: "sizes" })?;
        let sizes = parse_usize_vec(sizes_text, "sizes", dimension).map_err(|source| {
            NrrdStoredReadError::InvalidSizes {
                value: sizes_text.clone(),
                source,
            }
        })?;
        if let Some(axis) = sizes.iter().position(|size| *size == 0) {
            return Err(NrrdStoredReadError::EmptyAxis { axis });
        }
        let encoding_text = headers
            .get("encoding")
            .ok_or(NrrdStoredReadError::MissingHeaderField { field: "encoding" })?;
        let encoding = match encoding_text.trim().to_ascii_lowercase().as_str() {
            "raw" => NrrdEncoding::Raw,
            "ascii" | "text" | "txt" => NrrdEncoding::Ascii,
            "gzip" | "gz" => NrrdEncoding::Gzip,
            other => {
                return Err(NrrdStoredReadError::UnsupportedEncoding {
                    encoding: other.to_owned(),
                });
            }
        };
        let (element_size, _, _) = element_type_spec(&element_type).map_err(|_| {
            NrrdStoredReadError::UnsupportedElementType {
                element_type: element_type.clone(),
            }
        })?;
        let sample_type = sample_type(&element_type).map_err(|_| {
            NrrdStoredReadError::UnsupportedElementType {
                element_type: element_type.clone(),
            }
        })?;
        let line_skip = input::parse_line_skip(headers)?;
        let byte_skip = input::parse_byte_skip(headers)?;
        if byte_skip == -1 && encoding != NrrdEncoding::Raw {
            return Err(NrrdStoredReadError::InvalidByteSkip {
                value: byte_skip,
                reason: "-1 is only defined for raw encoding",
            });
        }

        Ok(Self {
            path: path.to_path_buf(),
            reader,
            header_data_start,
            header,
            dimension,
            sizes,
            element_type,
            element_size,
            sample_type,
            encoding,
            read_purpose,
            line_skip,
            byte_skip,
        })
    }

    pub(super) const fn header(&self) -> &NrrdHeader {
        &self.header
    }

    pub(super) const fn dimension(&self) -> usize {
        self.dimension
    }

    pub(super) fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub(super) fn element_type(&self) -> &str {
        &self.element_type
    }

    pub(in crate::reader) fn byte_order(&self) -> Result<ByteOrder, NrrdStoredReadError> {
        parse_byte_order(&self.header.fields, self.encoding, self.element_size)
    }

    pub(in crate::reader) fn read_payload(
        &mut self,
        budget: ImageReadBudget,
    ) -> Result<Vec<u8>, NrrdStoredReadError> {
        let sample_count = self.sample_count()?;
        let payload_bytes = sample_count.checked_mul(self.element_size).ok_or(
            NrrdStoredReadError::PayloadByteCountOverflow {
                voxel_count: sample_count,
                sample_width: self.element_size,
            },
        )?;
        self.check_payload_budget(budget, sample_count, payload_bytes)?;

        let Some(data_file) = self.header.fields.get("data file").map(String::as_str) else {
            return self.read_inline_payload(budget, payload_bytes, sample_count);
        };
        if data_file.eq_ignore_ascii_case("internal") {
            return self.read_inline_payload(budget, payload_bytes, sample_count);
        }
        let raw_path = source::resolve_detached_data_path(&self.path, data_file)?;
        let file =
            File::open(&raw_path).map_err(|source| NrrdStoredReadError::OpenDetachedData {
                path: raw_path.clone(),
                source,
            })?;
        source::check_encoded_source(&file, 0, payload_bytes, self.byte_skip, budget)?;
        let mut data_reader = BufReader::new(file);
        input::read_nrrd_payload(
            &mut data_reader,
            self.encoding,
            payload_bytes,
            sample_count,
            self.sample_type,
            &self.element_type,
            self.line_skip,
            self.byte_skip,
            0,
        )
    }

    fn read_inline_payload(
        &mut self,
        budget: ImageReadBudget,
        payload_bytes: usize,
        sample_count: usize,
    ) -> Result<Vec<u8>, NrrdStoredReadError> {
        source::check_encoded_source(
            self.reader.get_ref(),
            self.header_data_start,
            payload_bytes,
            self.byte_skip,
            budget,
        )?;
        input::read_nrrd_payload(
            &mut self.reader,
            self.encoding,
            payload_bytes,
            sample_count,
            self.sample_type,
            &self.element_type,
            self.line_skip,
            self.byte_skip,
            self.header_data_start,
        )
    }

    pub(in crate::reader) fn sample_count(&self) -> Result<usize, NrrdStoredReadError> {
        self.sizes.iter().try_fold(1_usize, |count, size| {
            count
                .checked_mul(*size)
                .ok_or_else(|| NrrdStoredReadError::ArrayElementCountOverflow {
                    sizes: self.sizes.clone(),
                })
        })
    }

    fn check_payload_budget(
        &self,
        budget: ImageReadBudget,
        sample_count: usize,
        payload_bytes: usize,
    ) -> Result<(), NrrdStoredReadError> {
        let decoded_bytes = sample_count
            .checked_mul(self.read_purpose.sample_width(self.sample_type))
            .ok_or(NrrdStoredReadError::DecodedByteCountOverflow {
                voxel_count: sample_count,
                sample_width: self.read_purpose.sample_width(self.sample_type),
            })?;
        let decoded_bytes_u64 = u64::try_from(decoded_bytes)
            .map_err(|_| NrrdStoredReadError::DecodedByteCountNotRepresentable { decoded_bytes })?;
        budget.check(ImageReadResource::DecodedBytes, decoded_bytes_u64)?;

        let payload_bytes_u64 = u64::try_from(payload_bytes).map_err(|_| {
            NrrdStoredReadError::PayloadLengthNotRepresentable {
                expected_bytes: payload_bytes,
            }
        })?;
        let gzip_skip_bytes = if self.encoding == NrrdEncoding::Gzip && self.byte_skip > 0 {
            u64::try_from(self.byte_skip).map_err(|_| NrrdStoredReadError::InvalidByteSkip {
                value: self.byte_skip,
                reason: "gzip byte skips must be nonnegative",
            })?
        } else {
            0
        };
        let expanded_bytes = payload_bytes_u64.checked_add(gzip_skip_bytes).ok_or(
            NrrdStoredReadError::ExpandedPayloadByteCountOverflow {
                payload_bytes: payload_bytes_u64,
                skipped_bytes: gzip_skip_bytes,
            },
        )?;
        budget.check(
            ImageReadResource::DecodedBytes,
            expanded_bytes.max(decoded_bytes_u64),
        )?;
        Ok(())
    }

    pub(in crate::reader) fn into_document_parts(
        self,
        sample_count: usize,
    ) -> (NrrdHeader, Vec<usize>, SampleType, usize) {
        (self.header, self.sizes, self.sample_type, sample_count)
    }
}

fn parse_byte_order(
    headers: &std::collections::HashMap<String, String>,
    encoding: NrrdEncoding,
    element_size: usize,
) -> Result<ByteOrder, NrrdStoredReadError> {
    if encoding == NrrdEncoding::Ascii {
        return Ok(ByteOrder::LeastSignificantByteFirst);
    }
    match headers.get("endian") {
        Some(endian) => {
            ByteOrder::from_nrrd(endian).map_err(|_| NrrdStoredReadError::InvalidByteOrder {
                endian: endian.clone(),
            })
        }
        None if element_size == 1 => Ok(ByteOrder::LeastSignificantByteFirst),
        None => Err(NrrdStoredReadError::MissingByteOrder {
            sample_width: element_size,
        }),
    }
}
