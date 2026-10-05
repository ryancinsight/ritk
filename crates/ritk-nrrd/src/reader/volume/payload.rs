//! NRRD header, spatial metadata, and stored payload parsing.

use anyhow::Result;
use ritk_codecs::{parse_usize_vec, ByteOrder};
use ritk_image_io::{ImageReadBudget, ImageReadResource};
use std::io::{BufReader, Seek};
use std::path::Path;

use super::super::decode::{element_type_spec, parse_space_direction_slots, sample_type};
use super::super::header::parse_nrrd_header_from_reader;
use super::super::stored::{NrrdSpatialMetadataField, NrrdStoredReadError};
use super::geometry;
use super::{NrrdReadPurpose, RawNrrd};
use crate::axes::{locate_acquisition_axis, AcquisitionAxis};

mod ascii;
mod input;
mod source;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NrrdEncoding {
    Raw,
    Ascii,
    Gzip,
}

pub(in crate::reader) fn parse_nrrd_raw<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
    read_purpose: NrrdReadPurpose,
) -> Result<RawNrrd, NrrdStoredReadError> {
    let path = path.as_ref();

    let file = std::fs::File::open(path).map_err(|source| NrrdStoredReadError::OpenHeader {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);

    let header = parse_nrrd_header_from_reader(&mut reader)
        .map_err(|source| NrrdStoredReadError::HeaderParse { source })?;
    let headers = &header.fields;
    let header_data_start = reader
        .stream_position()
        .map_err(|source| NrrdStoredReadError::PayloadIo { source })?;

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

    let direction_slots = if dimension == 2
        && !headers.contains_key("space")
        && !headers.contains_key("space dimension")
    {
        None
    } else {
        headers
            .get("space directions")
            .map(|s| {
                parse_space_direction_slots(s).map_err(|error| {
                    NrrdStoredReadError::SpatialMetadata {
                        field: NrrdSpatialMetadataField::SpaceDirections,
                        source: error,
                    }
                })
            })
            .transpose()?
    };
    let direction_flags: Option<Vec<bool>> = direction_slots
        .as_ref()
        .map(|slots| slots.iter().map(Option::is_some).collect());
    let acquisition = locate_acquisition_axis(
        dimension,
        headers.get("kinds").map(String::as_str),
        direction_flags.as_deref(),
    )
    .map_err(|error| NrrdStoredReadError::InvalidAcquisitionAxis { source: error })?;

    let sizes_str = headers
        .get("sizes")
        .ok_or(NrrdStoredReadError::MissingHeaderField { field: "sizes" })?;
    let sizes = parse_usize_vec(sizes_str, "sizes", dimension).map_err(|error| {
        NrrdStoredReadError::InvalidSizes {
            value: sizes_str.clone(),
            source: error,
        }
    })?;
    if let Some(axis) = sizes.iter().position(|size| *size == 0) {
        return Err(NrrdStoredReadError::EmptyAxis { axis });
    }

    let (volumes, spatial_sizes): (usize, &[usize]) = match acquisition {
        AcquisitionAxis::Absent => (1, &sizes[..]),
        AcquisitionAxis::Fastest => (sizes[0], &sizes[1..]),
        AcquisitionAxis::Slowest => (sizes[3], &sizes[..3]),
    };
    let nx = spatial_sizes[0];
    let ny = spatial_sizes[1];
    let nz = if spatial_sizes.len() >= 3 {
        spatial_sizes[2]
    } else {
        1
    };

    let encoding_text = headers
        .get("encoding")
        .ok_or(NrrdStoredReadError::MissingHeaderField { field: "encoding" })?;

    let encoding = match encoding_text.trim().to_ascii_lowercase().as_str() {
        "raw" => NrrdEncoding::Raw,
        "ascii" | "text" | "txt" => NrrdEncoding::Ascii,
        "gzip" | "gz" => NrrdEncoding::Gzip,
        other => {
            return Err(NrrdStoredReadError::UnsupportedEncoding {
                encoding: other.into(),
            })
        }
    };

    let (element_size, _, _) = element_type_spec(&element_type).map_err(|_| {
        NrrdStoredReadError::UnsupportedElementType {
            element_type: element_type.clone(),
        }
    })?;
    let sample_type =
        sample_type(&element_type).map_err(|_| NrrdStoredReadError::UnsupportedElementType {
            element_type: element_type.clone(),
        })?;
    let byte_order = if encoding == NrrdEncoding::Ascii {
        ByteOrder::LeastSignificantByteFirst
    } else {
        match headers.get("endian") {
            Some(endian) => {
                ByteOrder::from_nrrd(endian).map_err(|_| NrrdStoredReadError::InvalidByteOrder {
                    endian: endian.clone(),
                })?
            }
            None if element_size == 1 => ByteOrder::LeastSignificantByteFirst,
            None => {
                return Err(NrrdStoredReadError::MissingByteOrder {
                    sample_width: element_size,
                })
            }
        }
    };
    let line_skip = input::parse_line_skip(headers)?;
    let byte_skip = input::parse_byte_skip(headers)?;

    let spatial = geometry::parse_spatial_metadata(
        headers,
        dimension,
        acquisition,
        direction_flags.as_deref(),
    )?;

    let sizes_xyz = [nx, ny, nz];
    let voxels_per_volume = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or(NrrdStoredReadError::VoxelCountOverflow { sizes: sizes_xyz })?;
    let total_voxels = voxels_per_volume
        .checked_mul(volumes)
        .ok_or(NrrdStoredReadError::SeriesCountOverflow)?;
    let series_volumes =
        u64::try_from(volumes).map_err(|_| NrrdStoredReadError::SeriesCountOverflow)?;
    budget.check(ImageReadResource::SeriesVolumes, series_volumes)?;
    let output_sample_width = read_purpose.sample_width(sample_type);
    let decoded_bytes = total_voxels.checked_mul(output_sample_width).ok_or(
        NrrdStoredReadError::DecodedByteCountOverflow {
            voxel_count: total_voxels,
            sample_width: output_sample_width,
        },
    )?;
    let decoded_bytes_u64 = u64::try_from(decoded_bytes)
        .map_err(|_| NrrdStoredReadError::DecodedByteCountNotRepresentable { decoded_bytes })?;
    budget.check(ImageReadResource::DecodedBytes, decoded_bytes_u64)?;
    let expected_payload_bytes = total_voxels.checked_mul(element_size).ok_or(
        NrrdStoredReadError::PayloadByteCountOverflow {
            voxel_count: total_voxels,
            sample_width: element_size,
        },
    )?;
    let expected_payload_bytes_u64 = u64::try_from(expected_payload_bytes).map_err(|_| {
        NrrdStoredReadError::PayloadLengthNotRepresentable {
            expected_bytes: expected_payload_bytes,
        }
    })?;
    let gzip_skip_bytes = if encoding == NrrdEncoding::Gzip && byte_skip > 0 {
        u64::try_from(byte_skip).map_err(|_| NrrdStoredReadError::InvalidByteSkip {
            value: byte_skip,
            reason: "gzip byte skips must be nonnegative",
        })?
    } else {
        0
    };
    let expanded_payload_bytes = expected_payload_bytes_u64
        .checked_add(gzip_skip_bytes)
        .ok_or(NrrdStoredReadError::ExpandedPayloadByteCountOverflow {
            payload_bytes: expected_payload_bytes_u64,
            skipped_bytes: gzip_skip_bytes,
        })?;
    budget.check(
        ImageReadResource::DecodedBytes,
        expanded_payload_bytes.max(decoded_bytes_u64),
    )?;
    let coordinate_map = crate::coordinate_map::from_header_for_depth(&header.key_values, nz)
        .map_err(|error| NrrdStoredReadError::CoordinateMap { source: error })?;
    let series_axis = match read_purpose {
        NrrdReadPurpose::StoredVolume => {
            if acquisition != AcquisitionAxis::Absent {
                return Err(NrrdStoredReadError::AcquisitionAxisRequiresSeries {
                    axis: super::super::stored::acquisition_axis_index(acquisition),
                    kind: super::super::stored::acquisition_kind(&header, acquisition)
                        .map(str::to_owned),
                });
            }
            super::super::stored::stored_series_axis(&header, acquisition)?;
            None
        }
        NrrdReadPurpose::StoredSeries => Some(super::super::stored::stored_series_axis(
            &header,
            acquisition,
        )?),
        NrrdReadPurpose::ComputeF32 => None,
    };
    let data_file_field = headers.get("data file").cloned();
    let raw_bytes = match data_file_field.as_deref() {
        None => {
            source::check_encoded_source(
                reader.get_ref(),
                header_data_start,
                expected_payload_bytes,
                byte_skip,
                budget,
            )?;
            input::read_nrrd_payload(
                &mut reader,
                encoding,
                expected_payload_bytes,
                total_voxels,
                sample_type,
                &element_type,
                line_skip,
                byte_skip,
                header_data_start,
            )?
        }
        Some(data_file) if data_file.eq_ignore_ascii_case("internal") => {
            source::check_encoded_source(
                reader.get_ref(),
                header_data_start,
                expected_payload_bytes,
                byte_skip,
                budget,
            )?;
            input::read_nrrd_payload(
                &mut reader,
                encoding,
                expected_payload_bytes,
                total_voxels,
                sample_type,
                &element_type,
                line_skip,
                byte_skip,
                header_data_start,
            )?
        }
        Some(data_file) => {
            let raw_path = source::resolve_detached_data_path(path, data_file)?;
            let file = std::fs::File::open(&raw_path).map_err(|source| {
                NrrdStoredReadError::OpenDetachedData {
                    path: raw_path.clone(),
                    source,
                }
            })?;
            source::check_encoded_source(&file, 0, expected_payload_bytes, byte_skip, budget)?;
            let mut data_reader = BufReader::new(file);
            input::read_nrrd_payload(
                &mut data_reader,
                encoding,
                expected_payload_bytes,
                total_voxels,
                sample_type,
                &element_type,
                line_skip,
                byte_skip,
                0,
            )?
        }
    };

    Ok(RawNrrd {
        series_axis,
        raw_bytes,
        element_type,
        byte_order,
        acquisition,
        volumes,
        voxels_per_volume,
        dims: [nz, ny, nx],
        origin: spatial.origin,
        spacing: spatial.spacing,
        direction: spatial.direction,
        coordinate_map,
    })
}

#[cfg(test)]
mod budget_tests;
