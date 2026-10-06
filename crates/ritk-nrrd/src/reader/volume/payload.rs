//! NRRD header, spatial metadata, and stored payload parsing.

use anyhow::{anyhow, Result};
use ritk_codecs::{parse_usize_vec, ByteOrder, SampleType};
use ritk_image_io::{ImageReadBudget, ImageReadResource, SeriesAxis};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
use std::io::{BufReader, Seek};
use std::path::Path;

use super::super::decode::{
    element_type_spec, first_group_width, parse_space_direction_slots, sample_type,
    strip_none_token,
};
use super::super::header::{parse_nrrd_header_from_reader, NrrdHeader};
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

pub(crate) struct NrrdReadPlan {
    pub(crate) series_axis: Option<SeriesAxis>,
    pub(crate) volumes: usize,
    pub(crate) dims: [usize; 3],
    pub(crate) origin: Point<3>,
    pub(crate) spacing: Spacing<3>,
    pub(crate) direction: Direction<3>,
    pub(crate) coordinate_map: CoordinateMap,
    pub(crate) sample_type: SampleType,
    header_data_start: u64,
    element_type: String,
    encoding: NrrdEncoding,
    byte_order: ByteOrder,
    acquisition: AcquisitionAxis,
    voxels_per_volume: usize,
    total_voxels: usize,
    expected_payload_bytes: usize,
    line_skip: i32,
    byte_skip: i32,
    data_file_field: Option<String>,
}

/// Mark one slot per `space directions` entry without parsing values.
///
/// Acquisition location and axis validation consume presence, not values;
/// values resolve later through the width-selected parser. A 2-D field with
/// two-component vectors therefore marks present slots instead of failing
/// the 3-component parse. Malformed structure (unterminated groups, stray
/// text, `none`-prefixed words) fails here with the same vocabulary as the
/// value parser.
fn mark_space_direction_slots(s: &str) -> Result<Vec<bool>> {
    let mut present = Vec::new();
    let mut rest = s.trim();
    while !rest.is_empty() {
        if let Some(after_none) = strip_none_token(rest) {
            present.push(false);
            rest = after_none.trim_start();
            continue;
        }
        let Some(after_open) = rest.strip_prefix('(') else {
            return Err(anyhow!(
                "Unexpected text outside vector group in '{}': '{}'",
                s,
                rest
            ));
        };
        let Some(end) = after_open.find(')') else {
            return Err(anyhow!("Unterminated vector group in '{}'", s));
        };
        present.push(true);
        rest = after_open[end + 1..].trim_start();
    }
    Ok(present)
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
    parse_nrrd_raw_with_header(path, &mut reader, &header, budget, read_purpose)
}

fn parse_nrrd_raw_with_header(
    path: &Path,
    reader: &mut BufReader<std::fs::File>,
    header: &NrrdHeader,
    budget: ImageReadBudget,
    read_purpose: NrrdReadPurpose,
) -> Result<RawNrrd, NrrdStoredReadError> {
    let plan = parse_nrrd_read_plan(reader, header, budget, read_purpose)?;
    read_nrrd_payload(path, reader, plan, budget)
}

pub(crate) fn parse_nrrd_read_plan(
    reader: &mut BufReader<std::fs::File>,
    header: &NrrdHeader,
    budget: ImageReadBudget,
    read_purpose: NrrdReadPurpose,
) -> Result<NrrdReadPlan, NrrdStoredReadError> {
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

    let direction_flags: Option<Vec<bool>> = match headers.get("space directions") {
        None => None,
        Some(_)
            if dimension == 2
                && !headers.contains_key("space")
                && !headers.contains_key("space dimension") =>
        {
            None
        }
        Some(value) if dimension == 2 && first_group_width(value) == Some(2) => {
            // Two-component vectors in a 2-D field: only presence feeds
            // acquisition location and axis validation here; the values
            // resolve later through the planar promotion.
            Some(mark_space_direction_slots(value).map_err(|error| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source: error,
                }
            })?)
        }
        Some(value) => Some(
            parse_space_direction_slots(value)
                .map_err(|error| NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source: error,
                })?
                .iter()
                .map(Option::is_some)
                .collect(),
        ),
    };
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
                    kind: super::super::stored::acquisition_kind(header, acquisition)
                        .map(str::to_owned),
                });
            }
            super::super::stored::stored_series_axis(header, acquisition)?;
            None
        }
        NrrdReadPurpose::StoredSeries => Some(super::super::stored::stored_series_axis(
            header,
            acquisition,
        )?),
        NrrdReadPurpose::ComputeF32 => None,
    };
    Ok(NrrdReadPlan {
        series_axis,
        volumes,
        dims: [nz, ny, nx],
        origin: spatial.origin,
        spacing: spatial.spacing,
        direction: spatial.direction,
        coordinate_map,
        sample_type,
        header_data_start,
        element_type,
        encoding,
        byte_order,
        acquisition,
        voxels_per_volume,
        total_voxels,
        expected_payload_bytes,
        line_skip,
        byte_skip,
        data_file_field: headers.get("data file").cloned(),
    })
}

pub(crate) fn read_nrrd_payload(
    path: &Path,
    mut reader: &mut BufReader<std::fs::File>,
    plan: NrrdReadPlan,
    budget: ImageReadBudget,
) -> Result<RawNrrd, NrrdStoredReadError> {
    let NrrdReadPlan {
        series_axis,
        volumes,
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
        sample_type,
        header_data_start,
        element_type,
        encoding,
        byte_order,
        acquisition,
        voxels_per_volume,
        total_voxels,
        expected_payload_bytes,
        line_skip,
        byte_skip,
        data_file_field,
    } = plan;
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
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
    })
}

#[cfg(test)]
mod budget_tests;
