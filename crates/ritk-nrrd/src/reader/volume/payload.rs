//! NRRD header, spatial metadata, and stored payload parsing.

use anyhow::Result;
use ritk_codecs::{parse_f64_vec, parse_usize_vec, ByteOrder};
use ritk_image_io::{ImageReadBudget, ImageReadResource};
use ritk_spatial::Point;
use std::io::{BufReader, Seek};
use std::path::Path;

use super::super::decode::{
    element_type_spec, parse_nrrd_point, parse_nrrd_point_planar, parse_space_direction_slots,
    parse_space_directions, parse_space_directions_planar, parse_space_directions_planar_world,
    sample_type,
};
use super::super::header::parse_nrrd_header_from_reader;
use super::super::stored::{NrrdSpatialMetadataField, NrrdStoredReadError};
use super::{NrrdReadPurpose, RawNrrd};
use crate::axes::{locate_acquisition_axis, AcquisitionAxis};
use crate::spatial::{
    directions_to_lps, metadata_from_file_space_directions,
    metadata_from_planar_file_space_directions, vector_to_lps, world_to_lps_factors,
};

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
    if headers.contains_key("space directions") && headers.contains_key("spacings") {
        return Err(NrrdStoredReadError::ConflictingSpatialFields);
    }
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

    let world_to_lps = world_to_lps_factors(
        headers.get("space").map(String::as_str),
        headers.get("space dimension").map(String::as_str),
        headers.get("space units").map(String::as_str),
    )
    .map_err(|error| NrrdStoredReadError::SpatialMetadata {
        field: NrrdSpatialMetadataField::CoordinateSystem,
        source: error,
    })?;

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

    let spatial = if let Some(sd_str) = headers.get("space directions") {
        if dimension == 2 && headers.contains_key("space") {
            let planar_directions =
                parse_space_directions_planar_world(sd_str).map_err(|error| {
                    NrrdStoredReadError::SpatialMetadata {
                        field: NrrdSpatialMetadataField::SpaceDirections,
                        source: error,
                    }
                })?;
            metadata_from_planar_file_space_directions(planar_directions, world_to_lps).map_err(
                |error| NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source: error,
                },
            )?
        } else if dimension == 2 {
            let mut dirs = parse_space_directions_planar(sd_str).map_err(|error| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source: error,
                }
            })?;
            dirs[0] = vector_to_lps(dirs[0], world_to_lps).map_err(|source| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source,
                }
            })?;
            dirs[1] = vector_to_lps(dirs[1], world_to_lps).map_err(|source| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source,
                }
            })?;
            dirs[2] = [0.0, 0.0, 1.0];
            metadata_from_file_space_directions(dirs).map_err(|error| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source: error,
                }
            })?
        } else {
            let dirs = parse_space_directions(sd_str).map_err(|error| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source: error,
                }
            })?;
            let dirs = directions_to_lps(dirs, world_to_lps).map_err(|source| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source,
                }
            })?;
            metadata_from_file_space_directions(dirs).map_err(|error| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceDirections,
                    source: error,
                }
            })?
        }
    } else if let Some(sp_str) = headers.get("spacings") {
        let sp = parse_f64_vec(sp_str, "spacings", dimension).map_err(|error| {
            NrrdStoredReadError::SpatialMetadata {
                field: NrrdSpatialMetadataField::Spacings,
                source: error,
            }
        })?;
        let sp: Vec<f64> = match acquisition {
            AcquisitionAxis::Absent => sp,
            AcquisitionAxis::Fastest => sp[1..].to_vec(),
            AcquisitionAxis::Slowest => sp[..3].to_vec(),
        };
        let sz = if sp.len() >= 3 { sp[2] } else { 1.0 };
        let file_directions = [[sp[0], 0.0, 0.0], [0.0, sp[1], 0.0], [0.0, 0.0, sz]];
        let directions = if dimension == 2 {
            [
                vector_to_lps(file_directions[0], world_to_lps).map_err(|source| {
                    NrrdStoredReadError::SpatialMetadata {
                        field: NrrdSpatialMetadataField::Spacings,
                        source,
                    }
                })?,
                vector_to_lps(file_directions[1], world_to_lps).map_err(|source| {
                    NrrdStoredReadError::SpatialMetadata {
                        field: NrrdSpatialMetadataField::Spacings,
                        source,
                    }
                })?,
                [0.0, 0.0, 1.0],
            ]
        } else {
            directions_to_lps(file_directions, world_to_lps).map_err(|source| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::Spacings,
                    source,
                }
            })?
        };
        metadata_from_file_space_directions(directions).map_err(|error| {
            NrrdStoredReadError::SpatialMetadata {
                field: NrrdSpatialMetadataField::Spacings,
                source: error,
            }
        })?
    } else {
        let default_directions = directions_to_lps(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            world_to_lps,
        )
        .map_err(|source| NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::CoordinateSystem,
            source,
        })?;
        metadata_from_file_space_directions(default_directions).map_err(|error| {
            NrrdStoredReadError::SpatialMetadata {
                field: NrrdSpatialMetadataField::Spacings,
                source: error,
            }
        })?
    };

    let origin = if let Some(so_str) = headers.get("space origin") {
        if dimension == 2 && !headers.contains_key("space") {
            parse_nrrd_point_planar(so_str).map_err(|error| {
                NrrdStoredReadError::SpatialMetadata {
                    field: NrrdSpatialMetadataField::SpaceOrigin,
                    source: error,
                }
            })?
        } else {
            parse_nrrd_point(so_str).map_err(|error| NrrdStoredReadError::SpatialMetadata {
                field: NrrdSpatialMetadataField::SpaceOrigin,
                source: error,
            })?
        }
    } else {
        Point::new([0.0, 0.0, 0.0])
    };
    let file_origin = origin.to_array();
    let lps_origin = vector_to_lps(file_origin, world_to_lps).map_err(|source| {
        NrrdStoredReadError::SpatialMetadata {
            field: NrrdSpatialMetadataField::SpaceOrigin,
            source,
        }
    })?;
    let origin = Point::new(lps_origin);

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
        origin,
        spacing: spatial.spacing,
        direction: spatial.direction,
        coordinate_map,
    })
}

#[cfg(test)]
mod budget_tests;
