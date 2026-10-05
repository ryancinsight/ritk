//! NRRD header, spatial metadata, and stored payload parsing.

use anyhow::{anyhow, Result};
use ritk_image_io::{ImageReadBudget, ImageReadResource};
use std::path::Path;

use super::super::decode::{first_group_width, parse_space_direction_slots, strip_none_token};
use super::super::stored::{NrrdSpatialMetadataField, NrrdStoredReadError};
use super::geometry;
use super::{NrrdReadPurpose, RawNrrd};
use crate::axes::{locate_acquisition_axis, AcquisitionAxis};

mod ascii;
mod input;
mod plan;
mod source;
pub(in crate::reader) use plan::NrrdPayloadPlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NrrdEncoding {
    Raw,
    Ascii,
    Gzip,
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
    let mut plan = NrrdPayloadPlan::open(path, read_purpose)?;
    let dimension = plan.dimension();
    let sizes = plan.sizes();
    let (acquisition, volumes, dims, spatial, coordinate_map, series_axis) = {
        let header = plan.header();
        let headers = &header.fields;
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
                Some(mark_space_direction_slots(value).map_err(|source| {
                    NrrdStoredReadError::SpatialMetadata {
                        field: NrrdSpatialMetadataField::SpaceDirections,
                        source,
                    }
                })?)
            }
            Some(value) => Some(
                parse_space_direction_slots(value)
                    .map_err(|source| NrrdStoredReadError::SpatialMetadata {
                        field: NrrdSpatialMetadataField::SpaceDirections,
                        source,
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
        .map_err(|source| NrrdStoredReadError::InvalidAcquisitionAxis { source })?;
        let (volumes, spatial_sizes): (usize, &[usize]) = match acquisition {
            AcquisitionAxis::Absent => (1, sizes),
            AcquisitionAxis::Fastest => (sizes[0], &sizes[1..]),
            AcquisitionAxis::Slowest => (sizes[3], &sizes[..3]),
        };
        let nx = spatial_sizes[0];
        let ny = spatial_sizes[1];
        let nz = spatial_sizes.get(2).copied().unwrap_or(1);
        let spatial = geometry::parse_spatial_metadata(
            headers,
            dimension,
            acquisition,
            direction_flags.as_deref(),
        )?;
        let coordinate_map = crate::coordinate_map::from_header_for_depth(&header.key_values, nz)
            .map_err(|source| NrrdStoredReadError::CoordinateMap { source })?;
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
            NrrdReadPurpose::StoredDocument => None,
            NrrdReadPurpose::ComputeF32 => None,
        };
        (
            acquisition,
            volumes,
            [nz, ny, nx],
            spatial,
            coordinate_map,
            series_axis,
        )
    };

    let sizes_xyz = [dims[2], dims[1], dims[0]];
    let voxels_per_volume = dims[2]
        .checked_mul(dims[1])
        .and_then(|plane| plane.checked_mul(dims[0]))
        .ok_or(NrrdStoredReadError::VoxelCountOverflow { sizes: sizes_xyz })?;
    let total_voxels = voxels_per_volume
        .checked_mul(volumes)
        .ok_or(NrrdStoredReadError::SeriesCountOverflow)?;
    let expected_samples = plan.sample_count()?;
    if total_voxels != expected_samples {
        return Err(NrrdStoredReadError::ArraySampleCountMismatch {
            expected_samples,
            actual_samples: total_voxels,
        });
    }
    let series_volumes =
        u64::try_from(volumes).map_err(|_| NrrdStoredReadError::SeriesCountOverflow)?;
    budget.check(ImageReadResource::SeriesVolumes, series_volumes)?;

    let element_type = plan.element_type().to_owned();
    let byte_order = plan.byte_order()?;
    let raw_bytes = plan.read_payload(budget)?;
    Ok(RawNrrd {
        series_axis,
        raw_bytes,
        element_type,
        byte_order,
        acquisition,
        volumes,
        voxels_per_volume,
        dims,
        origin: spatial.origin,
        spacing: spatial.spacing,
        direction: spatial.direction,
        coordinate_map,
    })
}

#[cfg(test)]
mod budget_tests;
