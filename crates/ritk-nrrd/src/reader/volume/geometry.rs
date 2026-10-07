//! Interpretation of NRRD spatial and per-axis geometry metadata.

use anyhow::Result;
use ritk_codecs::parse_f64_vec;
use ritk_spatial::{Direction, Point, Spacing};
use std::collections::HashMap;

use super::super::decode::{
    first_group_width, parse_nrrd_point, parse_nrrd_point_planar, parse_space_directions,
    parse_space_directions_planar, parse_space_directions_planar_world,
};
use super::super::stored::{NrrdSpatialMetadataField, NrrdStoredReadError};
use crate::axes::AcquisitionAxis;
use crate::spatial::{
    directions_to_lps, metadata_from_file_space_directions,
    metadata_from_planar_file_space_directions, parse_axis_centerings, parse_axis_units,
    vector_to_lps, world_to_lps_factors,
};

/// Patient-space geometry resolved from a NRRD header before payload access.
pub(super) struct NrrdSpatialMetadata {
    pub(super) origin: Point<3>,
    pub(super) spacing: Spacing<3>,
    pub(super) direction: Direction<3>,
}

/// Resolves the declared sample grid into RITK's LPS-millimeter coordinates.
///
/// NRRD `space origin` locates the center of the first sample and each space
/// direction is the displacement to the next sample. The implementation
/// follows Teem's NRRD format specification, sections 4 and 6:
/// <https://teem.sourceforge.net/nrrd/format.html#space-directions>
///
/// # Errors
///
/// Returns a typed error for malformed or unsupported spatial fields. This
/// function runs before payload allocation so invalid geometry cannot force a
/// large read before rejection.
pub(super) fn parse_spatial_metadata(
    headers: &HashMap<String, String>,
    dimension: usize,
    acquisition: AcquisitionAxis,
    direction_flags: Option<&[bool]>,
) -> Result<NrrdSpatialMetadata, NrrdStoredReadError> {
    // A 2-D array declaring a 2-D world without a named space needs no
    // basis mapping: the world is already planar and the promotion below
    // appends the through-plane axis. Any other `space dimension` value
    // still fails in the basis conversion, as does a named space paired
    // with one.
    let space_dimension = match headers.get("space dimension").map(String::as_str) {
        Some(value) if dimension == 2 && !headers.contains_key("space") && value.trim() == "2" => {
            None
        }
        space_dimension => space_dimension,
    };
    let world_to_lps = world_to_lps_factors(
        headers.get("space").map(String::as_str),
        space_dimension,
        headers.get("space units").map(String::as_str),
    )
    .map_err(|source| spatial_error(NrrdSpatialMetadataField::CoordinateSystem, source))?;

    let file_axes = spatial_file_axes(dimension, acquisition);
    let axis_units = headers
        .get("units")
        .map(|value| {
            parse_axis_units(value, dimension)
                .map_err(|source| spatial_error(NrrdSpatialMetadataField::AxisUnits, source))
        })
        .transpose()?;
    reject_unrepresented_centering(headers, dimension)?;
    let spacings = parse_optional_axis_values(
        headers,
        "spacings",
        dimension,
        NrrdSpatialMetadataField::Spacings,
    )?;
    let axis_mins = parse_optional_axis_values(
        headers,
        "axis mins",
        dimension,
        NrrdSpatialMetadataField::AxisBounds,
    )?;
    let axis_maxs = parse_optional_axis_values(
        headers,
        "axis maxs",
        dimension,
        NrrdSpatialMetadataField::AxisBounds,
    )?;
    let has_directions = headers.contains_key("space directions");

    if !has_directions
        && spacings.is_none()
        && axis_units.as_deref().is_some_and(|units| {
            file_axes
                .iter()
                .any(|axis| units.get(*axis).copied().flatten().is_some())
        })
    {
        return Err(spatial_error(
            NrrdSpatialMetadataField::AxisUnits,
            anyhow::anyhow!("per-axis units without spacings or space directions cannot define physical geometry"),
        ));
    }

    if has_directions {
        validate_direction_axis_fields(
            dimension,
            &file_axes,
            direction_flags,
            spacings.as_deref(),
            axis_units.as_deref(),
            axis_mins.as_deref(),
            axis_maxs.as_deref(),
        )?;
    } else {
        validate_axis_bounds_without_directions(
            &file_axes,
            axis_mins.as_deref(),
            axis_maxs.as_deref(),
        )?;
        validate_acquisition_axis_fields(
            dimension,
            acquisition,
            spacings.as_deref(),
            axis_units.as_deref(),
        )?;
    }

    let metadata = if let Some(directions) = headers.get("space directions") {
        parse_space_directions_metadata(directions, headers, dimension, world_to_lps)?
    } else if let Some(spacings) = spacings.as_deref() {
        parse_scalar_spacings(
            spacings,
            axis_units.as_deref(),
            &file_axes,
            world_to_lps,
            dimension,
        )?
    } else {
        let directions = directions_to_lps(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            world_to_lps,
        )
        .map_err(|source| spatial_error(NrrdSpatialMetadataField::CoordinateSystem, source))?;
        metadata_from_file_space_directions(directions)
            .map_err(|source| spatial_error(NrrdSpatialMetadataField::Spacings, source))?
    };

    let origin = parse_origin(headers, dimension, world_to_lps)?;
    Ok(NrrdSpatialMetadata {
        origin,
        spacing: metadata.spacing,
        direction: metadata.direction,
    })
}

/// Promote 2-D file directions to a 3-D matrix with an identity
/// through-plane z-axis, mapping the in-plane vectors into LPS.
///
/// This is the 2-D-as-z1 convention shared by files with and without a named
/// space: the component width of the field (not the `space` key) decides
/// between this and the rank-2 world parser, so a 2-D file in a named space
/// with two-component vectors reads exactly like its spaceless twin.
fn promote_planar_directions(
    value: &str,
    world_to_lps: [f64; 3],
) -> Result<crate::spatial::InternalSpatialMetadata> {
    let mut directions = parse_space_directions_planar(value)
        .map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceDirections, source))?;
    directions[0] = vector_to_lps(directions[0], world_to_lps)
        .map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceDirections, source))?;
    directions[1] = vector_to_lps(directions[1], world_to_lps)
        .map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceDirections, source))?;
    directions[2] = [0.0, 0.0, 1.0];
    metadata_from_file_space_directions(directions)
}

fn parse_space_directions_metadata(
    value: &str,
    headers: &HashMap<String, String>,
    dimension: usize,
    world_to_lps: [f64; 3],
) -> Result<crate::spatial::InternalSpatialMetadata, NrrdStoredReadError> {
    let metadata = if dimension == 2
        && (!headers.contains_key("space") || first_group_width(value) == Some(2))
    {
        promote_planar_directions(value, world_to_lps)
    } else if dimension == 2 {
        let directions = parse_space_directions_planar_world(value)
            .map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceDirections, source))?;
        metadata_from_planar_file_space_directions(directions, world_to_lps)
    } else {
        let directions = parse_space_directions(value)
            .map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceDirections, source))?;
        let directions = directions_to_lps(directions, world_to_lps)
            .map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceDirections, source))?;
        metadata_from_file_space_directions(directions)
    };
    metadata.map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceDirections, source))
}

fn parse_scalar_spacings(
    spacings: &[f64],
    units: Option<&[Option<f64>]>,
    file_axes: &[usize],
    world_to_lps: [f64; 3],
    dimension: usize,
) -> Result<crate::spatial::InternalSpatialMetadata, NrrdStoredReadError> {
    let mut directions = [[0.0; 3]; 3];
    for (space_axis, file_axis) in file_axes.iter().copied().enumerate() {
        let spacing = spacings[file_axis];
        if !spacing.is_finite() || spacing == 0.0 {
            return Err(spatial_error(
                NrrdSpatialMetadataField::Spacings,
                anyhow::anyhow!("spatial axis {file_axis} has unknown, infinite, or zero spacing"),
            ));
        }
        let world_scale = units
            .and_then(|values| values[file_axis])
            .map(|millimeters| world_to_lps[space_axis].signum() * millimeters)
            .unwrap_or(world_to_lps[space_axis]);
        directions[space_axis][space_axis] = spacing * world_scale;
        if !directions[space_axis][space_axis].is_finite()
            || directions[space_axis][space_axis] == 0.0
        {
            return Err(spatial_error(
                NrrdSpatialMetadataField::Spacings,
                anyhow::anyhow!(
                    "spatial axis {file_axis} spacing is not representable in millimeters"
                ),
            ));
        }
    }
    if dimension == 2 {
        directions[2][2] = 1.0;
    }
    metadata_from_file_space_directions(directions)
        .map_err(|source| spatial_error(NrrdSpatialMetadataField::Spacings, source))
}

fn parse_origin(
    headers: &HashMap<String, String>,
    dimension: usize,
    world_to_lps: [f64; 3],
) -> Result<Point<3>, NrrdStoredReadError> {
    let origin = if let Some(value) = headers.get("space origin") {
        let planar = !headers.contains_key("space") || first_group_width(value) == Some(2);
        if dimension == 2 && planar {
            parse_nrrd_point_planar(value)
        } else {
            parse_nrrd_point(value)
        }
        .map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceOrigin, source))?
    } else {
        Point::new([0.0, 0.0, 0.0])
    };
    let lps = vector_to_lps(origin.to_array(), world_to_lps)
        .map_err(|source| spatial_error(NrrdSpatialMetadataField::SpaceOrigin, source))?;
    Ok(Point::new(lps))
}

fn parse_optional_axis_values(
    headers: &HashMap<String, String>,
    field: &'static str,
    dimension: usize,
    metadata_field: NrrdSpatialMetadataField,
) -> Result<Option<Vec<f64>>, NrrdStoredReadError> {
    headers
        .get(field)
        .map(|value| {
            parse_f64_vec(value, field, dimension)
                .map_err(|source| spatial_error(metadata_field, source))
                .and_then(|values| {
                    if values.iter().any(|value| value.is_infinite()) {
                        Err(spatial_error(
                            metadata_field,
                            anyhow::anyhow!("{field} contains an infinite value"),
                        ))
                    } else {
                        Ok(values)
                    }
                })
        })
        .transpose()
}

fn validate_direction_axis_fields(
    dimension: usize,
    file_axes: &[usize],
    direction_flags: Option<&[bool]>,
    spacings: Option<&[f64]>,
    units: Option<&[Option<f64>]>,
    axis_mins: Option<&[f64]>,
    axis_maxs: Option<&[f64]>,
) -> Result<(), NrrdStoredReadError> {
    if spacings.is_some_and(|values| values.iter().any(|value| value.is_infinite())) {
        return Err(spatial_error(
            NrrdSpatialMetadataField::Spacings,
            anyhow::anyhow!("spacings contains an infinite value"),
        ));
    }
    for axis in 0..dimension {
        let has_direction = direction_flags
            .and_then(|flags| flags.get(axis))
            .copied()
            .unwrap_or(file_axes.contains(&axis));
        if !has_direction {
            if spacings.is_some_and(|values| values[axis].is_finite()) {
                return Err(spatial_error(
                    NrrdSpatialMetadataField::Spacings,
                    anyhow::anyhow!("non-spatial acquisition axis {axis} has a finite spacing"),
                ));
            }
            if units.is_some_and(|values| values[axis].is_some()) {
                return Err(spatial_error(
                    NrrdSpatialMetadataField::AxisUnits,
                    anyhow::anyhow!("non-spatial acquisition axis {axis} declares a physical unit"),
                ));
            }
            if axis_mins.is_some_and(|values| values[axis].is_finite())
                || axis_maxs.is_some_and(|values| values[axis].is_finite())
            {
                return Err(spatial_error(
                    NrrdSpatialMetadataField::AxisBounds,
                    anyhow::anyhow!("non-spatial acquisition axis {axis} has finite axis bounds"),
                ));
            }
            continue;
        }
        if spacings.is_some_and(|values| values[axis].is_finite()) {
            return Err(NrrdStoredReadError::ConflictingSpatialFields);
        }
        if units.is_some_and(|values| values[axis].is_some()) {
            return Err(spatial_error(
                NrrdSpatialMetadataField::AxisUnits,
                anyhow::anyhow!("axis {axis} has both a space direction and a per-axis unit"),
            ));
        }
        if axis_mins.is_some_and(|values| values[axis].is_finite())
            || axis_maxs.is_some_and(|values| values[axis].is_finite())
        {
            return Err(spatial_error(
                NrrdSpatialMetadataField::AxisBounds,
                anyhow::anyhow!("axis {axis} has both a space direction and finite axis bounds"),
            ));
        }
    }
    Ok(())
}

fn validate_axis_bounds_without_directions(
    file_axes: &[usize],
    axis_mins: Option<&[f64]>,
    axis_maxs: Option<&[f64]>,
) -> Result<(), NrrdStoredReadError> {
    for axis in file_axes.iter().copied() {
        if axis_mins.is_some_and(|values| values[axis].is_finite())
            || axis_maxs.is_some_and(|values| values[axis].is_finite())
        {
            return Err(spatial_error(
                NrrdSpatialMetadataField::AxisBounds,
                anyhow::anyhow!("finite axis bounds require a centering-aware grid, which StoredVolume does not represent"),
            ));
        }
    }
    Ok(())
}

fn validate_acquisition_axis_fields(
    dimension: usize,
    acquisition: AcquisitionAxis,
    spacings: Option<&[f64]>,
    units: Option<&[Option<f64>]>,
) -> Result<(), NrrdStoredReadError> {
    let axis = match acquisition {
        AcquisitionAxis::Absent => return Ok(()),
        AcquisitionAxis::Fastest => 0,
        AcquisitionAxis::Slowest => dimension - 1,
    };
    if spacings.is_some_and(|values| values[axis].is_finite()) {
        return Err(spatial_error(
            NrrdSpatialMetadataField::Spacings,
            anyhow::anyhow!("non-spatial acquisition axis {axis} has a finite spacing"),
        ));
    }
    if units.is_some_and(|values| values[axis].is_some()) {
        return Err(spatial_error(
            NrrdSpatialMetadataField::AxisUnits,
            anyhow::anyhow!("non-spatial acquisition axis {axis} declares a physical unit"),
        ));
    }
    Ok(())
}

fn reject_unrepresented_centering(
    headers: &HashMap<String, String>,
    dimension: usize,
) -> Result<(), NrrdStoredReadError> {
    let Some(value) = headers.get("centers") else {
        return Ok(());
    };
    let centerings = parse_axis_centerings(value, dimension)
        .map_err(|source| spatial_error(NrrdSpatialMetadataField::Centering, source))?;
    if let Some(axis) = centerings
        .iter()
        .position(|is_cell_or_node| *is_cell_or_node)
    {
        return Err(spatial_error(
            NrrdSpatialMetadataField::Centering,
            anyhow::anyhow!("axis {axis} declares cell/node support, which the stored-volume model does not represent"),
        ));
    }
    Ok(())
}

fn spatial_file_axes(dimension: usize, acquisition: AcquisitionAxis) -> Vec<usize> {
    match acquisition {
        AcquisitionAxis::Absent => (0..dimension).collect(),
        AcquisitionAxis::Fastest => (1..dimension).collect(),
        AcquisitionAxis::Slowest => (0..dimension - 1).collect(),
    }
}

fn spatial_error(
    field: NrrdSpatialMetadataField,
    source: impl Into<anyhow::Error>,
) -> NrrdStoredReadError {
    NrrdStoredReadError::SpatialMetadata {
        field,
        source: source.into(),
    }
}
