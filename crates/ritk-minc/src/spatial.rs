//! Spatial metadata construction for MINC2 volumes.
//!
//! Provides functions to:
//! - determine the canonical direction cosine for each MINC2 axis,
//! - read dimension groups from HDF5,
//! - order dimensions by the `dimorder` attribute,
//! - build RITK spatial metadata (`origin`, `spacing`, `direction`).

use crate::{
    attrs::{extract_dimorder, parse_dimension_attrs},
    MincDimension, DIMENSIONS_PATH, SPATIAL_DIM_NAMES,
};
use anyhow::{bail, Result};
use ritk_spatial::{Direction, Point, Spacing, Vector};

// â”€â”€ Canonical direction cosines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Default direction cosines for a named spatial dimension.
///
/// - `xspace` â†’ `[1, 0, 0]`
/// - `yspace` â†’ `[0, 1, 0]`
/// - `zspace` â†’ `[0, 0, 1]`
/// - unknown  â†’ `[1, 0, 0]`
pub fn default_direction_cosines(name: &str) -> [f64; 3] {
    match name {
        "xspace" => [1.0, 0.0, 0.0],
        "yspace" => [0.0, 1.0, 0.0],
        "zspace" => [0.0, 0.0, 1.0],
        _ => [1.0, 0.0, 0.0],
    }
}

// â”€â”€ Dimension metadata reading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Read spatial dimension metadata from the MINC2 HDF5 dimensions group.
///
/// Navigates to `/minc-2.0/dimensions/` and reads attributes from each
/// recognized spatial dimension group (`xspace`, `yspace`, `zspace`).
///
/// Returns a `Vec<MincDimension>`, one per spatial dimension found.
/// An error is returned if no spatial dimensions are found.
pub fn read_dimension_metadata<R: consus_io::ReadAt + Sync>(
    hdf5: &consus_hdf5::file::Hdf5File<R>,
) -> Result<Vec<MincDimension>> {
    let dims_addr = hdf5
        .open_path(DIMENSIONS_PATH)
        .map_err(|e| anyhow::anyhow!("Cannot locate {}: {}", DIMENSIONS_PATH, e))?;

    let children = hdf5
        .list_group_at(dims_addr)
        .map_err(|e| anyhow::anyhow!("Cannot list dimensions group: {}", e))?;

    let mut dimensions = Vec::with_capacity(3);

    for (name, addr, _link_type) in &children {
        if !SPATIAL_DIM_NAMES.contains(&name.as_str()) {
            continue;
        }

        let attrs = hdf5
            .attributes_at(*addr)
            .map_err(|e| anyhow::anyhow!("Cannot read attributes for dimension {}: {}", name, e))?;

        let dim = parse_dimension_attrs(name, &attrs)?;
        dimensions.push(dim);
    }

    if dimensions.is_empty() {
        bail!(
            "No spatial dimension groups found under {}",
            DIMENSIONS_PATH
        );
    }

    Ok(dimensions)
}

// â”€â”€ Dimension ordering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Order parsed dimensions according to the dimorder specification.
///
/// Returns a 3-element vector where index 0 corresponds to the outermost
/// (slowest-varying) tensor axis and index 2 to the innermost (fastest).
///
/// # Errors
///
/// Returns `Err` if a dimension name referenced by `dimorder` is not present
/// in `dimensions`, or if fewer than 3 entries are produced.
pub fn order_dimensions_by_dimorder(
    dimensions: &[MincDimension],
    dimorder: &[String],
) -> Result<Vec<MincDimension>> {
    let mut ordered = Vec::with_capacity(3);

    for dim_name in dimorder.iter().take(3) {
        let dim = dimensions
            .iter()
            .find(|d| d.name == *dim_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "dimorder references '{}' but no matching dimension found",
                    dim_name
                )
            })?;
        ordered.push(dim.clone());
    }

    if ordered.len() != 3 {
        bail!(
            "Expected 3 spatial dimensions from dimorder, found {}",
            ordered.len()
        );
    }

    Ok(ordered)
}

// â”€â”€ Image attribute helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Extract `dimorder` from image dataset attributes.
pub fn read_dimorder(attrs: &[consus_hdf5::attribute::Hdf5Attribute]) -> Result<Vec<String>> {
    extract_dimorder(attrs)
}

// â”€â”€ Spatial metadata construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Build RITK spatial metadata from ordered dimension metadata.
///
/// # Arguments
///
/// `ordered_dims`: dimensions ordered by dimorder (index 0 = outermost axis).
///
/// # Returns
///
/// `(origin, spacing, direction)` triple for `Image` construction.
///
/// # Derivation
///
/// - `origin[i] = ordered_dims[i].start`
/// - `spacing[i] = |ordered_dims[i].step|`
/// - `direction` columns = `ordered_dims[i].direction_cosines`
///
/// Step sign is absorbed into the direction cosines: if `step < 0`,
/// the direction cosine vector is negated to maintain positive spacing.
pub fn build_spatial_metadata(
    ordered_dims: &[MincDimension],
) -> (Point<3>, Spacing<3>, Direction<3>) {
    let mut origin_arr = [0.0f64; 3];
    let mut spacing_arr = [0.0f64; 3];
    let mut dir_columns: [[f64; 3]; 3] = [[0.0; 3]; 3];

    for (i, dim) in ordered_dims.iter().enumerate() {
        origin_arr[i] = dim.start;

        let abs_step = dim.step.abs();
        spacing_arr[i] = if abs_step > 0.0 { abs_step } else { 1.0 };

        // Negate direction cosines when step is negative so that spacing
        // is always positive and orientation is preserved.
        let sign = if dim.step < 0.0 { -1.0 } else { 1.0 };
        dir_columns[i] = [
            dim.direction_cosines[0] * sign,
            dim.direction_cosines[1] * sign,
            dim.direction_cosines[2] * sign,
        ];
    }

    let origin = Point::new(origin_arr);
    let spacing = Spacing::new(spacing_arr);

    let direction = Direction::from_columns([
        Vector::new(dir_columns[0]),
        Vector::new(dir_columns[1]),
        Vector::new(dir_columns[2]),
    ]);

    (origin, spacing, direction)
}

#[cfg(test)]
#[path = "tests_spatial.rs"]
mod tests_spatial;
