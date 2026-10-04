//! NRRD file-space conversion for RITK's internal ZYX image axes.
//!
//! RITK stores tensors and spatial metadata in `[depth,row,col] = [z,y,x]`
//! order. NRRD `sizes` and `space directions` fields list file axes as
//! `[x,y,z]`. Therefore the NRRD file vectors and internal metadata columns
//! differ only by column order:
//!
//! ```text
//! A_internal[:, depth] = A_nrrd[:, z]
//! A_internal[:, row]   = A_nrrd[:, y]
//! A_internal[:, col]   = A_nrrd[:, x]
//! ```

use anyhow::{anyhow, bail, Context, Result};
use ritk_spatial::{Direction, Spacing, Vector};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InternalSpatialMetadata {
    pub(crate) spacing: Spacing<3>,
    pub(crate) direction: Direction<3>,
}

/// Convert NRRD physical coordinates into RITK's canonical LPS millimeters.
///
/// NRRD names the coordinate basis independently of the image-axis order.
/// Files without a named space or units retain RITK's historical LPS/mm
/// interpretation. Anonymous or scanner coordinate systems cannot be mapped
/// to patient LPS without additional registration data.
pub(crate) fn world_to_lps_factors(
    space: Option<&str>,
    space_dimension: Option<&str>,
    units: Option<&str>,
) -> Result<[f64; 3]> {
    let basis = world_to_lps_basis(space, space_dimension)?;
    let millimeters = parse_space_units(units)?;
    Ok(std::array::from_fn(|axis| basis[axis] * millimeters[axis]))
}

/// Convert named NRRD world-axis directions into the LPS basis without
/// applying physical-coordinate units.
pub(crate) fn world_to_lps_basis(
    space: Option<&str>,
    space_dimension: Option<&str>,
) -> Result<[f64; 3]> {
    if space.is_some() && space_dimension.is_some() {
        bail!("NRRD orientation cannot declare both 'space' and 'space dimension'");
    }
    if let Some(dimension) = space_dimension {
        bail!("NRRD anonymous 'space dimension' {dimension:?} cannot be converted to patient LPS");
    }
    let signs = match space.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("left-posterior-superior") | Some("lps") => [1.0, 1.0, 1.0],
        Some("right-anterior-superior") | Some("ras") => [-1.0, -1.0, 1.0],
        Some("left-anterior-superior") | Some("las") => [1.0, -1.0, 1.0],
        Some(other) => {
            bail!("NRRD space '{other}' cannot be converted to patient LPS")
        }
    };
    Ok(signs)
}

/// Parse three quoted NRRD physical-coordinate units into millimeter scales.
fn parse_space_units(units: Option<&str>) -> Result<[f64; 3]> {
    let Some(units) = units else {
        return Ok([1.0; 3]);
    };
    let tokens = parse_quoted_strings(units, "space units", 3)?;
    let scales = tokens
        .iter()
        .map(|unit| millimeters_per_unit(unit, "space units"))
        .collect::<Result<Vec<_>>>()?;
    scales
        .try_into()
        .map_err(|values: Vec<f64>| anyhow!("expected 3 NRRD space units, found {}", values.len()))
}

/// Parse per-axis NRRD units into optional millimeter scales.
///
/// Empty unit strings carry no scale. Non-empty values must name one of the
/// supported metric length units. Other axis dimensions and unit systems have
/// no representation in image geometry and are rejected by the caller.
pub(crate) fn parse_axis_units(units: &str, dimension: usize) -> Result<Vec<Option<f64>>> {
    parse_quoted_strings(units, "units", dimension)?
        .iter()
        .map(|unit| {
            if unit.is_empty() {
                Ok(None)
            } else {
                millimeters_per_unit(unit, "units").map(Some)
            }
        })
        .collect()
}

/// Parses per-axis sample centering and reports whether each axis is cell- or
/// node-centered.
pub(crate) fn parse_axis_centerings(value: &str, dimension: usize) -> Result<Vec<bool>> {
    parse_quoted_strings(value, "centers", dimension)?
        .iter()
        .map(|centering| match centering.to_ascii_lowercase().as_str() {
            "cell" | "node" => Ok(true),
            "none" | "???" => Ok(false),
            _ => bail!("NRRD centers value {centering:?} is unsupported"),
        })
        .collect()
}

fn parse_quoted_strings(value: &str, field: &str, expected: usize) -> Result<Vec<String>> {
    let mut characters = value.chars().peekable();
    let mut strings = Vec::with_capacity(expected);
    loop {
        while matches!(characters.peek(), Some(' ' | '\t')) {
            characters.next();
        }
        if characters.peek().is_none() {
            break;
        }
        if strings.len() == expected {
            bail!("NRRD {field} contains more than {expected} quoted values");
        }
        if characters.next() != Some('"') {
            bail!("NRRD {field} values must be quoted");
        }
        let mut string = String::new();
        let mut closed = false;
        while let Some(character) = characters.next() {
            match character {
                '"' => {
                    closed = true;
                    break;
                }
                '\\' => match characters.next() {
                    Some('"') => string.push('"'),
                    _ => bail!("NRRD {field} only permits escaped double quotes"),
                },
                value => string.push(value),
            }
        }
        if !closed {
            bail!("NRRD {field} contains an unterminated quoted value");
        }
        if characters
            .peek()
            .is_some_and(|character| !matches!(character, ' ' | '\t'))
        {
            bail!("NRRD {field} quoted values must be separated by horizontal whitespace");
        }
        strings.push(string);
    }
    if strings.len() != expected {
        bail!(
            "NRRD {field} must contain {expected} quoted values, found {}",
            strings.len()
        );
    }
    Ok(strings)
}

fn millimeters_per_unit(unit: &str, field: &str) -> Result<f64> {
    match unit {
        "mm" => Ok(1.0),
        "cm" => Ok(10.0),
        "m" => Ok(1_000.0),
        "um" => Ok(0.001),
        "nm" => Ok(0.000_001),
        _ => bail!("NRRD {field} unit {unit:?} cannot be converted to millimeters"),
    }
}

/// Apply a world-axis scale and handedness transform to a coordinate vector.
pub(crate) fn vector_to_lps(vector: [f64; 3], factors: [f64; 3]) -> Result<[f64; 3]> {
    let converted: [f64; 3] = std::array::from_fn(|axis| vector[axis] * factors[axis]);
    if let Some(axis) =
        vector
            .iter()
            .zip(converted)
            .enumerate()
            .find_map(|(axis, (source, value))| {
                (!value.is_finite() || (*source != 0.0 && value == 0.0)).then_some(axis)
            })
    {
        bail!("NRRD world-coordinate conversion makes vector component {axis} unrepresentable");
    }
    Ok(converted)
}

/// Apply the world-to-LPS transform to NRRD file-axis direction vectors.
pub(crate) fn directions_to_lps(
    file_vectors: [[f64; 3]; 3],
    factors: [f64; 3],
) -> Result<[[f64; 3]; 3]> {
    Ok([
        vector_to_lps(file_vectors[0], factors)?,
        vector_to_lps(file_vectors[1], factors)?,
        vector_to_lps(file_vectors[2], factors)?,
    ])
}

/// Convert NRRD `[x,y,z]` space-direction vectors into RITK internal
/// `[depth,row,col]` spacing and direction columns.
///
/// # Errors
///
/// Returns an error when a direction vector has zero or non-finite length.
/// `Spacing` requires every component to be finite and strictly positive and
/// asserts it, so a degenerate vector from a corrupt `space directions` field
/// would abort the process rather than fail the read.
pub(crate) fn metadata_from_file_space_directions(
    file_vectors: [[f64; 3]; 3],
) -> Result<InternalSpatialMetadata> {
    let scaled_columns = [
        vector_from_array(file_vectors[2]),
        vector_from_array(file_vectors[1]),
        vector_from_array(file_vectors[0]),
    ];

    metadata_from_internal_scaled_columns(scaled_columns)
}

/// Promote two NRRD image-axis vectors in 3-D world space to one-slice volume
/// metadata. The missing slice axis uses a one-millimeter step along the
/// normalized plane normal.
pub(crate) fn metadata_from_planar_file_space_directions(
    file_vectors: [[f64; 3]; 2],
    world_to_lps: [f64; 3],
) -> Result<InternalSpatialMetadata> {
    let file_x = Vector::new(vector_to_lps(file_vectors[0], world_to_lps)?);
    let file_y = Vector::new(vector_to_lps(file_vectors[1], world_to_lps)?);
    let unit_x = file_x
        .normalized()
        .ok_or_else(|| anyhow!("NRRD rank-2 X direction is zero or non-finite"))?;
    let unit_y = file_y
        .normalized()
        .ok_or_else(|| anyhow!("NRRD rank-2 Y direction is zero or non-finite"))?;
    let normal = unit_x
        .cross(&unit_y)
        .normalized()
        .ok_or_else(|| anyhow!("NRRD rank-2 directions do not define a physical plane"))?;

    metadata_from_file_space_directions([file_x.to_array(), file_y.to_array(), normal.to_array()])
}

/// Build NRRD `[x,y,z]` space-direction vectors from RITK internal
/// `[depth,row,col]` metadata.
pub(crate) fn file_space_directions_from_internal(
    spacing: [f64; 3],
    direction_row_major: [f64; 9],
) -> [[f64; 3]; 3] {
    let internal_columns = [
        scaled_direction_column(direction_row_major, spacing, 0),
        scaled_direction_column(direction_row_major, spacing, 1),
        scaled_direction_column(direction_row_major, spacing, 2),
    ];

    [
        internal_columns[2],
        internal_columns[1],
        internal_columns[0],
    ]
}

fn metadata_from_internal_scaled_columns(
    scaled_columns: [Vector<3>; 3],
) -> Result<InternalSpatialMetadata> {
    let spacing = Spacing::try_new([
        scaled_columns[0].norm(),
        scaled_columns[1].norm(),
        scaled_columns[2].norm(),
    ])
    .context("NRRD spatial metadata does not describe a physical grid")?;

    let direction_columns = [
        normalized_physical_axis(scaled_columns[0], spacing[0])?,
        normalized_physical_axis(scaled_columns[1], spacing[1])?,
        normalized_physical_axis(scaled_columns[2], spacing[2])?,
    ];
    let direction = Direction::from_columns(direction_columns);
    let determinant = direction.determinant();
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(anyhow!("NRRD physical grid direction matrix is singular"));
    }

    Ok(InternalSpatialMetadata { spacing, direction })
}

fn normalized_physical_axis(scaled_axis: Vector<3>, length: f64) -> Result<Vector<3>> {
    let components = scaled_axis.to_array();
    let normalized = std::array::from_fn(|index| components[index] / length);
    if components
        .into_iter()
        .zip(normalized)
        .any(|(component, direction)| component != 0.0 && direction == 0.0)
    {
        bail!("NRRD physical grid direction loses a nonzero component when normalized");
    }
    Ok(Vector::new(normalized))
}

fn vector_from_array(value: [f64; 3]) -> Vector<3> {
    Vector::new(value)
}

fn scaled_direction_column(
    direction_row_major: [f64; 9],
    spacing: [f64; 3],
    column: usize,
) -> [f64; 3] {
    [
        direction_row_major[column] * spacing[column],
        direction_row_major[3 + column] * spacing[column],
        direction_row_major[6 + column] * spacing[column],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn assert_close(got: f64, expected: f64) {
        assert!(
            (got - expected).abs() < EPS,
            "got {got:.12}, expected {expected:.12}"
        );
    }

    #[test]
    fn file_space_directions_are_reordered_into_internal_axes() {
        let metadata = metadata_from_file_space_directions([
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        .expect("valid direction vectors");

        assert_close(metadata.spacing[0], 2.0);
        assert_close(metadata.spacing[1], 3.0);
        assert_close(metadata.spacing[2], 4.0);

        let expected = Direction::from_row_major([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(metadata.direction, expected);
    }

    #[test]
    fn internal_metadata_columns_are_reordered_into_file_axes() {
        let directions = file_space_directions_from_internal(
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        );

        assert_eq!(directions[0], [4.0, 0.0, 0.0]);
        assert_eq!(directions[1], [0.0, 3.0, 0.0]);
        assert_eq!(directions[2], [0.0, 0.0, 2.0]);
    }

    #[test]
    fn patient_spaces_and_units_normalize_to_lps_millimeters() {
        assert_eq!(
            world_to_lps_factors(Some("RAS"), None, Some("\"cm\" \"cm\" \"cm\""))
                .expect("supported RAS units"),
            [-10.0, -10.0, 10.0]
        );
        assert_eq!(
            world_to_lps_factors(Some("left-anterior-superior"), None, None)
                .expect("supported LAS basis"),
            [1.0, -1.0, 1.0]
        );
        assert_eq!(
            world_to_lps_basis(Some("RAS"), None).expect("supported RAS basis"),
            [-1.0, -1.0, 1.0]
        );
        assert_eq!(
            vector_to_lps([1.0, 2.0, 3.0], [-10.0, -10.0, 10.0])
                .expect("finite coordinate transform"),
            [-10.0, -20.0, 30.0]
        );
    }

    #[test]
    fn coordinate_unit_conversion_rejects_lost_nonzero_components() {
        let error = vector_to_lps([1.0, f64::from_bits(1), 0.0], [1.0e-6, 1.0e-6, 1.0e-6])
            .expect_err("unit conversion cannot erase a nonzero direction component");

        assert_eq!(
            error.to_string(),
            "NRRD world-coordinate conversion makes vector component 1 unrepresentable"
        );
    }

    #[test]
    fn unsupported_space_and_units_are_rejected() {
        assert!(world_to_lps_factors(Some("scanner-xyz"), None, None).is_err());
        assert!(world_to_lps_factors(Some("LPS"), Some("3"), None).is_err());
        assert!(world_to_lps_factors(None, Some("3"), None).is_err());
        assert!(world_to_lps_factors(Some("RAS"), None, Some("\"parsec\" \"mm\" \"mm\"")).is_err());
        assert!(world_to_lps_factors(Some("LPS"), None, Some("mm mm mm")).is_err());
        assert!(world_to_lps_factors(Some("LPS"), None, Some("\"mm\" \"mm\"")).is_err());
    }

    #[test]
    fn singular_file_directions_are_rejected() {
        assert!(metadata_from_file_space_directions([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        .is_err());
    }

    #[test]
    fn normalization_rejects_lost_physical_axis_components() {
        let error = metadata_from_file_space_directions([
            [1e308, 1e-308, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        .expect_err("a nonzero physical-axis component cannot disappear");

        assert!(error.to_string().contains("loses a nonzero component"));
    }
}
