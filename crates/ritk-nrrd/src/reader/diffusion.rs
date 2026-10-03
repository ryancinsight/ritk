//! NRRD DWI key/value and measurement-frame decoding.

use std::{collections::HashMap, path::Path};

use anyhow::{anyhow, bail, Context, Result};
use ritk_codecs::parse_usize_vec;
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::Vector;

use super::{
    decode::{parse_parenthesized_vectors, parse_space_direction_slots},
    header::{read_nrrd_header, NrrdHeader},
};
use crate::{
    axes::{locate_acquisition_axis, AcquisitionAxis},
    spatial::{vector_to_lps, world_to_lps_basis},
};

/// Read the diffusion gradient scheme from a NRRD header.
///
/// Extracts `DWMRI_gradient_NNNN` direction keys and `DWMRI_b-value` from the
/// NRRD header and returns a validated [`ritk_diffusion_scheme::GradientScheme`].
/// Directions are mapped through the measurement frame into the declared
/// world space and returned in RITK physical LPS coordinates. Every encoded
/// nonzero effective weighting remains weighted; scanner-input baseline
/// thresholding is not applied while reading stored NRRD metadata.
///
/// # Errors
///
/// Returns an error when the file cannot be opened, the header is missing
/// required DWMRI fields, or the gradient table fails validation.
pub fn read_nrrd_gradient_scheme<P: AsRef<Path>>(
    path: P,
) -> Result<ritk_diffusion_scheme::GradientScheme> {
    let header = read_nrrd_header(path)?;
    scheme_from_header(&header)
}

/// Decode a validated gradient scheme from NRRD fields and key/value metadata.
///
/// The NRRD DWI convention stores one nominal `DWMRI_b-value`; each raw
/// gradient magnitude scales its effective weighting quadratically. The
/// measurement frame maps raw gradient coordinates into the declared world
/// space. RAS world coordinates are converted once to RITK physical LPS.
pub(super) fn scheme_from_header(header: &NrrdHeader) -> Result<GradientScheme> {
    let fields = &header.fields;
    let key_values = &header.key_values;
    let modality = required_value(key_values, "modality")?;
    if !modality.eq_ignore_ascii_case("DWMRI") {
        bail!("NRRD modality must be DWMRI, got '{modality}'");
    }
    if key_values
        .keys()
        .any(|key| key.starts_with("DWMRI_B-matrix_"))
    {
        bail!("NRRD DWMRI_B-matrix metadata is not supported by the gradient-vector reader");
    }
    if key_values.keys().any(|key| key.starts_with("DWMRI_NEX_")) {
        bail!("NRRD DWMRI_NEX compressed acquisition metadata is not supported");
    }

    let nominal = parse_finite(
        required_value(key_values, "DWMRI_b-value")?,
        "DWMRI_b-value",
    )?;
    if nominal < 0.0 {
        bail!("NRRD DWMRI_b-value must be nonnegative, got {nominal}");
    }

    let mut indexed = key_values
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("DWMRI_gradient_")
                .map(|index| (index, value))
        })
        .map(|(index, value)| {
            let index = index
                .parse::<usize>()
                .with_context(|| format!("invalid DWMRI gradient index '{index}'"))?;
            Ok((index, parse_gradient(value)?))
        })
        .collect::<Result<Vec<_>>>()?;
    if indexed.is_empty() {
        bail!("NRRD DWI header has no DWMRI_gradient_NNNN entries");
    }
    indexed.sort_by_key(|(index, _)| *index);
    for (position, (index, _)) in indexed.iter().enumerate() {
        if *index != position {
            bail!(
                "NRRD DWMRI gradient indices must be contiguous from zero: expected {position}, got {index}"
            );
        }
    }
    let (acquisition, volume_count) = acquisition_volume_count(fields)?;
    if let Some(kind) = fields.get("kinds").and_then(|kinds| {
        let axis = match acquisition {
            AcquisitionAxis::Absent => return None,
            AcquisitionAxis::Fastest => 0,
            AcquisitionAxis::Slowest => 3,
        };
        kinds.split_whitespace().nth(axis)
    }) && !kind.eq_ignore_ascii_case("list")
    {
        bail!("NRRD DWI acquisition kind {kind:?} is not supported; expected 'list'");
    }
    if indexed.len() != volume_count {
        bail!(
            "NRRD DWI gradient count {} does not match acquisition-axis extent {volume_count}",
            indexed.len()
        );
    }

    let maximum_norm = indexed
        .iter()
        .map(|(_, vector)| vector.norm())
        .max_by(f64::total_cmp)
        .ok_or_else(|| anyhow!("NRRD DWI header has no gradients"))?;
    if !maximum_norm.is_finite() {
        bail!("NRRD gradient magnitude is not finite");
    }
    if nominal == 0.0 && maximum_norm != 0.0 {
        bail!("NRRD nominal b-value is zero but a gradient vector is nonzero");
    }

    let measurement_frame = parse_measurement_frame(fields.get("measurement frame"))?;
    let world_to_lps = world_to_lps_basis(
        Some(required_value(fields, "space")?),
        fields.get("space dimension").map(String::as_str),
    )?;
    let mut directions = Vec::with_capacity(indexed.len());
    for (index, (_, raw)) in indexed.into_iter().enumerate() {
        let norm = raw.norm();
        if norm == 0.0 {
            let weighting = DiffusionWeighting::from_seconds_per_square_millimeter(0.0)
                .with_context(|| format!("invalid DWMRI weighting at acquisition index {index}"))?;
            directions.push(
                GradientDirection::new(weighting, Vector::new([0.0, 0.0, 0.0])).with_context(
                    || format!("invalid DWMRI direction at acquisition index {index}"),
                )?,
            );
            continue;
        }
        let effective = nominal * (norm / maximum_norm).powi(2);
        let unit = raw / norm;
        let world = multiply_columns(measurement_frame, unit);
        let lps = Vector::new(vector_to_lps(world, world_to_lps)?);
        let weighting = DiffusionWeighting::from_seconds_per_square_millimeter(effective)
            .with_context(|| format!("invalid DWMRI weighting at acquisition index {index}"))?;
        directions
            .push(GradientDirection::new(weighting, lps).with_context(|| {
                format!("invalid DWMRI direction at acquisition index {index}")
            })?);
    }

    GradientScheme::new(directions, GradientFrame::Lps).map_err(anyhow::Error::from)
}

fn acquisition_volume_count(headers: &HashMap<String, String>) -> Result<(AcquisitionAxis, usize)> {
    let dimension = required_value(headers, "dimension")?
        .parse::<usize>()
        .context("NRRD DWI 'dimension' is not a valid integer")?;
    if dimension != 4 {
        bail!("NRRD DWI metadata requires dimension 4, got {dimension}");
    }
    let sizes = parse_usize_vec(required_value(headers, "sizes")?, "sizes", dimension)?;
    let direction_slots = headers
        .get("space directions")
        .map(|value| parse_space_direction_slots(value))
        .transpose()?;
    let direction_flags = direction_slots
        .as_ref()
        .map(|slots| slots.iter().map(Option::is_some).collect::<Vec<_>>());
    let acquisition = locate_acquisition_axis(
        dimension,
        headers.get("kinds").map(String::as_str),
        direction_flags.as_deref(),
    )?;
    let count = match acquisition {
        AcquisitionAxis::Fastest => sizes[0],
        AcquisitionAxis::Slowest => sizes[3],
        AcquisitionAxis::Absent => {
            bail!("NRRD DWI dimension 4 has no declared acquisition axis")
        }
    };
    if count == 0 {
        bail!("NRRD DWI acquisition axis must contain at least one volume");
    }
    Ok((acquisition, count))
}

fn required_value<'a>(headers: &'a HashMap<String, String>, key: &str) -> Result<&'a str> {
    headers
        .get(key)
        .map(String::as_str)
        .map(str::trim)
        .ok_or_else(|| anyhow!("NRRD DWI header is missing required '{key}' field"))
}

fn parse_finite(value: &str, field: &str) -> Result<f64> {
    let parsed = value
        .parse::<f64>()
        .with_context(|| format!("cannot parse {field} value '{value}'"))?;
    if !parsed.is_finite() {
        bail!("{field} must be finite, got {parsed}");
    }
    Ok(parsed)
}

fn parse_gradient(value: &str) -> Result<Vector<3>> {
    let components = value
        .split_whitespace()
        .map(|token| parse_finite(token, "DWMRI gradient component"))
        .collect::<Result<Vec<_>>>()?;
    let components: [f64; 3] = components.try_into().map_err(|values: Vec<f64>| {
        anyhow!(
            "DWMRI gradient must contain 3 components, got {}",
            values.len()
        )
    })?;
    Ok(Vector::new(components))
}

fn parse_measurement_frame(value: Option<&String>) -> Result<[[f64; 3]; 3]> {
    let Some(value) = value else {
        return Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    };
    let columns = parse_parenthesized_vectors(value)?;
    let columns: [[f64; 3]; 3] = columns.try_into().map_err(|values: Vec<[f64; 3]>| {
        anyhow!(
            "NRRD measurement frame must contain 3 column vectors, got {}",
            values.len()
        )
    })?;
    if columns.iter().flatten().any(|value| !value.is_finite()) {
        bail!("NRRD measurement frame contains a non-finite component");
    }
    Ok(columns)
}

fn multiply_columns(columns: [[f64; 3]; 3], vector: Vector<3>) -> [f64; 3] {
    let [x, y, z] = vector.to_array();
    [
        columns[0][0] * x + columns[1][0] * y + columns[2][0] * z,
        columns[0][1] * x + columns[1][1] * y + columns[2][1] * z,
        columns[0][2] * x + columns[1][2] * y + columns[2][2] * z,
    ]
}
