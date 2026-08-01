//! NRRD DWI key/value and measurement-frame decoding.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
use ritk_spatial::Vector;

use super::decode::parse_parenthesized_vectors;

/// Decode a validated gradient scheme from lowercased NRRD header fields.
///
/// The NRRD DWI convention stores one nominal `DWMRI_b-value`; each raw
/// gradient magnitude scales its effective weighting quadratically. The
/// measurement frame maps raw gradient coordinates into the declared world
/// space. RAS world coordinates are converted once to RITK physical LPS.
pub(super) fn scheme_from_headers(headers: &HashMap<String, String>) -> Result<GradientScheme> {
    let modality = required_value(headers, "modality")?;
    if !modality.eq_ignore_ascii_case("DWMRI") {
        bail!("NRRD modality must be DWMRI, got '{modality}'");
    }
    if headers.keys().any(|key| key.starts_with("dwmri_b-matrix_")) {
        bail!("NRRD DWMRI_B-matrix metadata is not supported by the gradient-vector reader");
    }
    if headers.keys().any(|key| key.starts_with("dwmri_nex_")) {
        bail!("NRRD DWMRI_NEX compressed acquisition metadata is not supported");
    }

    let nominal = parse_finite(required_value(headers, "dwmri_b-value")?, "DWMRI_b-value")?;
    if nominal < 0.0 {
        bail!("NRRD DWMRI_b-value must be nonnegative, got {nominal}");
    }

    let mut indexed = headers
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("dwmri_gradient_")
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

    let measurement_frame = parse_measurement_frame(headers.get("measurement frame"))?;
    let ras_to_lps = world_to_lps(required_value(headers, "space")?)?;
    let mut pairs = Vec::with_capacity(indexed.len());
    for (_, raw) in indexed {
        let norm = raw.norm();
        if norm == 0.0 {
            pairs.push((0.0, Vector::new([0.0, 0.0, 0.0])));
            continue;
        }
        let effective = nominal * (norm / maximum_norm).powi(2);
        let unit = raw / norm;
        let world = multiply_columns(measurement_frame, unit);
        let lps = Vector::new([
            ras_to_lps[0] * world[0],
            ras_to_lps[1] * world[1],
            ras_to_lps[2] * world[2],
        ]);
        pairs.push((effective, lps));
    }

    GradientScheme::from_seconds_per_square_millimeter(pairs, GradientFrame::Lps)
        .map_err(anyhow::Error::from)
}

fn required_value<'a>(headers: &'a HashMap<String, String>, key: &str) -> Result<&'a str> {
    headers
        .get(key)
        .map(String::as_str)
        .map(strip_key_value_marker)
        .ok_or_else(|| anyhow!("NRRD DWI header is missing required '{key}' field"))
}

fn strip_key_value_marker(value: &str) -> &str {
    value.strip_prefix('=').unwrap_or(value).trim()
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
    let components = strip_key_value_marker(value)
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

fn world_to_lps(space: &str) -> Result<[f64; 3]> {
    match space.to_ascii_lowercase().as_str() {
        "left-posterior-superior" | "lps" => Ok([1.0, 1.0, 1.0]),
        "right-anterior-superior" | "ras" => Ok([-1.0, -1.0, 1.0]),
        other => bail!(
            "NRRD DWI space '{other}' is unsupported; expected left-posterior-superior or right-anterior-superior"
        ),
    }
}

fn multiply_columns(columns: [[f64; 3]; 3], vector: Vector<3>) -> [f64; 3] {
    let [x, y, z] = vector.to_array();
    [
        columns[0][0] * x + columns[1][0] * y + columns[2][0] * z,
        columns[0][1] * x + columns[1][1] * y + columns[2][1] * z,
        columns[0][2] * x + columns[1][2] * y + columns[2][2] * z,
    ]
}
