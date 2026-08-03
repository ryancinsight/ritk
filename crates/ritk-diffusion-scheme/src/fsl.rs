//! FSL `.bval` and `.bvec` companion metadata.

use ritk_spatial::Vector;

use crate::{
    DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme, GradientSchemeError,
};

/// Parse whitespace-separated FSL b-values in s/mm².
///
/// # Errors
///
/// Returns a typed error for empty input, a non-numeric token, or a negative
/// or non-finite weighting.
pub fn parse_fsl_bval(contents: &str) -> Result<Vec<DiffusionWeighting>, GradientSchemeError> {
    let tokens = contents.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err(GradientSchemeError::Empty);
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| {
            let value = token
                .parse::<f64>()
                .map_err(|_| GradientSchemeError::InvalidToken {
                    field: "FSL b-value",
                    token: token.to_owned(),
                })?;
            DiffusionWeighting::at_index(value, index)
        })
        .collect()
}

/// Parse a three-row FSL b-vector table.
///
/// # Errors
///
/// Returns a typed error unless there are exactly three nonempty rows with
/// equal finite component counts.
pub fn parse_fsl_bvec(contents: &str) -> Result<Vec<Vector<3>>, GradientSchemeError> {
    let lines = contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.len() != 3 {
        return Err(GradientSchemeError::InvalidBVectorTable(format!(
            "expected three rows, got {}",
            lines.len()
        )));
    }
    let rows = lines
        .iter()
        .map(|line| {
            line.split_whitespace()
                .map(|token| {
                    let value =
                        token
                            .parse::<f64>()
                            .map_err(|_| GradientSchemeError::InvalidToken {
                                field: "FSL b-vector component",
                                token: token.to_owned(),
                            })?;
                    if !value.is_finite() {
                        return Err(GradientSchemeError::InvalidToken {
                            field: "FSL b-vector component",
                            token: token.to_owned(),
                        });
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = rows[0].len();
    if count == 0 || rows[1].len() != count || rows[2].len() != count {
        return Err(GradientSchemeError::InvalidBVectorTable(format!(
            "row lengths are {}, {}, and {}",
            rows[0].len(),
            rows[1].len(),
            rows[2].len()
        )));
    }
    Ok((0..count)
        .map(|index| Vector::new([rows[0][index], rows[1][index], rows[2][index]]))
        .collect())
}

/// Parse paired FSL b-value and b-vector contents.
///
/// The returned directions use [`GradientFrame::ImageAxis`].
///
/// # Errors
///
/// Returns the first parse, count, weighting, or direction validation error.
pub fn read_fsl_scheme(
    bval_contents: &str,
    bvec_contents: &str,
) -> Result<GradientScheme, GradientSchemeError> {
    let weightings = parse_fsl_bval(bval_contents)?;
    let directions = parse_fsl_bvec(bvec_contents)?;
    if weightings.len() != directions.len() {
        return Err(GradientSchemeError::LengthMismatch {
            weightings: weightings.len(),
            directions: directions.len(),
        });
    }
    let entries = weightings
        .into_iter()
        .zip(directions)
        .enumerate()
        .map(|(index, (weighting, direction))| {
            GradientDirection::at_index(weighting, direction, index)
        })
        .collect::<Result<Vec<_>, _>>()?;
    GradientScheme::new(entries, GradientFrame::ImageAxis)
}

/// Serialise a gradient scheme to FSL `.bval` / `.bvec` companion strings.
///
/// Returns `(bval, bvec)` where bval is a single line of space-separated
/// s/mm² values and bvec is a three-line table of direction components.
pub fn write_fsl_scheme(scheme: &GradientScheme) -> (String, String) {
    let mut bval_parts = Vec::with_capacity(scheme.len());
    let mut bvec_x = Vec::with_capacity(scheme.len());
    let mut bvec_y = Vec::with_capacity(scheme.len());
    let mut bvec_z = Vec::with_capacity(scheme.len());

    for entry in scheme.directions() {
        let b = entry.weighting().seconds_per_square_millimeter();
        bval_parts.push(format!("{b}"));
        let [x, y, z] = entry.direction().to_array();
        bvec_x.push(format!("{x}"));
        bvec_y.push(format!("{y}"));
        bvec_z.push(format!("{z}"));
    }

    let bval = bval_parts.join(" ");
    let bvec = format!(
        "{}\n{}\n{}",
        bvec_x.join(" "),
        bvec_y.join(" "),
        bvec_z.join(" ")
    );
    (bval, bvec)
}
