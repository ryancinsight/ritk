//! MRtrix `.mif` embedded diffusion gradient scheme.
//!
//! MRtrix stores the acquisition gradient table as a `DW_scheme` key in
//! the text header of its `.mif` container format.  The value is a matrix
//! with four columns per volume: gradient x, y, z (unit direction, or
//! [0,0,0] for b = 0), and b-value in s/mm².
//!
//! The gradient frame is the image/scanner frame — i.e.
//! [`GradientFrame::ImageAxis`].

use ritk_spatial::Vector;

use crate::{
    DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme, GradientSchemeError,
};

/// Read a gradient scheme from a `.mif` header `DW_scheme` block.
///
/// `header_text` is the full `.mif` text header up to and including the
/// `END` marker.  Only the `DW_scheme` lines are consumed.
///
/// # Errors
///
/// Returns [`GradientSchemeError::Empty`] when no `DW_scheme` key is found,
/// a [`GradientSchemeError::InvalidToken`] for unparseable components, or
/// a [`GradientSchemeError::InvalidMrtrixHeader`] for structural violations
/// (wrong column count, dimension mismatch, non-finite or negative values).
pub fn read_mrtrix_scheme(header_text: &str) -> Result<GradientScheme, GradientSchemeError> {
    let matrix = extract_dw_scheme_lines(header_text)?;
    if matrix.is_empty() {
        return Err(GradientSchemeError::InvalidMrtrixHeader(
            "DW_scheme: matrix is empty".to_owned(),
        ));
    }

    let mut pairs = Vec::with_capacity(matrix.len());
    for (index, row) in matrix.iter().enumerate() {
        if row.len() != 4 {
            return Err(GradientSchemeError::InvalidMrtrixHeader(format!(
                "DW_scheme: row {index} has {} columns, expected 4",
                row.len()
            )));
        }
        let value = row[3];
        if !value.is_finite() || value < 0.0 {
            return Err(GradientSchemeError::InvalidToken {
                field: "MRtrix b-value",
                token: format!("{value}"),
            });
        }
        let weighting = DiffusionWeighting::at_index(value, index)?;
        let direction = Vector::new([row[0], row[1], row[2]]);
        pairs.push((weighting, direction));
    }

    let directions = pairs
        .into_iter()
        .enumerate()
        .map(|(index, (weighting, direction))| {
            GradientDirection::at_index(weighting, direction, index)
        })
        .collect::<Result<Vec<_>, _>>()?;
    GradientScheme::new(directions, GradientFrame::ImageAxis)
}

/// Serialise a gradient scheme to `.mif` `DW_scheme` header lines.
///
/// Returns the complete `DW_scheme` block ready for inclusion in a `.mif`
/// text header.  The first line is `DW_scheme: N,4` and each subsequent
/// line encodes one volume as four comma-separated values: gradient x, y, z
/// and b-value in s/mm².
pub fn write_mrtrix_scheme(scheme: &GradientScheme) -> String {
    let n = scheme.len();
    let mut output = format!("DW_scheme: {n},4\n");
    for entry in scheme.directions() {
        let [x, y, z] = entry.direction().to_array();
        let b = entry.weighting().seconds_per_square_millimeter();
        output.push_str(&format!("{x},{y},{z},{b}\n"));
    }
    output
}

/// Extract the numeric matrix from the `DW_scheme` block.
///
/// Scans forward from the line matching `DW_scheme:`, reads the dimension
/// declaration (N,4), then consumes N data lines.
fn extract_dw_scheme_lines(header: &str) -> Result<Vec<Vec<f64>>, GradientSchemeError> {
    let mut lines = header.lines().peekable();

    // Find the DW_scheme key line.
    let dim_line = loop {
        let Some(line) = lines.next() else {
            return Err(GradientSchemeError::InvalidMrtrixHeader(
                "DW_scheme: key not found in header".to_owned(),
            ));
        };
        if let Some(rest) = line.trim().strip_prefix("DW_scheme:") {
            break rest.trim().to_owned();
        }
    };

    // Parse dimensions: "N,4".
    let dim_err = || {
        GradientSchemeError::InvalidMrtrixHeader(format!(
            "DW_scheme: expected 'N,4' dimension declaration, got '{dim_line}'"
        ))
    };
    let parts: Vec<&str> = dim_line.split(',').collect();
    if parts.len() != 2 {
        return Err(dim_err());
    }
    let expected: usize = parts[0]
        .trim()
        .parse()
        .map_err(|_| dim_err())?;
    let columns: usize = parts[1]
        .trim()
        .parse()
        .map_err(|_| dim_err())?;
    if columns != 4 {
        return Err(GradientSchemeError::InvalidMrtrixHeader(format!(
            "DW_scheme: expected 4 columns, got {columns}"
        )));
    }

    // Read N data rows.
    let mut matrix = Vec::with_capacity(expected);
    for row_index in 0..expected {
        let Some(line) = lines.next() else {
            return Err(GradientSchemeError::InvalidMrtrixHeader(format!(
                "DW_scheme: expected {expected} data rows, got {row_index}"
            )));
        };
        let row: Vec<f64> = line
            .trim()
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|token| {
                token.parse::<f64>().map_err(|_| {
                    GradientSchemeError::InvalidToken {
                        field: "MRtrix DW_scheme component",
                        token: token.to_owned(),
                    }
                })
            })
            .collect::<Result<_, _>>()?;
        if row.len() != 4 {
            return Err(GradientSchemeError::InvalidMrtrixHeader(format!(
                "DW_scheme: row {row_index} has {} values, expected 4",
                row.len()
            )));
        }
        matrix.push(row);
    }

    Ok(matrix)
}
