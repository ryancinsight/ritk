//! Whitespace-separated numeric fields of text headers.
//!
//! MetaImage and NRRD headers carry vectors (sizes, spacings, origins,
//! transform matrices) as whitespace-separated tokens; these parsers read a
//! field into exactly the number of components the header's dimension
//! implies.

use anyhow::{anyhow, Context, Result};

/// Parse a whitespace-separated list of exactly `expected` values of `T`.
///
/// Generic over `T: FromStr`; the error is annotated with the field name and
/// the offending token for easier debugging. Surplus or deficit tokens both
/// return an error.
pub fn parse_floats<T>(s: &str, field: &str, expected: usize) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let vals: Vec<T> = s
        .split_whitespace()
        .map(|t| {
            t.parse::<T>()
                .with_context(|| format!("Invalid value in '{}': '{}'", field, t))
        })
        .collect::<Result<Vec<_>>>()?;

    if vals.len() != expected {
        return Err(anyhow!(
            "'{}' must have {} components, got {}",
            field,
            expected,
            vals.len()
        ));
    }
    Ok(vals)
}

/// `parse_floats` specialised to `usize` — common case in header parsers
/// (dimension sizes, component counts). Saves the per-call-site turbofish.
pub fn parse_usize_vec(s: &str, field: &str, expected: usize) -> Result<Vec<usize>> {
    parse_floats(s, field, expected)
}

/// `parse_floats` specialised to `f64` — common case in header parsers
/// (spacings, origins, transform-matrix entries). Saves the per-call-site
/// turbofish.
pub fn parse_f64_vec(s: &str, field: &str, expected: usize) -> Result<Vec<f64>> {
    parse_floats(s, field, expected)
}

#[cfg(test)]
#[path = "tests_header_text.rs"]
mod tests;
