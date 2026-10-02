//! Whitespace-separated numeric fields of text headers.
//!
//! The parsers reject missing or surplus components and do not include an
//! offending header token in diagnostics.

use anyhow::{anyhow, Context, Result};

/// Parse a whitespace-separated list of exactly `expected` values of `T`.
///
/// # Errors
///
/// Returns an error when a value is invalid or the component count differs
/// from `expected`. Diagnostics name `field` without echoing input tokens.
pub fn parse_floats<T>(s: &str, field: &str, expected: usize) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let mut values = Vec::new();
    for token in s.split_whitespace() {
        if values.len() == expected {
            return Err(anyhow!(
                "'{}' must have exactly {} components",
                field,
                expected
            ));
        }
        values.push(
            token
                .parse::<T>()
                .with_context(|| format!("Invalid value in '{}'.", field))?,
        );
    }
    if values.len() != expected {
        return Err(anyhow!(
            "'{}' must have exactly {} components",
            field,
            expected
        ));
    }
    Ok(values)
}

/// Parse dimension sizes from a text header.
///
/// # Errors
///
/// Returns an error when the field has the wrong number of values or a value
/// is not a `usize`.
pub fn parse_usize_vec(s: &str, field: &str, expected: usize) -> Result<Vec<usize>> {
    parse_floats(s, field, expected)
}

/// Parse spatial coordinates or spacings from a text header.
///
/// # Errors
///
/// Returns an error when the field has the wrong number of values or a value
/// is not an `f64`.
pub fn parse_f64_vec(s: &str, field: &str, expected: usize) -> Result<Vec<f64>> {
    parse_floats(s, field, expected)
}

#[cfg(test)]
#[path = "tests_header_text.rs"]
mod tests;
