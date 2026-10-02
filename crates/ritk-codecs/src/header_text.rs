//! Whitespace-separated numeric fields of text headers.
//!
//! MetaImage and NRRD headers carry vectors (sizes, spacings, origins,
//! transform matrices) as whitespace-separated tokens; these parsers read a
//! field into exactly the number of components the header's dimension
//! implies.

use anyhow::{anyhow, Context, Result};

/// Parse a whitespace-separated list of exactly `expected` values of `T`.
///
/// Generic over `T: FromStr`.
///
/// # Errors
///
/// Returns an error naming `field` when a token does not parse as `T` (the
/// token is quoted) or when `s` holds more or fewer than `expected` tokens.
pub fn parse_header_values<T>(s: &str, field: &str, expected: usize) -> Result<Vec<T>>
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

#[cfg(test)]
#[path = "tests_header_text.rs"]
mod tests;
