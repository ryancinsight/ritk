//! Shared byte-decoding and whitespace-parse helpers.
//!
//! Single source of truth for the low-level format-agnostic primitives that
//! were previously duplicated across [`ritk-metaimage`], [`ritk-nrrd`],
//! [`ritk-vtk`], and [`ritk-minc`]:
//!
//! - [`ByteOrder`] â€” multi-byte element byte-order discriminant.
//! - [`decode_bytes_to_f32`] â€” convert a raw byte buffer to `Vec<f32>` given
//!   element size, signedness, and float-or-integer flavour. Used by every
//!   format reader that targets RITK's `f32` tensor contract.
//! - [`require_bytes`] â€” bounds check for the above.
//! - [`parse_floats`] â€” whitespace-separated list of `FromStr`-parseable
//!   scalars with length validation. Used for header / metadata parsing.
//! - [`parse_usize_vec`] â€” `parse_floats` specialised to `usize`.
//!
//! [`ritk-metaimage`]: https://docs.rs/ritk-metaimage
//! [`ritk-nrrd`]: https://docs.rs/ritk-nrrd
//! [`ritk-vtk`]: https://docs.rs/ritk-vtk
//! [`ritk-minc`]: https://docs.rs/ritk-minc

use anyhow::{anyhow, Context, Result};

/// Byte order for multi-byte element data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Most-significant byte first (big-endian).
    MostSignificantByteFirst,
    /// Least-significant byte first (little-endian).
    LeastSignificantByteFirst }

impl ByteOrder {
    /// Parse the textual byte-order markers used by MetaImage (`"True"` /
    /// `"False"`) and NRRD (`"big"` / `"little"`). Unknown values fall back
    /// to little-endian to preserve the pre-refactor default.
    pub fn from_metaimage_msb(value: &str) -> Self {
        if value.eq_ignore_ascii_case("TRUE") {
            Self::MostSignificantByteFirst
        } else {
            Self::LeastSignificantByteFirst
        }
    }

    /// NRRD byte-order string per the NRRD spec Â§3.5: only `"big"` or
    /// `"little"` (case-insensitive, leading/trailing whitespace allowed).
    /// Unknown values default to [`Self::LeastSignificantByteFirst`] to
    /// preserve the pre-refactor behavior of silently accepting
    /// unspecified / misspelled byte-order strings as little-endian.
    pub fn from_nrrd(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "big" => Self::MostSignificantByteFirst,
            _ => Self::LeastSignificantByteFirst }
    }
}

/// Verify that a buffer has at least `count * elem_size` bytes available.
///
/// Returns a context-bearing error otherwise; the type name is included in
/// the message for easier debugging.
pub fn require_bytes(have: usize, count: usize, elem_size: usize, type_name: &str) -> Result<()> {
    let need = count * elem_size;
    if have < need {
        Err(anyhow!(
            "Insufficient data for type '{}': need {} bytes, got {}",
            type_name,
            need,
            have
        ))
    } else {
        Ok(())
    }
}

/// Decode a raw byte buffer into `Vec<f32>`.
///
/// Generic over the (element size, signedness, float-or-integer) triple so
/// format readers can translate their type-name strings into a single
/// parameter pack. Exactly `count` values are produced; surplus bytes are
/// ignored.
///
/// # Errors
/// - Buffer is shorter than `count * elem_size` (delegated to
///   [`require_bytes`])
/// - `is_float` is true but `elem_size` is not 4 or 8
/// - `is_float` is false but `elem_size` is not 1, 2, 4, or 8
pub fn decode_bytes_to_f32(
    bytes: &[u8],
    elem_size: usize,
    signed: bool,
    is_float: bool,
    byte_order: ByteOrder,
    count: usize,
    type_name: &str,
) -> Result<Vec<f32>> {
    require_bytes(bytes.len(), count, elem_size, type_name)?;
    let be = byte_order == ByteOrder::MostSignificantByteFirst;

    if is_float {
        match elem_size {
            4 => Ok(bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    if be {
                        f32::from_be_bytes(b)
                    } else {
                        f32::from_le_bytes(b)
                    }
                })
                .collect()),
            8 => Ok(bytes
                .chunks_exact(8)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    let v = if be {
                        f64::from_be_bytes(b)
                    } else {
                        f64::from_le_bytes(b)
                    };
                    v as f32
                })
                .collect()),
            other => Err(anyhow!(
                "Unsupported float element size {} for type '{}'",
                other,
                type_name
            )) }
    } else {
        // Integer path
        Ok(match (elem_size, signed) {
            (1, false) => bytes[..count].iter().map(|&b| b as f32).collect(),
            (1, true) => bytes[..count].iter().map(|&b| (b as i8) as f32).collect(),
            (2, true) => bytes
                .chunks_exact(2)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1]];
                    let v = if be {
                        i16::from_be_bytes(b)
                    } else {
                        i16::from_le_bytes(b)
                    };
                    v as f32
                })
                .collect(),
            (2, false) => bytes
                .chunks_exact(2)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1]];
                    let v = if be {
                        u16::from_be_bytes(b)
                    } else {
                        u16::from_le_bytes(b)
                    };
                    v as f32
                })
                .collect(),
            (4, true) => bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    let v = if be {
                        i32::from_be_bytes(b)
                    } else {
                        i32::from_le_bytes(b)
                    };
                    v as f32
                })
                .collect(),
            (4, false) => bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    let v = if be {
                        u32::from_be_bytes(b)
                    } else {
                        u32::from_le_bytes(b)
                    };
                    v as f32
                })
                .collect(),
            (8, true) => bytes
                .chunks_exact(8)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    let v = if be {
                        i64::from_be_bytes(b)
                    } else {
                        i64::from_le_bytes(b)
                    };
                    v as f32
                })
                .collect(),
            (8, false) => bytes
                .chunks_exact(8)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    let v = if be {
                        u64::from_be_bytes(b)
                    } else {
                        u64::from_le_bytes(b)
                    };
                    v as f32
                })
                .collect(),
            (other, _) => {
                return Err(anyhow!(
                    "Unsupported integer element size {} for type '{}'",
                    other,
                    type_name
                ));
            }
        })
    }
}

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

/// `parse_floats` specialised to `usize` â€” common case in header parsers
/// (dimension sizes, component counts). Saves the per-call-site turbofish.
pub fn parse_usize_vec(s: &str, field: &str, expected: usize) -> Result<Vec<usize>> {
    parse_floats(s, field, expected)
}

/// `parse_floats` specialised to `f64` â€” common case in header parsers
/// (spacings, origins, transform-matrix entries). Saves the per-call-site
/// turbofish.
pub fn parse_f64_vec(s: &str, field: &str, expected: usize) -> Result<Vec<f64>> {
    parse_floats(s, field, expected)
}

#[cfg(test)]
#[path = "tests_byte_decode.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_byte_decode_int.rs"]
mod tests_int;
