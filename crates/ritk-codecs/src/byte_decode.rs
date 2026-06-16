//! Shared byte-decoding and whitespace-parse helpers.
//!
//! Single source of truth for the low-level format-agnostic primitives that
//! were previously duplicated across [`ritk-metaimage`], [`ritk-nrrd`],
//! [`ritk-vtk`], and [`ritk-minc`]:
//!
//! - [`ByteOrder`] — multi-byte element byte-order discriminant.
//! - [`decode_bytes_to_f32`] — convert a raw byte buffer to `Vec<f32>` given
//!   element size, signedness, and float-or-integer flavour. Used by every
//!   format reader that targets RITK's `f32` tensor contract.
//! - [`require_bytes`] — bounds check for the above.
//! - [`parse_floats`] — whitespace-separated list of `FromStr`-parseable
//!   scalars with length validation. Used for header / metadata parsing.
//! - [`parse_usize_vec`] — `parse_floats` specialised to `usize`.
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
    LeastSignificantByteFirst,
}

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

    /// NRRD byte-order string per the NRRD spec §3.5: only `"big"` or
    /// `"little"` (case-insensitive, leading/trailing whitespace allowed).
    /// Unknown values default to [`Self::LeastSignificantByteFirst`] to
    /// preserve the pre-refactor behavior of silently accepting
    /// unspecified / misspelled byte-order strings as little-endian.
    pub fn from_nrrd(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "big" => Self::MostSignificantByteFirst,
            _ => Self::LeastSignificantByteFirst,
        }
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
            )),
        }
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
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── ByteOrder::from_nrrd ──────────────────────────────────────────────

    #[test]
    fn from_nrrd_big_is_most_significant_byte_first() {
        assert_eq!(
            ByteOrder::from_nrrd("big"),
            ByteOrder::MostSignificantByteFirst
        );
    }

    #[test]
    fn from_nrrd_little_is_least_significant_byte_first() {
        assert_eq!(
            ByteOrder::from_nrrd("little"),
            ByteOrder::LeastSignificantByteFirst
        );
    }

    #[test]
    fn from_nrrd_is_case_insensitive() {
        assert_eq!(
            ByteOrder::from_nrrd("BIG"),
            ByteOrder::MostSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_nrrd("Little"),
            ByteOrder::LeastSignificantByteFirst
        );
    }

    #[test]
    fn from_nrrd_trims_whitespace() {
        assert_eq!(
            ByteOrder::from_nrrd("  big  "),
            ByteOrder::MostSignificantByteFirst
        );
    }

    #[test]
    fn from_nrrd_unknown_defaults_to_little_endian() {
        // Pre-refactor behavior: unknown / misspelled values fall back to
        // little-endian silently. Non-spec aliases ("msbfirst",
        // "mostsignificantbytefirst") are also rejected to this default.
        assert_eq!(
            ByteOrder::from_nrrd("msbfirst"),
            ByteOrder::LeastSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_nrrd("mostsignificantbytefirst"),
            ByteOrder::LeastSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_nrrd(""),
            ByteOrder::LeastSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_nrrd("middle"),
            ByteOrder::LeastSignificantByteFirst
        );
    }

    #[test]
    fn from_nrrd_embedded_whitespace_defaults_to_little_endian() {
        // "bi g" / "lit tle" contain spaces INSIDE the token — not removed
        // by the outer `trim()` — so the match arm falls through to LSB.
        // This pins the strict-equality behavior: only the trimmed whole
        // string is matched, not any substring.
        assert_eq!(
            ByteOrder::from_nrrd("bi g"),
            ByteOrder::LeastSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_nrrd("lit tle"),
            ByteOrder::LeastSignificantByteFirst
        );
    }

    #[test]
    fn from_nrrd_partial_match_defaults_to_little_endian() {
        // "bigg" / "littl" are NOT exact matches for "big" / "little" — the
        // match arm falls through to the LSB default. This documents the
        // exact-match contract: typos do not silently coerce to MSB.
        assert_eq!(
            ByteOrder::from_nrrd("bigg"),
            ByteOrder::LeastSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_nrrd("littl"),
            ByteOrder::LeastSignificantByteFirst
        );
    }

    // ── ByteOrder::from_metaimage_msb ─────────────────────────────────────

    #[test]
    fn from_metaimage_msb_true_is_most_significant_byte_first() {
        assert_eq!(
            ByteOrder::from_metaimage_msb("True"),
            ByteOrder::MostSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_metaimage_msb("TRUE"),
            ByteOrder::MostSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_metaimage_msb("true"),
            ByteOrder::MostSignificantByteFirst
        );
    }

    #[test]
    fn from_metaimage_msb_false_or_unknown_is_little_endian() {
        assert_eq!(
            ByteOrder::from_metaimage_msb("False"),
            ByteOrder::LeastSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_metaimage_msb("FALSE"),
            ByteOrder::LeastSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_metaimage_msb(""),
            ByteOrder::LeastSignificantByteFirst
        );
    }

    #[test]
    fn from_metaimage_msb_mixed_case() {
        // Case-insensitive match: "tRuE" / "tRue" resolve to MSB;
        // "fAlSe" / "fALse" resolve to LSB.
        assert_eq!(
            ByteOrder::from_metaimage_msb("tRuE"),
            ByteOrder::MostSignificantByteFirst
        );
        assert_eq!(
            ByteOrder::from_metaimage_msb("fAlSe"),
            ByteOrder::LeastSignificantByteFirst
        );
    }

    #[test]
    fn from_metaimage_msb_non_bool_string_defaults_to_little_endian() {
        // Anything other than "True" (case-insensitive) resolves to LSB —
        // matches the pre-refactor MetaImage behavior of treating any
        // non-"True" value as little-endian. Covers common synonyms /
        // 1/0 / yes-no that someone might mistakenly pass.
        for v in ["yes", "1", "on", "true ", " true", "True.", "big"] {
            assert_eq!(
                ByteOrder::from_metaimage_msb(v),
                ByteOrder::LeastSignificantByteFirst,
                "expected LSB for non-bool input {v:?}"
            );
        }
    }

    // ── parse_floats round-trip ──────────────────────────────────────────

    #[test]
    fn parse_floats_f64_round_trip() {
        // Use the full-precision f64 string representation of -PI so that
        // parsing the string recovers the exact same f64 bit pattern.
        let s = format!("1.0 2.5 {} 4.0", -PI);
        let v: Vec<f64> = parse_floats(&s, "test_field", 4).unwrap();
        assert_eq!(v, vec![1.0, 2.5, -PI, 4.0]);
    }

    #[test]
    fn parse_floats_usize_round_trip() {
        let s = "10 20 30";
        let v: Vec<usize> = parse_floats(s, "sizes", 3).unwrap();
        assert_eq!(v, vec![10, 20, 30]);
    }

    #[test]
    fn parse_floats_length_mismatch_returns_error() {
        let s = "1.0 2.0";
        let err = parse_floats::<f64>(s, "field", 3).unwrap_err();
        assert!(
            err.to_string().contains("must have 3 components, got 2"),
            "expected length-mismatch error, got: {err:#}"
        );
    }

    #[test]
    fn parse_floats_bad_token_returns_error() {
        let s = "1.0 not_a_number 3.0";
        let err = parse_floats::<f64>(s, "field", 3).unwrap_err();
        assert!(
            err.to_string().contains("not_a_number"),
            "expected error to mention the offending token, got: {err:#}"
        );
    }

    // ── require_bytes ────────────────────────────────────────────────────

    #[test]
    fn require_bytes_ok_when_buffer_is_large_enough() {
        assert!(require_bytes(100, 10, 4, "f32").is_ok());
    }

    #[test]
    fn require_bytes_err_when_buffer_is_too_short() {
        let err = require_bytes(10, 10, 4, "f32").unwrap_err();
        assert!(
            err.to_string().contains("need 40 bytes, got 10"),
            "expected explicit byte-count error, got: {err:#}"
        );
    }

    // ── decode_bytes_to_f32 round-trip (one combo per branch) ─────────────

    #[test]
    fn decode_u8_le_round_trips_to_f32() {
        let bytes = vec![0u8, 1, 127, 255];
        let out = decode_bytes_to_f32(
            &bytes,
            1,
            false,
            false,
            ByteOrder::LeastSignificantByteFirst,
            4,
            "uint8",
        )
        .unwrap();
        assert_eq!(out, vec![0.0, 1.0, 127.0, 255.0]);
    }

    #[test]
    fn decode_i16_be_round_trips_to_f32() {
        let raw: Vec<u8> = [1_i16, -1, 32767, -32768]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            2,
            true,
            false,
            ByteOrder::MostSignificantByteFirst,
            4,
            "int16",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, -1.0, 32767.0, -32768.0]);
    }

    #[test]
    fn decode_f32_le_round_trips() {
        let raw: Vec<u8> = [0.0_f32, 1.0, -1.0, std::f32::consts::PI]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            4,
            false,
            true,
            ByteOrder::LeastSignificantByteFirst,
            4,
            "float",
        )
        .unwrap();
        for (i, expected) in [0.0_f32, 1.0, -1.0, std::f32::consts::PI]
            .iter()
            .enumerate()
        {
            assert_eq!(out[i].to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn decode_f64_be_narrows_to_f32() {
        let raw: Vec<u8> = [1.5_f64, -2.5]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            8,
            false,
            true,
            ByteOrder::MostSignificantByteFirst,
            2,
            "double",
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.5_f32).abs() < 1e-6);
        assert!((out[1] - -2.5_f32).abs() < 1e-6);
    }

    #[test]
    fn decode_unsupported_element_size_returns_error() {
        let bytes = vec![0u8; 12];
        let err = decode_bytes_to_f32(
            &bytes,
            3,
            false,
            false,
            ByteOrder::LeastSignificantByteFirst,
            4,
            "weird",
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Unsupported integer element size 3"),
            "expected explicit error, got: {err:#}"
        );
    }

    // ── decode_bytes_to_f32 full matrix (8 signed + 8 unsigned + 2 float) ──

    // ── 1-byte signed: i8 ─────────────────────────────────────────────────

    #[test]
    fn decode_i8_le_round_trips_to_f32() {
        // i8 is 1 byte, so byte order is irrelevant; passing LE for
        // symmetry with the multi-byte cases.
        let bytes = vec![0u8, 1, 127, 128, 255];
        let out = decode_bytes_to_f32(
            &bytes,
            1,
            true,
            false,
            ByteOrder::LeastSignificantByteFirst,
            5,
            "int8",
        )
        .unwrap();
        assert_eq!(out, vec![0.0, 1.0, 127.0, -128.0, -1.0]);
    }

    #[test]
    fn decode_i8_be_round_trips_to_f32() {
        // Same data; passing BE exercises the parameter pass-through on the
        // 1-byte path. Output is identical because elem_size=1.
        let bytes = vec![0u8, 1, 127, 128, 255];
        let out = decode_bytes_to_f32(
            &bytes,
            1,
            true,
            false,
            ByteOrder::MostSignificantByteFirst,
            5,
            "int8",
        )
        .unwrap();
        assert_eq!(out, vec![0.0, 1.0, 127.0, -128.0, -1.0]);
    }

    // ── 2-byte signed: i16 (i16BE already covered above) ─────────────────

    #[test]
    fn decode_i16_le_round_trips_to_f32() {
        let raw: Vec<u8> = [1_i16, -1, 32767, -32768]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            2,
            true,
            false,
            ByteOrder::LeastSignificantByteFirst,
            4,
            "int16",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, -1.0, 32767.0, -32768.0]);
    }

    // ── 4-byte signed: i32 ────────────────────────────────────────────────

    #[test]
    fn decode_i32_le_round_trips_to_f32() {
        let raw: Vec<u8> = [1_i32, -1, i32::MAX, i32::MIN]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            4,
            true,
            false,
            ByteOrder::LeastSignificantByteFirst,
            4,
            "int32",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, -1.0, i32::MAX as f32, i32::MIN as f32]);
    }

    #[test]
    fn decode_i32_be_round_trips_to_f32() {
        let raw: Vec<u8> = [1_i32, -1, i32::MAX, i32::MIN]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            4,
            true,
            false,
            ByteOrder::MostSignificantByteFirst,
            4,
            "int32",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, -1.0, i32::MAX as f32, i32::MIN as f32]);
    }

    // ── 8-byte signed: i64 ────────────────────────────────────────────────

    #[test]
    fn decode_i64_le_round_trips_to_f32() {
        // 123_456_789_012_345 fits exactly in f64; the resulting f32 may
        // lose ~3 digits of precision at the low end, so use a tolerance
        // comparison.
        let raw: Vec<u8> = [1_i64, -1, 123_456_789_012_345_i64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            8,
            true,
            false,
            ByteOrder::LeastSignificantByteFirst,
            3,
            "int64",
        )
        .unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], -1.0);
        assert!(
            (out[2] - 123_456_789_012_345.0_f32).abs() < 1.0,
            "expected ~{:.0}, got {}",
            123_456_789_012_345.0_f32,
            out[2]
        );
    }

    #[test]
    fn decode_i64_be_round_trips_to_f32() {
        let raw: Vec<u8> = [1_i64, -1, 123_456_789_012_345_i64]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            8,
            true,
            false,
            ByteOrder::MostSignificantByteFirst,
            3,
            "int64",
        )
        .unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], -1.0);
        assert!((out[2] - 123_456_789_012_345.0_f32).abs() < 1.0);
    }

    // ── 1-byte unsigned: u8 (u8LE already covered above) ─────────────────

    #[test]
    fn decode_u8_be_round_trips_to_f32() {
        // u8 is 1 byte, so byte order is ignored. Passing BE exercises the
        // parameter pass-through without changing the output.
        let bytes = vec![0u8, 1, 127, 255];
        let out = decode_bytes_to_f32(
            &bytes,
            1,
            false,
            false,
            ByteOrder::MostSignificantByteFirst,
            4,
            "uint8",
        )
        .unwrap();
        assert_eq!(out, vec![0.0, 1.0, 127.0, 255.0]);
    }

    // ── 2-byte unsigned: u16 ──────────────────────────────────────────────

    #[test]
    fn decode_u16_le_round_trips_to_f32() {
        let raw: Vec<u8> = [1_u16, 1000, 65535]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            2,
            false,
            false,
            ByteOrder::LeastSignificantByteFirst,
            3,
            "uint16",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 1000.0, 65535.0]);
    }

    #[test]
    fn decode_u16_be_round_trips_to_f32() {
        let raw: Vec<u8> = [1_u16, 1000, 65535]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            2,
            false,
            false,
            ByteOrder::MostSignificantByteFirst,
            3,
            "uint16",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 1000.0, 65535.0]);
    }

    // ── 4-byte unsigned: u32 ──────────────────────────────────────────────

    #[test]
    fn decode_u32_le_round_trips_to_f32() {
        let raw: Vec<u8> = [1_u32, 1_000_000, u32::MAX]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            4,
            false,
            false,
            ByteOrder::LeastSignificantByteFirst,
            3,
            "uint32",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 1_000_000.0, u32::MAX as f32]);
    }

    #[test]
    fn decode_u32_be_round_trips_to_f32() {
        let raw: Vec<u8> = [1_u32, 1_000_000, u32::MAX]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            4,
            false,
            false,
            ByteOrder::MostSignificantByteFirst,
            3,
            "uint32",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 1_000_000.0, u32::MAX as f32]);
    }

    // ── 8-byte unsigned: u64 ──────────────────────────────────────────────

    #[test]
    fn decode_u64_le_round_trips_to_f32() {
        let raw: Vec<u8> = [1_u64, 123_456_789_012_345_u64, u64::MAX]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            8,
            false,
            false,
            ByteOrder::LeastSignificantByteFirst,
            3,
            "uint64",
        )
        .unwrap();
        assert_eq!(out[0], 1.0);
        assert!((out[1] - 123_456_789_012_345.0_f32).abs() < 1.0);
        // u64::MAX = 18_446_744_073_709_551_615; as f32 = 1.8446744e19 (rounded)
        assert!((out[2] - u64::MAX as f32).abs() / (u64::MAX as f32) < 1e-5);
    }

    #[test]
    fn decode_u64_be_round_trips_to_f32() {
        let raw: Vec<u8> = [1_u64, 123_456_789_012_345_u64, u64::MAX]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let out = decode_bytes_to_f32(
            &raw,
            8,
            false,
            false,
            ByteOrder::MostSignificantByteFirst,
            3,
            "uint64",
        )
        .unwrap();
        assert_eq!(out[0], 1.0);
        assert!((out[1] - 123_456_789_012_345.0_f32).abs() < 1.0);
        assert!((out[2] - u64::MAX as f32).abs() / (u64::MAX as f32) < 1e-5);
    }

    // ── Edge cases & error paths ──────────────────────────────────────────

    #[test]
    fn decode_count_zero_returns_empty() {
        // count=0 should succeed and return an empty Vec, regardless of
        // the byte buffer contents.
        let bytes = vec![0u8; 4];
        let out = decode_bytes_to_f32(
            &bytes,
            4,
            false,
            true,
            ByteOrder::LeastSignificantByteFirst,
            0,
            "float",
        )
        .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn decode_type_name_passes_through_to_error() {
        // The function does NOT validate the type_name string — it is only
        // used in error messages. Verify a happy-path call with an
        // arbitrary type_name succeeds, and a buffer-too-short error
        // mentions the type_name verbatim.
        let bytes: Vec<u8> = vec![0u8; 4];
        // Happy path: arbitrary type_name is accepted.
        let out = decode_bytes_to_f32(
            &bytes,
            4,
            false,
            true,
            ByteOrder::LeastSignificantByteFirst,
            1,
            "MET_CARDIAC_3D",
        )
        .unwrap();
        assert_eq!(out, vec![0.0]);
        // Error path: type_name appears in the "Insufficient data" message.
        let err = decode_bytes_to_f32(
            &bytes,
            4,
            false,
            true,
            ByteOrder::LeastSignificantByteFirst,
            10,
            "MET_CARDIAC_3D",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("MET_CARDIAC_3D"),
            "expected error to mention the type name, got: {err:#}"
        );
    }

    #[test]
    fn decode_unsupported_float_element_size_returns_error() {
        // is_float=true with elem_size=12 (neither 4 nor 8) returns an
        // explicit error rather than panicking.
        let bytes = vec![0u8; 24];
        let err = decode_bytes_to_f32(
            &bytes,
            12,
            false,
            true,
            ByteOrder::LeastSignificantByteFirst,
            2,
            "long_double",
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Unsupported float element size 12"),
            "expected explicit error, got: {err:#}"
        );
    }
}
