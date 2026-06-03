//! Attribute extraction helpers for MINC2 HDF5 files.
//!
//! Provides functions to decode typed `AttributeValue` instances into
//! Rust scalars, arrays, and strings, plus dimension-group attribute
//! parsing and `dimorder` extraction.

use crate::MincDimension;
use anyhow::{bail, Context, Result};
use consus_core::AttributeValue;

// ── Scalar extractors ────────────────────────────────────────────────────────

/// Extract a scalar `f64` from an `AttributeValue`.
pub fn extract_f64(val: &AttributeValue) -> Result<f64> {
    match val {
        AttributeValue::Float(v) => Ok(*v),
        AttributeValue::Int(v) => Ok(*v as f64),
        AttributeValue::Uint(v) => Ok(*v as f64),
        AttributeValue::FloatArray(arr) if arr.len() == 1 => Ok(arr[0]),
        other => bail!("Expected scalar float, got {:?}", other),
    }
}

/// Extract a scalar `i64` from an `AttributeValue`.
pub fn extract_i64(val: &AttributeValue) -> Result<i64> {
    match val {
        AttributeValue::Int(v) => Ok(*v),
        AttributeValue::Uint(v) => Ok(*v as i64),
        AttributeValue::Float(v) => Ok(*v as i64),
        other => bail!("Expected scalar integer, got {:?}", other),
    }
}

/// Extract a 3-element `f64` array from an `AttributeValue`.
pub fn extract_f64_array_3(val: &AttributeValue) -> Result<[f64; 3]> {
    match val {
        AttributeValue::FloatArray(arr) if arr.len() >= 3 => Ok([arr[0], arr[1], arr[2]]),
        AttributeValue::Float(v) => {
            // Scalar: replicate (unusual but handled gracefully).
            Ok([*v, *v, *v])
        }
        other => bail!("Expected float array of length >= 3, got {:?}", other),
    }
}

/// Extract a string from an `AttributeValue`.
pub fn extract_string(val: &AttributeValue) -> Result<String> {
    match val {
        AttributeValue::String(s) => Ok(s.clone()),
        AttributeValue::Bytes(b) => {
            // Strip trailing nulls and decode as UTF-8.
            let end = b.iter().position(|&x| x == 0).unwrap_or(b.len());
            Ok(String::from_utf8_lossy(&b[..end]).into_owned())
        }
        other => bail!("Expected string, got {:?}", other),
    }
}

// ── Dimorder parsing ─────────────────────────────────────────────────────────

/// Extract the `dimorder` attribute from image dataset attributes.
///
/// Returns a vector of dimension names in dataset axis order.
/// Example: `"zspace,yspace,xspace"` → `["zspace", "yspace", "xspace"]`.
///
/// If `dimorder` is absent, returns the default `["zspace", "yspace", "xspace"]`.
pub fn extract_dimorder(attrs: &[consus_hdf5::attribute::Hdf5Attribute]) -> Result<Vec<String>> {
    for attr in attrs {
        if attr.name == "dimorder" {
            let decoded = attr
                .decode_value()
                .map_err(|e| anyhow::anyhow!("Cannot decode dimorder: {}", e))?;
            let s = extract_string(&decoded)?;
            let order: Vec<String> = s
                .split(',')
                .map(|d| d.trim().to_string())
                .filter(|d| !d.is_empty())
                .collect();
            if order.len() < 3 {
                bail!("dimorder has fewer than 3 entries: {:?}", order);
            }
            return Ok(order);
        }
    }
    // Default: zspace varies slowest (outermost), xspace fastest (innermost).
    Ok(vec![
        "zspace".to_string(),
        "yspace".to_string(),
        "xspace".to_string(),
    ])
}

// ── Dimension attribute parsing ──────────────────────────────────────────────

/// Parse dimension attributes into a `MincDimension`.
///
/// Extracts `start`, `step`, `length`, and `direction_cosines` from
/// the attribute list. Missing optional attributes use sensible defaults:
/// - `start` defaults to `0.0`
/// - `step` defaults to `1.0`
/// - `direction_cosines` defaults to the canonical axis direction
///   (`[1,0,0]` for xspace, `[0,1,0]` for yspace, `[0,0,1]` for zspace)
///
/// `length` is required; its absence is an error.
pub fn parse_dimension_attrs(
    name: &str,
    attrs: &[consus_hdf5::attribute::Hdf5Attribute],
) -> Result<MincDimension> {
    use crate::spatial::default_direction_cosines;

    let mut start: f64 = 0.0;
    let mut step: f64 = 1.0;
    let mut length: Option<usize> = None;
    let mut dir_cos: Option<[f64; 3]> = None;

    for attr in attrs {
        let decoded = attr
            .decode_value()
            .map_err(|e| anyhow::anyhow!("Cannot decode attribute '{}': {}", attr.name, e))?;

        match attr.name.as_str() {
            "start" => {
                start = extract_f64(&decoded)
                    .with_context(|| format!("Invalid 'start' attribute on {}", name))?;
            }
            "step" => {
                step = extract_f64(&decoded)
                    .with_context(|| format!("Invalid 'step' attribute on {}", name))?;
            }
            "length" => {
                let v = extract_i64(&decoded)
                    .with_context(|| format!("Invalid 'length' attribute on {}", name))?;
                if v <= 0 {
                    bail!("Dimension '{}' has non-positive length: {}", name, v);
                }
                length = Some(v as usize);
            }
            "direction_cosines" => {
                dir_cos = Some(
                    extract_f64_array_3(&decoded)
                        .with_context(|| format!("Invalid 'direction_cosines' on {}", name))?,
                );
            }
            _ => {}
        }
    }

    let length = length.ok_or_else(|| {
        anyhow::anyhow!("Dimension '{}' missing required 'length' attribute", name)
    })?;

    let direction_cosines = dir_cos.unwrap_or_else(|| default_direction_cosines(name));

    Ok(MincDimension {
        name: name.to_string(),
        start,
        step,
        length,
        direction_cosines,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::approx_constant)]
    fn extract_f64_from_float() {
        let val = AttributeValue::Float(3.14);
        assert!((extract_f64(&val).unwrap() - 3.14).abs() < 1e-10);
    }

    #[test]
    fn extract_f64_from_int() {
        let val = AttributeValue::Int(42);
        assert!((extract_f64(&val).unwrap() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn extract_f64_from_uint() {
        let val = AttributeValue::Uint(7);
        assert!((extract_f64(&val).unwrap() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn extract_f64_from_single_element_array() {
        let val = AttributeValue::FloatArray(vec![2.5]);
        assert!((extract_f64(&val).unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn extract_f64_rejects_multi_element_array() {
        let val = AttributeValue::FloatArray(vec![1.0, 2.0]);
        assert!(extract_f64(&val).is_err());
    }

    #[test]
    fn extract_i64_from_int() {
        let val = AttributeValue::Int(-5);
        assert_eq!(extract_i64(&val).unwrap(), -5);
    }

    #[test]
    fn extract_i64_from_uint() {
        let val = AttributeValue::Uint(100);
        assert_eq!(extract_i64(&val).unwrap(), 100);
    }

    #[test]
    fn extract_i64_from_float_truncates() {
        let val = AttributeValue::Float(3.9);
        assert_eq!(extract_i64(&val).unwrap(), 3);
    }

    #[test]
    fn extract_f64_array_3_from_exact() {
        let val = AttributeValue::FloatArray(vec![0.5, 0.7, 0.3]);
        let arr = extract_f64_array_3(&val).unwrap();
        assert!((arr[0] - 0.5).abs() < 1e-10);
        assert!((arr[1] - 0.7).abs() < 1e-10);
        assert!((arr[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn extract_f64_array_3_from_longer() {
        let val = AttributeValue::FloatArray(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = extract_f64_array_3(&val).unwrap();
        assert!((arr[0] - 1.0).abs() < 1e-10);
        assert!((arr[1] - 2.0).abs() < 1e-10);
        assert!((arr[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn extract_f64_array_3_too_short_errors() {
        let val = AttributeValue::FloatArray(vec![1.0, 2.0]);
        assert!(extract_f64_array_3(&val).is_err());
    }

    #[test]
    fn extract_f64_array_3_from_scalar_replicates() {
        let val = AttributeValue::Float(0.707);
        let arr = extract_f64_array_3(&val).unwrap();
        assert!((arr[0] - 0.707).abs() < 1e-10);
        assert!((arr[1] - 0.707).abs() < 1e-10);
        assert!((arr[2] - 0.707).abs() < 1e-10);
    }

    #[test]
    fn extract_string_from_string() {
        let val = AttributeValue::String("zspace,yspace,xspace".to_string());
        assert_eq!(extract_string(&val).unwrap(), "zspace,yspace,xspace");
    }

    #[test]
    fn extract_string_from_bytes_strips_null() {
        let val = AttributeValue::Bytes(b"hello\0\0\0".to_vec());
        assert_eq!(extract_string(&val).unwrap(), "hello");
    }

    #[test]
    fn extract_dimorder_default_when_absent() {
        let attrs: Vec<consus_hdf5::attribute::Hdf5Attribute> = vec![];
        let order = extract_dimorder(&attrs).unwrap();
        assert_eq!(order, vec!["zspace", "yspace", "xspace"]);
    }
}
