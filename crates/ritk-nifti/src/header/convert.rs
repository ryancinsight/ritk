//! `f64` → `f32` narrowing helpers for NIfTI-1 header encoding and affine
//! derivation, with finiteness and range validation.

use anyhow::{bail, Result};

/// Narrow a `f64` to `f32`, asserting the value is finite and representable.
///
/// # Panics
/// Panics if `value` is non-finite or outside the `f32` range. Used only on
/// fields whose validity is established at header construction.
pub(super) fn f64_to_f32(value: f64, field: &str) -> f32 {
    assert!(
        value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX),
        "invariant: {field} must be finite and f32-representable for NIfTI-1 encoding"
    );
    value as f32
}

pub(super) fn f64x4_to_f32x4(values: [f64; 4], field: &str) -> Result<[f32; 4]> {
    Ok([
        checked_f64_to_f32(values[0], field)?,
        checked_f64_to_f32(values[1], field)?,
        checked_f64_to_f32(values[2], field)?,
        checked_f64_to_f32(values[3], field)?,
    ])
}

pub(super) fn f64_affine_to_f32(values: [[f64; 4]; 4]) -> Result<[[f32; 4]; 4]> {
    let mut out = [[0.0_f32; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = checked_f64_to_f32(values[row][col], "affine")?;
        }
    }
    Ok(out)
}

/// Narrow a `f64` to `f32`, returning an error if non-finite or out of range.
fn checked_f64_to_f32(value: f64, field: &str) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        bail!("NIfTI {field} value must be finite and f32-representable, got {value}");
    }
    Ok(value as f32)
}
