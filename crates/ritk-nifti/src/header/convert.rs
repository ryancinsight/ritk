#![expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "NIfTI wire and legacy intensity conversions are centralized here with explicit contracts"
)]

//! Numeric conversions at the NIfTI boundary.

use anyhow::{bail, Result};

/// Encodes an NIfTI-1 header scalar in its required `f32` field.
///
/// The conversion uses binary32 round-to-nearest, ties-to-even, including
/// subnormal rounding. Callers validate finite `f32` representability first;
/// NIfTI-2 retains the original `f64` header values.
pub(super) fn encode_header_scalar(value: f64) -> f32 {
    value as f32
}

/// Checks whether a scalar is finite and representable in an NIfTI-1 field.
pub(super) fn validate_nifti1_scalar(value: f64, field: &str) -> Result<()> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        bail!("NIfTI-1 {field} must be finite and f32-representable, got {value}");
    }
    Ok(())
}

/// Converts a signed stored voxel to the legacy floating-point image API.
///
/// Integers with magnitude at most 2^24 are exact. Larger values are rounded
/// to the nearest representable `f32`, with ties to even.
#[must_use]
pub(super) fn intensity_from_signed_voxel(value: i32) -> f32 {
    value as f32
}

/// Converts an unsigned stored voxel to the legacy floating-point image API.
///
/// Integers at most 2^24 are exact. Larger values are rounded to the nearest
/// representable `f32`, with ties to even.
#[must_use]
pub(super) fn intensity_from_unsigned_voxel(value: u32) -> f32 {
    value as f32
}

/// Converts a floating-point intensity to the legacy nonnegative label API.
///
/// Negative values and NaN map to zero, positive values round to the nearest
/// integer with halfway cases away from zero, and values above `u32::MAX`
/// saturate at `u32::MAX`.
#[must_use]
pub(super) fn label_value_from_intensity(value: f32) -> u32 {
    value.max(0.0).round() as u32
}

/// Converts an exactly integral nonnegative header offset to `usize`.
#[must_use]
pub(super) fn voxel_offset(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    let bits = i32::try_from(usize::BITS).expect("invariant: usize width fits i32");
    let exclusive_limit = 2.0_f64.powi(bits);
    if value >= exclusive_limit {
        return None;
    }
    Some(value as usize)
}

#[cfg(test)]
mod tests {
    use super::{
        encode_header_scalar, intensity_from_signed_voxel, label_value_from_intensity,
        validate_nifti1_scalar, voxel_offset,
    };

    #[test]
    fn label_conversion_keeps_rounding_and_saturation_contract() {
        assert_eq!(label_value_from_intensity(-1.5), 0);
        assert_eq!(label_value_from_intensity(f32::NAN), 0);
        assert_eq!(label_value_from_intensity(2.5), 3);
        assert_eq!(label_value_from_intensity(f32::INFINITY), u32::MAX);
        assert_eq!(intensity_from_signed_voxel(16_777_217), 16_777_216.0);
    }

    #[test]
    fn header_scalar_conversion_rounds_and_range_validation_rejects_overflow() {
        assert_eq!(encode_header_scalar(16_777_217.0), 16_777_216.0);
        assert!(validate_nifti1_scalar(16_777_217.0, "sform").is_ok());
        assert!(validate_nifti1_scalar(f64::MAX, "sform").is_err());
    }

    #[test]
    fn voxel_offset_conversion_rejects_nonintegral_and_out_of_range_values() {
        assert_eq!(voxel_offset(352.0), Some(352));
        assert_eq!(voxel_offset(-0.0), Some(0));
        assert_eq!(voxel_offset(-1.0), None);
        assert_eq!(voxel_offset(352.5), None);
        assert_eq!(voxel_offset(f64::INFINITY), None);
        assert_eq!(voxel_offset(f64::NAN), None);
        let bits = i32::try_from(usize::BITS).expect("usize width fits i32");
        assert_eq!(voxel_offset(2.0_f64.powi(bits)), None);
    }
}
