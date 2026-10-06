#![expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "NIfTI image and header fields have specified narrower scalar representations"
)]

use anyhow::{bail, Context, Result};

/// Converts a finite header quantity to the NIfTI-1 field representation.
pub(super) fn header_scalar(value: f64, field: &str) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        bail!("NIfTI-1 {field} must be finite and representable, got {value}");
    }
    Ok(narrow_scalar(value))
}

pub(super) fn exact_header_scalar(value: f64, field: &str) -> Result<()> {
    let encoded = header_scalar(value, field)?;
    if f64::from(encoded) != value {
        bail!("NIfTI-1 {field} cannot preserve {value} exactly");
    }
    Ok(())
}

pub(super) fn finite_header_scalar(value: f64, field: &str) -> Result<()> {
    if !value.is_finite() {
        bail!("NIfTI-2 {field} must be finite, got {value}");
    }
    Ok(())
}

pub(super) fn affine_row_to_image(values: [f64; 4], field: &str) -> Result<[f32; 4]> {
    Ok([
        header_scalar(values[0], field)?,
        header_scalar(values[1], field)?,
        header_scalar(values[2], field)?,
        header_scalar(values[3], field)?,
    ])
}

pub(super) fn affine_to_image(values: [[f64; 4]; 4]) -> Result<[[f32; 4]; 4]> {
    let mut out = [[0.0_f32; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            out[row][column] = image_scalar(values[row][column])?;
        }
    }
    Ok(out)
}

pub(super) fn voxel_offset(value: usize) -> Result<f32> {
    let value = u32::try_from(value).context("NIfTI-1 voxel offset does not fit u32")?;
    header_scalar(f64::from(value), "vox_offset")
}

fn image_scalar(value: f64) -> Result<f32> {
    if value.is_finite() && (value < f64::from(f32::MIN) || value > f64::from(f32::MAX)) {
        bail!("NIfTI sample value is outside the f32 image range: {value}");
    }
    Ok(narrow_scalar(value))
}

fn narrow_scalar(value: f64) -> f32 {
    value as f32
}

pub(crate) trait LabelValue {
    fn to_label_value(self) -> u32;
}

impl LabelValue for f32 {
    fn to_label_value(self) -> u32 {
        self.max(0.0).round() as u32
    }
}

impl LabelValue for f64 {
    fn to_label_value(self) -> u32 {
        self.max(0.0).round() as u32
    }
}
