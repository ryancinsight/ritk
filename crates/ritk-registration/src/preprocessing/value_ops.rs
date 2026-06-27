//! Shared scalar preprocessing operations.

use anyhow::{anyhow, Result};

use super::step::IntensityRescaleMode;

const MIN_INTENSITY_RANGE: f32 = 1.0e-8;

pub(crate) fn normalize_values(vals: &[f32], mode: &IntensityRescaleMode) -> Vec<f32> {
    if vals.is_empty() {
        return Vec::new();
    }

    match mode {
        IntensityRescaleMode::ZScore => zscore(vals),
        IntensityRescaleMode::MinMax { out_min, out_max } => minmax(vals, *out_min, *out_max),
    }
}

pub(crate) fn clamp_values(vals: &[f32], lower: f32, upper: f32) -> Vec<f32> {
    vals.iter().map(|&v| v.clamp(lower, upper)).collect()
}

pub(crate) fn apply_mask_values(vals: &[f32], mask: &[u8]) -> Result<Vec<f32>> {
    if vals.len() != mask.len() {
        return Err(anyhow!(
            "image value count {} does not match mask length {}",
            vals.len(),
            mask.len()
        ));
    }

    Ok(vals
        .iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m == 0 { 0.0 } else { v })
        .collect())
}

pub(crate) fn validate_mask(
    mask: &[u8],
    mask_dims: [usize; 3],
    image_shape: [usize; 3],
) -> Result<usize> {
    if mask_dims != image_shape {
        return Err(anyhow!(
            "mask dims {:?} do not match image shape {:?}",
            mask_dims,
            image_shape
        ));
    }

    let expected = checked_voxel_count(mask_dims)?;
    if mask.len() != expected {
        return Err(anyhow!(
            "mask length {} != voxel count {}",
            mask.len(),
            expected
        ));
    }
    Ok(expected)
}

fn zscore(vals: &[f32]) -> Vec<f32> {
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    let variance = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    if std < MIN_INTENSITY_RANGE {
        vec![0.0; vals.len()]
    } else {
        vals.iter().map(|&v| (v - mean) / std).collect()
    }
}

fn minmax(vals: &[f32], out_min: f32, out_max: f32) -> Vec<f32> {
    let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(MIN_INTENSITY_RANGE);
    vals.iter()
        .map(|&v| {
            let normalized = (v - min) / range;
            normalized * (out_max - out_min) + out_min
        })
        .collect()
}

fn checked_voxel_count(dims: [usize; 3]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("mask shape {:?} product overflows usize", dims))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_mask_rejects_overflowing_shape_product() {
        let err = validate_mask(&[], [usize::MAX, 2, 1], [usize::MAX, 2, 1]).unwrap_err();

        assert_eq!(
            err.to_string(),
            format!(
                "mask shape {:?} product overflows usize",
                [usize::MAX, 2, 1]
            )
        );
    }

    #[test]
    fn apply_mask_values_rejects_length_mismatch() {
        let err = apply_mask_values(&[1.0, 2.0], &[1]).unwrap_err();

        assert_eq!(
            err.to_string(),
            "image value count 2 does not match mask length 1"
        );
    }
}
