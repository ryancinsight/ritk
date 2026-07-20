//! Threshold-based intensity suppression filter.
//!
//! # Mathematical Specification
//!
//! Three modes:
//! - Below:   output(x) = if I(x) < threshold { outside_value } else { I(x) }
//! - Above:   output(x) = if I(x) > threshold { outside_value } else { I(x) }
//! - Outside: output(x) = if I(x) < lower || I(x) > upper { outside_value } else { I(x) }

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Threshold mode controlling which pixels are replaced by outside_value.
#[derive(Debug, Clone)]
pub enum ThresholdMode {
    /// Replace pixels strictly below threshold with outside_value.
    Below { threshold: f32, outside_value: f32 },
    /// Replace pixels strictly above threshold with outside_value.
    Above { threshold: f32, outside_value: f32 },
    /// Replace pixels outside [lower, upper] with outside_value.
    Outside {
        lower: f32,
        upper: f32,
        outside_value: f32,
    },
}

/// Conditionally replaces pixel values based on a threshold condition.
#[derive(Debug, Clone)]
pub struct ThresholdImageFilter {
    pub mode: ThresholdMode,
}

impl ThresholdImageFilter {
    pub fn below(threshold: f32, outside_value: f32) -> Self {
        Self {
            mode: ThresholdMode::Below {
                threshold,
                outside_value,
            },
        }
    }
    pub fn above(threshold: f32, outside_value: f32) -> Self {
        Self {
            mode: ThresholdMode::Above {
                threshold,
                outside_value,
            },
        }
    }
    pub fn outside(lower: f32, upper: f32, outside_value: f32) -> Self {
        Self {
            mode: ThresholdMode::Outside {
                lower,
                upper,
                outside_value,
            },
        }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = threshold_vec(&vals, &self.mode);
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`ThresholdImageFilter::apply`].
    ///
    /// Runs the identical per-voxel threshold suppression via the shared
    /// `threshold_vec` host core on the image's contiguous host buffer, so the
    /// result is bitwise-identical to the Coeus path. No tensor is
    /// constructed. Spatial metadata (origin, spacing, direction) is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, _dims| {
            threshold_vec(vals, &self.mode)
        })
    }
}

/// Substrate-agnostic host core for [`ThresholdImageFilter`].
///
/// Replaces every voxel failing the mode's retention predicate with the mode's
/// `outside_value`; retained voxels pass through unchanged.
pub(crate) fn threshold_vec(vals: &[f32], mode: &ThresholdMode) -> Vec<f32> {
    match mode {
        ThresholdMode::Below {
            threshold,
            outside_value,
        } => vals
            .iter()
            .map(|&v| if v < *threshold { *outside_value } else { v })
            .collect(),
        ThresholdMode::Above {
            threshold,
            outside_value,
        } => vals
            .iter()
            .map(|&v| if v > *threshold { *outside_value } else { v })
            .collect(),
        ThresholdMode::Outside {
            lower,
            upper,
            outside_value,
        } => vals
            .iter()
            .map(|&v| {
                if v < *lower || v > *upper {
                    *outside_value
                } else {
                    v
                }
            })
            .collect(),
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
#[path = "tests_threshold.rs"]
mod tests;
