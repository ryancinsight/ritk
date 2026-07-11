//! Threshold-based intensity suppression filter.
//!
//! # Mathematical Specification
//!
//! Three modes:
//! - Below:   output(x) = if I(x) < threshold { outside_value } else { I(x) }
//! - Above:   output(x) = if I(x) > threshold { outside_value } else { I(x) }
//! - Outside: output(x) = if I(x) < lower || I(x) > upper { outside_value } else { I(x) }

use crate::native_support::map_flat_image;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::tensor::Backend;
use ritk_image::{native::Image as NativeImage, Image};
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

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        Ok(rebuild(self.apply_values(&vals), dims, image))
    }

    /// Apply the threshold operation to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &NativeImage<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        map_flat_image(image, backend, |values, _| self.apply_values(values))
    }

    fn apply_values(&self, vals: &[f32]) -> Vec<f32> {
        match &self.mode {
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
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
#[path = "tests_threshold.rs"]
mod tests;
