//! Binary threshold indicator filter.
//!
//! The filter maps values in the inclusive interval `[lower, upper]` to the
//! foreground value and all other values to the background value.

use crate::morphology::types::ForegroundValue;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Maps pixels inside `[lower_threshold, upper_threshold]` to foreground.
#[derive(Debug, Clone)]
pub struct BinaryThresholdImageFilter {
    /// Lower bound of the foreground interval.
    pub lower_threshold: f32,
    /// Upper bound of the foreground interval.
    pub upper_threshold: f32,
    /// Output value for pixels inside the interval.
    pub foreground: ForegroundValue,
    /// Output value for pixels outside the interval.
    pub background: f32,
}

impl BinaryThresholdImageFilter {
    /// Creates a binary threshold filter.
    pub fn new(
        lower_threshold: f32,
        upper_threshold: f32,
        foreground: impl Into<ForegroundValue>,
        background: f32,
    ) -> Self {
        Self {
            lower_threshold,
            upper_threshold,
            foreground: foreground.into(),
            background,
        }
    }

    /// Applies the indicator to a legacy image.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (values, dims) = extract_vec(image)?;
        let output = binary_threshold_vec(
            &values,
            self.lower_threshold,
            self.upper_threshold,
            f32::from(self.foreground),
            self.background,
        );
        Ok(rebuild(output, dims, image))
    }

    /// Applies the indicator to a native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let foreground = f32::from(self.foreground);
        crate::native_support::map_flat_image(image, backend, |values, _| {
            binary_threshold_vec(
                values,
                self.lower_threshold,
                self.upper_threshold,
                foreground,
                self.background,
            )
        })
    }
}

/// Substrate-agnostic binary indicator kernel.
pub(crate) fn binary_threshold_vec(
    values: &[f32],
    lower: f32,
    upper: f32,
    foreground: f32,
    background: f32,
) -> Vec<f32> {
    values
        .iter()
        .map(|&value| {
            if value >= lower && value <= upper {
                foreground
            } else {
                background
            }
        })
        .collect()
}
