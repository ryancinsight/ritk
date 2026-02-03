//! Metric trait for image similarity measurement.
//!
//! This module defines the core Metric trait that all similarity metrics
//! must implement for image registration.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;

/// Metric trait for measuring similarity between images.
///
/// Metrics compute a loss value that represents the dissimilarity between
/// a fixed (reference) image and a moving (aligned) image. Lower values
/// indicate better alignment.
///
/// # Type Parameters
/// * `B` - The tensor backend
/// * `D` - The spatial dimensionality (2 or 3)
pub trait Metric<B: Backend, const D: usize> {
    /// Calculate the loss (dissimilarity) between fixed and moving images.
    ///
    /// # Arguments
    /// * `fixed` - The fixed (reference) image
    /// * `moving` - The moving (aligned) image
    /// * `transform` - The spatial transform applied to map from fixed to moving image space
    ///
    /// # Returns
    /// Scalar tensor representing the loss value
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1>;

    /// Get the name of this metric.
    ///
    /// # Returns
    /// String identifier for the metric
    fn name(&self) -> &'static str;
}

/// Normalization mode for metrics.
///
/// Controls how metric values are normalized to ensure consistent ranges
/// across different image pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMode {
    /// No normalization applied.
    None,
    /// Normalize by the number of samples.
    ByCount,
    /// Normalize by the variance of the fixed image.
    ByFixedVariance,
    /// Normalize by the product of standard deviations.
    ByJointStd,
}

impl Default for NormalizationMode {
    fn default() -> Self {
        Self::ByCount
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_mode_default() {
        let mode: NormalizationMode = Default::default();
        assert_eq!(mode, NormalizationMode::ByCount);
    }
}
