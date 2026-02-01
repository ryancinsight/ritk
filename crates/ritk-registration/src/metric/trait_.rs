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

/// Common utility functions for metrics.
pub mod utils {
    use burn::tensor::{Tensor, TensorData, Shape};
    use burn::tensor::backend::Backend;

    /// Generate a grid of continuous indices for the given image shape.
    ///
    /// Returns a tensor of shape `[N, D]` where N is the total number of voxels
    /// and D is the dimensionality.
    ///
    /// # Arguments
    /// * `shape` - The image shape `[D0, D1, ...]`
    /// * `device` - The device to create the tensor on
    ///
    /// # Type Parameters
    /// * `B` - The tensor backend
    /// * `D` - The spatial dimensionality
    ///
    /// # Returns
    /// Tensor of shape `[N, D]` containing continuous indices
    pub fn generate_grid<B, const D: usize>(
        shape: [usize; D],
        device: &B::Device,
    ) -> Tensor<B, 2>
    where
        B: Backend,
    {
        match D {
            3 => {
                let s: &[usize] = shape.as_slice();
                generate_grid_3d(s.try_into().expect("Shape must be 3D"), device)
            }
            2 => {
                let s: &[usize] = shape.as_slice();
                generate_grid_2d(s.try_into().expect("Shape must be 2D"), device)
            }
            _ => panic!("Only 2D and 3D grids are supported"),
        }
    }

    /// Generate a grid of continuous indices for a 3D image shape.
    ///
    /// Returns a tensor of shape `[N, 3]` where N is the total number of voxels.
    ///
    /// # Arguments
    /// * `shape` - The image shape `[D, H, W]`
    /// * `device` - The device to create the tensor on
    ///
    /// # Returns
    /// Tensor of shape `[N, 3]` containing continuous indices
    pub fn generate_grid_3d<B>(
        shape: [usize; 3],
        device: &B::Device,
    ) -> Tensor<B, 2>
    where
        B: Backend,
    {
        let d = shape[0];
        let h = shape[1];
        let w = shape[2];
        let total = d * h * w;

        let mut grid = Vec::with_capacity(total * 3);
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    grid.push(x as f32);
                    grid.push(y as f32);
                    grid.push(z as f32);
                }
            }
        }

        Tensor::<B, 1>::from_data(TensorData::new(grid, Shape::new([total * 3])), device)
            .reshape([total, 3])
    }

    /// Generate a grid of continuous indices for a 2D image shape.
    ///
    /// Returns a tensor of shape `[N, 2]` where N is the total number of pixels.
    ///
    /// # Arguments
    /// * `shape` - The image shape `[H, W]`
    /// * `device` - The device to create the tensor on
    ///
    /// # Returns
    /// Tensor of shape `[N, 2]` containing continuous indices
    pub fn generate_grid_2d<B>(
        shape: [usize; 2],
        device: &B::Device,
    ) -> Tensor<B, 2>
    where
        B: Backend,
    {
        let h = shape[0];
        let w = shape[1];
        let total = h * w;

        let mut grid = Vec::with_capacity(total * 2);
        for y in 0..h {
            for x in 0..w {
                grid.push(x as f32);
                grid.push(y as f32);
            }
        }

        Tensor::<B, 1>::from_data(TensorData::new(grid, Shape::new([total * 2])), device)
            .reshape([total, 2])
    }

    /// Compute the mean of a tensor.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor
    ///
    /// # Returns
    /// Scalar tensor containing the mean value
    pub fn compute_mean<B: Backend>(tensor: Tensor<B, 1>) -> Tensor<B, 1> {
        tensor.mean()
    }

    /// Compute the variance of a tensor.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor
    ///
    /// # Returns
    /// Scalar tensor containing the variance
    pub fn compute_variance<B: Backend>(tensor: Tensor<B, 1>) -> Tensor<B, 1> {
        let mean = tensor.clone().mean();
        let diff = tensor - mean;
        (diff.clone() * diff).mean()
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
