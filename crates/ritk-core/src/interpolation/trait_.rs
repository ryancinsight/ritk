//! Interpolator trait for sampling values at continuous coordinates.
//!
//! This module defines the core Interpolator trait that all interpolation methods must implement.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Interpolator trait for sampling values at continuous coordinates.
///
/// Interpolators are used to sample image values at non-integer coordinates,
/// which is essential for image registration and resampling.
///
/// # Type Parameters
/// * `B` - The Burn backend
pub trait Interpolator<B: Backend> {
    /// Interpolate values from a tensor at given continuous indices.
    ///
    /// # Arguments
    /// * `data` - The source tensor (e.g., 3D volume `[D, H, W]` or 2D image `[H, W]`)
    /// * `indices` - The indices at which to interpolate `[Batch, Rank]`
    ///               Rank must match `data` dimensionality
    ///
    /// # Returns
    /// Tensor of sampled values `[Batch]`
    fn interpolate<const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1>;
}
