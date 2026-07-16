//! Interpolator trait for sampling values at continuous coordinates.
//!
//! This module defines the core Interpolator trait that all interpolation methods must implement.

use coeus_core::Backend;
use coeus_tensor::Tensor;

/// Interpolator trait for sampling values at continuous coordinates.
///
/// Interpolators are used to sample image values at non-integer coordinates,
/// which is essential for image registration and resampling.
///
/// # Type Parameters
/// * `B` - The compute backend
pub trait Interpolator<B: Backend> {
    /// Interpolate values from a tensor at given continuous indices.
    ///
    /// # Arguments
    /// * `data` - The source tensor (flat f32 data)
    /// * `indices` - The indices at which to interpolate
    ///
    /// # Returns
    /// Tensor of sampled values
    fn interpolate(
        &self,
        data: &Tensor<f32, B>,
        indices: Tensor<f32, B>,
    ) -> Tensor<f32, B>;
}
