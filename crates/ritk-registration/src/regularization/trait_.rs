//! Regularizer trait definition.
//!
//! This module defines the core trait for all regularization techniques.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Trait for deformation field regularizers.
///
/// Regularizers constrain deformation fields to ensure smoothness and
/// physical plausibility during registration.
///
/// # Type Parameters
/// * `B` - The backend type
pub trait Regularizer<B: Backend> {
    /// Compute the regularization loss for a displacement field.
    ///
    /// # Arguments
    /// * `displacement` - The displacement field tensor with shape `[..., D]`
    ///   where `...` represents spatial dimensions and `D` is the displacement dimension.
    ///
    /// # Returns
    /// A scalar tensor containing the regularization loss.
    fn compute_loss<const D: usize>(&self, displacement: Tensor<B, D>) -> Tensor<B, 1>;

    /// Get the weight (scaling factor) for this regularizer.
    fn weight(&self) -> f64;

    /// Set the weight (scaling factor) for this regularizer.
    fn set_weight(&mut self, weight: f64);
}
