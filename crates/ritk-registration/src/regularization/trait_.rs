//! Regularizer trait definition.
//!
//! This module defines the core trait for all regularization techniques.

use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;

/// Trait for deformation field regularizers.
///
/// Regularizers constrain deformation fields to ensure smoothness and
/// physical plausibility during registration.
///
/// The displacement field is a Coeus tensor of rank 4 (`[B, C, H, W]`, 2-D) or
/// rank 5 (`[B, C, D, H, W]`, 3-D); the rank is validated at dispatch. The
/// regularizer computes a scalar penalty value in the field's native precision.
///
/// # Type Parameters
/// * `T` - The scalar element type of the displacement field.
/// * `B` - The compute backend the field is stored on.
pub trait Regularizer<T: Scalar, B: ComputeBackend> {
    /// Compute the regularization loss for a displacement field.
    ///
    /// # Arguments
    /// * `displacement` - The displacement field tensor with shape
    ///   `[B, C, H, W]` (rank 4) or `[B, C, D, H, W]` (rank 5), where the
    ///   channel dimension `C` holds the per-axis displacement components.
    ///
    /// # Returns
    /// The scalar regularization loss.
    ///
    /// # Panics
    /// Panics if `displacement` is not rank 4 or rank 5.
    fn compute_loss(&self, displacement: &Tensor<T, B>) -> T;

    /// Get the weight (scaling factor) for this regularizer.
    fn weight(&self) -> f64;

    /// Set the weight (scaling factor) for this regularizer.
    fn set_weight(&mut self, weight: f64);
}
