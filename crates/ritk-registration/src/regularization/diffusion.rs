//! Diffusion regularization for displacement fields.
//!
//! Diffusion regularization is a first-order regularization that penalizes
//! spatial derivatives of the displacement field, encouraging smooth
//! deformations.
//!
//! The diffusion regularization term is:
//!
//! R(u) = ∫_Ω |∇u|² dx
//!
//! where u is the displacement field and ∇u is its spatial gradient.
//!
//! This is also known as:
//! - Tikhonov regularization (first-order)
//! - Membrane energy
//! - L2 smoothness penalty

use super::trait_::Regularizer;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Diffusion regularizer for displacement fields.
///
/// Penalizes first-order spatial derivatives to encourage smooth deformations
/// similar to membrane energy.
///
/// # Example
///
/// ```rust,ignore
/// use ritk_registration::regularization::DiffusionRegularizer;
/// use burn::tensor::Tensor;
///
/// let reg = DiffusionRegularizer::new(0.1);
/// // displacement field: [B, 2, H, W] for 2D
/// let displacement = Tensor::zeros([1, 2, 64, 64], &device);
/// let loss = reg.compute_loss::<4>(displacement);
/// ```
#[derive(Clone, Debug)]
pub struct DiffusionRegularizer {
    weight: f64,
}

impl DiffusionRegularizer {
    /// Create a new diffusion regularizer.
    ///
    /// # Arguments
    /// * `weight` - The weight (scaling factor) for this regularizer.
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

impl Default for DiffusionRegularizer {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl<B: Backend> Regularizer<B> for DiffusionRegularizer {
    fn compute_loss<const D: usize>(&self, displacement: Tensor<B, D>) -> Tensor<B, 1> {
        super::dispatch::dispatch_diffusion(displacement, self.weight)
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn uniform_field_has_near_zero_loss() {
        type Backend = NdArray;
        let device = Default::default();

        let reg = DiffusionRegularizer::new(0.1);

        // Create displacement field with gradient
        let displacement = Tensor::<Backend, 4>::ones([1, 2, 32, 32], &device);
        let loss: f32 = reg.compute_loss(displacement).into_scalar();

        // Loss should be very small for uniform field
        assert!(loss < 0.01);
    }
}
