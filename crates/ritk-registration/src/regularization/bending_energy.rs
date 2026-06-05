//! Bending energy regularization for displacement fields.
//!
//! Bending energy regularization (also known as thin-plate bending energy)
//! penalizes second-order spatial derivatives of the displacement field,
//! encouraging deformations with minimal curvature.
//!
//! The bending energy term is:
//!
//! R(u) = ∫_Ω [|∂²u/∂x²|² + |∂²u/∂y²|² + 2|∂²u/∂x∂y|²] dx  (2D)
//!
//! R(u) = ∫_Ω [|∇²u|²] dx  (simplified form)
//!
//! where ∇²u is the Laplacian of the displacement field.
//!
//! This is also known as:
//! - Thin-plate bending energy
//! - Curvature regularization (second-order)
//! - L2 penalty on second derivatives

use super::trait_::Regularizer;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Bending energy regularizer for displacement fields.
///
/// Penalizes second-order spatial derivatives to encourage deformations
/// with minimal bending, similar to thin-plate splines.
///
/// # Example
///
/// ```rust,ignore
/// use ritk_registration::regularization::BendingEnergyRegularizer;
/// use burn::tensor::Tensor;
///
/// let reg = BendingEnergyRegularizer::new(0.1);
/// // displacement field: [B, 2, H, W] for 2D
/// let displacement = Tensor::zeros([1, 2, 64, 64], &device);
/// let loss = reg.compute_loss::<4>(displacement);
/// ```
#[derive(Clone, Debug)]
pub struct BendingEnergyRegularizer {
    weight: f64,
}

impl BendingEnergyRegularizer {
    /// Create a new bending energy regularizer.
    ///
    /// # Arguments
    /// * `weight` - The weight (scaling factor) for this regularizer.
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

impl Default for BendingEnergyRegularizer {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl<B: Backend> Regularizer<B> for BendingEnergyRegularizer {
    fn compute_loss<const D: usize>(&self, displacement: Tensor<B, D>) -> Tensor<B, 1> {
        super::dispatch::dispatch_bending_energy(displacement, self.weight)
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
    fn test_bending_energy_2d_uniform() {
        type Backend = NdArray;
        let device = Default::default();

        let reg = BendingEnergyRegularizer::new(0.1);

        // Uniform displacement field should have zero bending
        let displacement = Tensor::<Backend, 4>::ones([1, 2, 32, 32], &device);
        let loss: f32 = reg.compute_loss(displacement).into_scalar();

        assert!(loss < 0.01);
    }
}
