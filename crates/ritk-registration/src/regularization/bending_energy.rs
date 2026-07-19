//! Bending energy regularization for displacement fields.
//!
//! Bending energy regularization (also known as thin-plate bending energy)
//! penalizes second-order spatial derivatives of the displacement field,
//! encouraging deformations with minimal curvature.
//!
//! The bending energy term is:
//!
//! R(u) = âˆ«_Î© [|âˆ‚Â²u/âˆ‚xÂ²|Â² + |âˆ‚Â²u/âˆ‚yÂ²|Â² + 2|âˆ‚Â²u/âˆ‚xâˆ‚y|Â²] dx  (2D)
//!
//! R(u) = âˆ«_Î© [|âˆ‡Â²u|Â²] dx  (simplified form)
//!
//! where âˆ‡Â²u is the Laplacian of the displacement field.
//!
//! This is also known as:
//! - Thin-plate bending energy
//! - Curvature regularization (second-order)
//! - L2 penalty on second derivatives

use super::trait_::Regularizer;
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;

/// Bending energy regularizer for displacement fields.
///
/// Penalizes second-order spatial derivatives to encourage deformations
/// with minimal bending, similar to thin-plate splines.
///
/// # Example
///
/// ```rust,ignore
/// use ritk_registration::regularization::BendingEnergyRegularizer;
/// use ritk_registration::regularization::Regularizer;
/// use coeus_tensor::Tensor;
///
/// let reg = BendingEnergyRegularizer::new(0.1);
/// // displacement field: [B, 2, H, W] for 2D
/// let displacement: Tensor<f32, _> = Tensor::zeros([1, 2, 64, 64]);
/// let loss = reg.compute_loss(&displacement);
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

impl<T, B> Regularizer<T, B> for BendingEnergyRegularizer
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    fn compute_loss(&self, displacement: &Tensor<T, B>) -> T {
        super::dispatch::dispatch_bending_energy(displacement, self.weight)
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
}
