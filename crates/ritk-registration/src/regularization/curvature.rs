//! Curvature regularization for displacement fields.
//!
//! Curvature regularization is a higher-order regularization that penalizes
//! the rate of change of the field's curvature, encouraging even smoother
//! deformations than first or second-order regularization.
//!
//! The curvature regularization term is based on the Laplacian of the
//! displacement field:
//!
//! R(u) = âˆ«_Î© |âˆ‡Â²u|Â² dx
//!
//! where âˆ‡Â²u is the Laplacian (sum of second derivatives).
//!
//! Differences from bending energy:
//! - Bending energy: Second-order (penalizes curvature directly)
//! - Curvature regularization: Measures smoothness of curvature itself
//!
//! ## When to use curvature regularization
//!
//! - When extremely smooth deformations are desired
//! - For surfaces or thin structures
//! - When bending energy is not sufficient
//! - In deformable models for computational anatomy

use super::trait_::Regularizer;
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;

/// Curvature regularizer for displacement fields.
///
/// Encourages extremely smooth deformations by penalizing the Laplacian
/// of the displacement field. This provides higher-order smoothness than
/// diffusion or even bending energy.
///
/// # Example
///
/// ```rust,ignore
/// use ritk_registration::regularization::CurvatureRegularizer;
/// use ritk_registration::regularization::Regularizer;
/// use coeus_tensor::Tensor;
///
/// let reg = CurvatureRegularizer::new(0.01);
/// // displacement field: [B, 2, H, W] for 2D
/// let displacement: Tensor<f32, _> = Tensor::zeros([1, 2, 64, 64]);
/// let loss = reg.compute_loss(&displacement);
/// ```
#[derive(Clone, Debug)]
pub struct CurvatureRegularizer {
    weight: f64,
}

impl CurvatureRegularizer {
    /// Create a new curvature regularizer.
    ///
    /// # Arguments
    /// * `weight` - The weight (scaling factor) for this regularizer.
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

impl Default for CurvatureRegularizer {
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl<T, B> Regularizer<T, B> for CurvatureRegularizer
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    fn compute_loss(&self, displacement: &Tensor<T, B>) -> T {
        super::dispatch::dispatch_curvature(displacement, self.weight)
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
}
