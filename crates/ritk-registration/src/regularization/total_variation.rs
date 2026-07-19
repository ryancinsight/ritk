//! Total Variation regularization for displacement fields.
//!
//! Total Variation (TV) regularization is an L1-based regularization that
//! encourages piecewise-constant or piecewise-smooth solutions while
//! preserving sharp edges. Unlike L2 regularization, TV can maintain
//! discontinuities in the solution.
//!
//! The total variation term is:
//!
//! R(u) = âˆ«_Î© |âˆ‡u| dx
//!
//! where |âˆ‡u| is the L1 norm of the gradient.
//!
//! ## Characteristics
//!
//! - **Edge-preserving**: Maintains sharp boundaries
//! - **Non-smooth**: Uses L1 norm (not L2)
//! - **Sparsity**: Promotes sparse gradient estimates
//!
//! ## When to use TV regularization
//!
//! - When preserving edges is important
//! - For images with discontinuities
//! - When you want piecewise-constant solutions
//! - In inverse problems with sharp transitions
//!
//! ## References
//! - Rudin, Osher, Fatemi (1992): Original TV denoising
//! - Chan et al.: TV for image restoration
//! - Extensions for flow and displacement fields

use super::trait_::Regularizer;
use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;

/// Total Variation regularizer for displacement fields.
///
/// Encourages piecewise-smooth deformations by penalizing the L1 norm
/// of the gradient. This preserves edges while smoothing homogeneous
/// regions.
///
/// The isotropic TV variant penalizes:
/// TV(u) = âˆ«_Î© âˆš(âˆ‘|âˆ‚u_i/âˆ‚x_j|Â²) dx
///
/// # Example
///
/// ```rust,ignore
/// use ritk_registration::regularization::TotalVariationRegularizer;
/// use ritk_registration::regularization::Regularizer;
/// use coeus_tensor::Tensor;
///
/// let reg = TotalVariationRegularizer::new(0.05);
/// // displacement field: [B, 2, H, W] for 2D
/// let displacement: Tensor<f32, _> = Tensor::zeros([1, 2, 64, 64]);
/// let loss = reg.compute_loss(&displacement);
/// ```
#[derive(Clone, Debug)]
pub struct TotalVariationRegularizer {
    weight: f64,
}

impl TotalVariationRegularizer {
    /// Create a new Total Variation regularizer.
    ///
    /// # Arguments
    /// * `weight` - The weight (scaling factor) for this regularizer.
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

impl Default for TotalVariationRegularizer {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl<T, B> Regularizer<T, B> for TotalVariationRegularizer
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    fn compute_loss(&self, displacement: &Tensor<T, B>) -> T {
        super::dispatch::dispatch_total_variation(displacement, self.weight)
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
}
