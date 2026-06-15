//! Total Variation regularization for displacement fields.
//!
//! Total Variation (TV) regularization is an L1-based regularization that
//! encourages piecewise-constant or piecewise-smooth solutions while
//! preserving sharp edges. Unlike L2 regularization, TV can maintain
//! discontinuities in the solution.
//!
//! The total variation term is:
//!
//! R(u) = ∫_Ω |∇u| dx
//!
//! where |∇u| is the L1 norm of the gradient.
//!
//! ## Characteristics
//!
use super::trait_::Regularizer;
use burn::tensor::backend::Backend;
/// - **Edge-preserving**: Maintains sharp boundaries
/// - **Non-smooth**: Uses L1 norm (not L2)
/// - **Sparsity**: Promotes sparse gradient estimates
///
/// ## When to use TV regularization
///
/// - When preserving edges is important
/// - For images with discontinuities
/// - When you want piecewise-constant solutions
/// - In inverse problems with sharp transitions
///
/// ## References
/// - Rudin, Osher, Fatemi (1992): Original TV denoising
/// - Chan et al.: TV for image restoration
/// - Extensions for flow and displacement fields
use burn::tensor::Tensor;

/// Total Variation regularizer for displacement fields.
///
/// Encourages piecewise-smooth deformations by penalizing the L1 norm
/// of the gradient. This preserves edges while smoothing homogeneous
/// regions.
///
/// The isotropic TV variant penalizes:
/// TV(u) = ∫_Ω √(∑|∂u_i/∂x_j|²) dx
///
/// # Example
///
/// ```rust,ignore
/// use ritk_registration::regularization::TotalVariationRegularizer;
/// use burn::tensor::Tensor;
///
/// let reg = TotalVariationRegularizer::new(0.05);
/// // displacement field: [B, 2, H, W] for 2D
/// let displacement = Tensor::zeros([1, 2, 64, 64], &device);
/// let loss = reg.compute_loss::<4>(displacement);
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

impl<B: Backend> Regularizer<B> for TotalVariationRegularizer {
    fn compute_loss<const D: usize>(&self, displacement: Tensor<B, D>) -> Tensor<B, 1> {
        super::dispatch::dispatch_total_variation(displacement, self.weight)
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

        let reg = TotalVariationRegularizer::new(0.1);

        // Uniform field should have zero TV
        let displacement = Tensor::<Backend, 4>::ones([1, 2, 32, 32], &device);
        let loss: f32 = reg.compute_loss(displacement).into_scalar();

        assert!(loss < 0.01);
    }

    #[test]
    fn test_tv_weight() {
        type B = NdArray<f32>;
        let reg = TotalVariationRegularizer::new(0.5);
        let weight = <TotalVariationRegularizer as Regularizer<B>>::weight(&reg);
        assert!((weight - 0.5).abs() < 1e-6);
    }
}
