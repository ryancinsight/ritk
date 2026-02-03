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
use burn::tensor::backend::Backend;
use super::trait_::Regularizer;
use super::trait_::utils::{spatial_gradient_2d, spatial_gradient_3d};

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
        match D {
            4 => {
                // 2D displacement field: [B, 2, H, W]
                let shape = displacement.shape();
                let batch = shape.dims[0];
                let components = shape.dims[1];
                let height = shape.dims[2];
                let width = shape.dims[3];
                let displacement_4d: Tensor<B, 4> = displacement.reshape([batch, components, height, width]);
                
                let (grad_h, grad_w) = spatial_gradient_2d(displacement_4d);
                
                // Isotropic TV: sqrt(grad_h^2 + grad_w^2)
                let grad_mag = (grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0)).sqrt();
                
                grad_mag.mean().mul_scalar(self.weight)
            }
            5 => {
                // 3D displacement field: [B, 3, D, H, W]
                let shape = displacement.shape();
                let batch = shape.dims[0];
                let components = shape.dims[1];
                let depth = shape.dims[2];
                let height = shape.dims[3];
                let width = shape.dims[4];
                let displacement_5d: Tensor<B, 5> = displacement.reshape([batch, components, depth, height, width]);
                
                let (grad_d, grad_h, grad_w) = spatial_gradient_3d(displacement_5d);
                
                // Isotropic TV: sqrt(grad_d^2 + grad_h^2 + grad_w^2)
                let grad_mag = (grad_d.powf_scalar(2.0) + grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0)).sqrt();
                
                grad_mag.mean().mul_scalar(self.weight)
            }
            _ => panic!("TotalVariationRegularizer only supports 4D (2D) or 5D (3D) displacement fields"),
        }
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
    fn test_tv_2d_uniform() {
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
        let reg = TotalVariationRegularizer::new(0.5);
        assert!((reg.weight() - 0.5).abs() < 1e-6);
    }
}
