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

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use super::trait_::Regularizer;
use super::trait_::utils::{spatial_gradient_2d, spatial_gradient_3d};

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
        match D {
            4 => {
                // 2D displacement field: [B, 2, H, W]
                // Reshape to concrete type for gradient computation
                let shape = displacement.shape();
                let batch = shape.dims[0];
                let components = shape.dims[1];
                let height = shape.dims[2];
                let width = shape.dims[3];
                let displacement_4d: Tensor<B, 4> = displacement.reshape([batch, components, height, width]);
                let (grad_h, grad_w) = spatial_gradient_2d(displacement_4d);
                
                // Compute squared magnitude of gradient
                let grad_mag_sq = grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0);
                
                // Mean over all dimensions
                grad_mag_sq.mean().mul_scalar(self.weight)
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
                
                // Compute squared magnitude of gradient
                let grad_mag_sq = grad_d.powf_scalar(2.0) + grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0);
                
                // Mean over all dimensions
                grad_mag_sq.mean().mul_scalar(self.weight)
            }
            _ => panic!("DiffusionRegularizer only supports 4D (2D) or 5D (3D) displacement fields"),
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
    fn test_diffusion_2d() {
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
