//! Elastic regularization for displacement fields.
//!
//! Elastic regularization combines first-order (diffusion) and second-order
//! (bending) terms, balancing smoothness and flexibility for image registration.
//!
//! The elastic regularization term is derived from linear elasticity theory:
//!
//! R(u) = ∫_Ω [μ|∇u + (∇u)^T|² + λ(div u)²] dx
//!
//! where μ is the shear modulus and λ is the first Lamé parameter.
//!
//! Simplified forms used in practice:
//! - Hyperelastic: Combines membrane energy with volume preservation
//! - Linear elastic: Standard elasticity formulation
//!
//! ## Mathematical Background
//!
//! The elastic energy measures how much a deformation deviates from
//! being rigid (uniform translation/rotation), penalizing both shearing
//! and volume change.
//!
//! ## References
//! - Broit (1981): Original elastic registration formulation
//! - Christensen et al. (1996): Consistent linear elastic registration
//! - Modern variants: Hyperelastic (volume-preserving) formulations

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use super::trait_::Regularizer;
use super::trait_::utils::{spatial_gradient_2d, spatial_gradient_3d};

/// Elastic regularizer combining membrane and volume-preserving terms.
///
/// This regularizer balances smoothness with the ability to preserve
/// local volumes during deformation (hyperelastic behavior).
///
/// The regularization term is:
/// R(u) = alpha * membrane_energy + beta * volume_preservation_term
///
/// where:
/// - membrane_energy penalizes spatial variation
/// - volume_preservation_term discourages volume change
///
/// # Example
///
/// ```rust,ignore
/// use ritk_registration::regularization::ElasticRegularizer;
/// use burn::tensor::Tensor;
///
/// // Create hyperelastic regularizer
/// let reg = ElasticRegularizer::hyperelastic(0.1, 0.01);
/// let displacement = Tensor::zeros([1, 2, 64, 64], &device);
/// let loss = reg.compute_loss::<4>(displacement);
/// ```
#[derive(Clone, Debug)]
pub struct ElasticRegularizer {
    /// Weight for membrane (first-order smoothness) term
    alpha: f64,
    /// Weight for volume-preservation term
    beta: f64,
}

impl ElasticRegularizer {
    /// Create a new elastic regularizer with custom weights.
    ///
    /// # Arguments
    /// * `alpha` - Weight for membrane energy (shear modulus)
    /// * `beta` - Weight for volume preservation (bulk modulus)
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }
    
    /// Create a hyperelastic regularizer favoring volume preservation.
    ///
    /// This configuration is suitable for applications where maintaining
    /// local volumes is important (e.g., brain registration).
    ///
    /// # Arguments
    /// * `alpha` - Weight for membrane energy (typically ~0.1)
    /// * `beta` - Weight for volume preservation (typically smaller than alpha)
    pub fn hyperelastic(alpha: f64, beta: f64) -> Self {
        Self::new(alpha, beta)
    }
    
    /// Create a standard linear elastic regularizer.
    ///
    /// Uses equal weighting for both terms.
    pub fn linear(weight: f64) -> Self {
        Self::new(weight, weight)
    }
}

impl Default for ElasticRegularizer {
    fn default() -> Self {
        // Default to hyperelastic with moderate volume preservation
        Self::hyperelastic(0.1, 0.01)
    }
}

impl<B: Backend> Regularizer<B> for ElasticRegularizer {
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
                
                // Compute spatial gradients
                let (grad_h, grad_w) = spatial_gradient_2d(displacement_4d.clone());
                
                // Membrane energy: trace of Jacobian transpose * Jacobian
                let membrane = grad_h.clone().powf_scalar(2.0) + grad_w.clone().powf_scalar(2.0);
                
                // Volume preservation: (d/dx u_x + d/dy u_y)^2
                // For 2D: trace of Jacobian = div(u) = ∂u_x/∂x + ∂u_y/∂y
                // grad_h has [dH/dy, dW/dy] components for 2 displacements
                // We need the divergence: d/dy of first displacement + d/dx of second
                let div_u = grad_h.narrow(1, 0, 1) + grad_w.narrow(1, 1, 1);
                let volume_term = div_u.powf_scalar(2.0);
                
                let total = membrane.mean() * self.alpha + volume_term.mean() * self.beta;
                total
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
                
                let (grad_d, grad_h, grad_w) = spatial_gradient_3d(displacement_5d.clone());
                
                // Membrane energy
                let membrane = grad_d.clone().powf_scalar(2.0) + grad_h.clone().powf_scalar(2.0) + grad_w.clone().powf_scalar(2.0);
                
                // Volume preservation: (div u)^2
                // For 3D: div(u) = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z
                let div_u = grad_d.narrow(1, 0, 1) + grad_h.narrow(1, 1, 1) + grad_w.narrow(1, 2, 1);
                let volume_term = div_u.powf_scalar(2.0);
                
                let total = membrane.mean() * self.alpha + volume_term.mean() * self.beta;
                total
            }
            _ => panic!("ElasticRegularizer only supports 4D (2D) or 5D (3D) displacement fields"),
        }
    }
    
    fn weight(&self) -> f64 {
        self.alpha // Return primary weight
    }
    
    fn set_weight(&mut self, weight: f64) {
        self.alpha = weight;
        self.beta = weight * 0.1; // Maintain ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_elastic_2d() {
        type Backend = NdArray;
        let device = Default::default();
        
        let reg = ElasticRegularizer::hyperelastic(0.1, 0.01);
        
        // Create displacement field
        let displacement = Tensor::<Backend, 4>::ones([1, 2, 32, 32], &device);
        let loss: f32 = reg.compute_loss(displacement).into_scalar();
        
        // Uniform field should have minimal loss
        assert!(loss < 0.01);
    }
    
    #[test]
    fn test_elastic_weights() {
        let reg = ElasticRegularizer::hyperelastic(0.2, 0.02);
        assert!((reg.weight() - 0.2).abs() < 1e-6);
    }
}
