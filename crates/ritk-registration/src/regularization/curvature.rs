//! Curvature regularization for displacement fields.
//!
//! Curvature regularization is a higher-order regularization that penalizes
//! the rate of change of the field's curvature, encouraging even smoother
//! deformations than first or second-order regularization.
//!
//! The curvature regularization term is based on the Laplacian of the
//! displacement field:
//!
//! R(u) = ∫_Ω |∇²u|² dx
//!
//! where ∇²u is the Laplacian (sum of second derivatives).
//!
//! Differences from bending energy:
//! - Bending energy: Second-order (penalizes curvature directly)
//! - Curvature regularization: Measures smoothness of curvature itself
//!
//! ## When to use curvature regularization
//!
/// - When extremely smooth deformations are desired
/// - For surfaces or thin structures
/// - When bending energy is not sufficient
/// - In deformable models for computational anatomy

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use super::trait_::Regularizer;
use super::trait_::utils::{laplacian, spatial_laplacian_3d};

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
/// use burn::tensor::Tensor;
///
/// let reg = CurvatureRegularizer::new(0.01);
/// // displacement field: [B, 2, H, W] for 2D
/// let displacement = Tensor::zeros([1, 2, 64, 64], &device);
/// let loss = reg.compute_loss::<4>(displacement);
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

impl<B: Backend> Regularizer<B> for CurvatureRegularizer {
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
                
                // Compute Laplacian
                let laplacian = laplacian(displacement_4d);
                
                // Penalize squared Laplacian
                laplacian.powf_scalar(2.0).mean().mul_scalar(self.weight)
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
                
                // Compute Laplacian
                let laplacian = spatial_laplacian_3d(displacement_5d);
                
                // Penalize squared Laplacian
                laplacian.powf_scalar(2.0).mean().mul_scalar(self.weight)
            }
            _ => panic!("CurvatureRegularizer only supports 4D (2D) or 5D (3D) displacement fields"),
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
    fn test_curvature_2d() {
        type Backend = NdArray;
        let device = Default::default();
        
        let reg = CurvatureRegularizer::new(0.01);
        
        // Uniform field should have zero curvature
        let displacement = Tensor::<Backend, 4>::ones([1, 2, 32, 32], &device);
        let loss: f32 = reg.compute_loss(displacement).into_scalar();
        
        assert!(loss < 0.01);
    }
}
