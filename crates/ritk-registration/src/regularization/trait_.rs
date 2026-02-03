//! Regularizer trait definition.
//!
//! This module defines the core trait for all regularization techniques.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Trait for deformation field regularizers.
///
/// Regularizers constrain deformation fields to ensure smoothness and
/// physical plausibility during registration.
///
/// # Type Parameters
/// * `B` - The backend type
pub trait Regularizer<B: Backend> {
    /// Compute the regularization loss for a displacement field.
    ///
    /// # Arguments
    /// * `displacement` - The displacement field tensor with shape `[..., D]`
    ///   where `...` represents spatial dimensions and `D` is the displacement dimension.
    ///
    /// # Returns
    /// A scalar tensor containing the regularization loss.
    fn compute_loss<const D: usize>(&self, displacement: Tensor<B, D>) -> Tensor<B, 1>;
    
    /// Get the weight (scaling factor) for this regularizer.
    fn weight(&self) -> f64;
    
    /// Set the weight (scaling factor) for this regularizer.
    fn set_weight(&mut self, weight: f64);
}

/// Utility functions for computing spatial gradients.
pub mod utils {
    use burn::tensor::Tensor;
    use burn::tensor::backend::Backend;
    
    /// Compute spatial gradients using finite differences.
    ///
    /// # Arguments
    /// * `field` - Input tensor with shape `[B, D, H, W]` for 2D or `[B, D, D, H, W]` for 3D
    ///
    /// # Returns
    /// A tuple of gradient tensors, one for each spatial dimension.
    pub fn spatial_gradient_2d<B: Backend>(field: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [b, c, h, w] = field.dims();
        
        // Gradient in H direction (vertical)
        let top = field.clone().slice([0..b, 0..c, 1..h, 0..w]);
        let bottom = field.clone().slice([0..b, 0..c, 0..(h-1), 0..w]);
        let grad_h = top - bottom;
        
        // Pad to maintain original size
        let zeros_h = Tensor::zeros([b, c, 1, w], &field.device());
        let grad_h = Tensor::cat(vec![grad_h, zeros_h], 2);
        
        // Gradient in W direction (horizontal)
        let left = field.clone().slice([0..b, 0..c, 0..h, 1..w]);
        let right = field.clone().slice([0..b, 0..c, 0..h, 0..(w-1)]);
        let grad_w = left - right;
        
        // Pad to maintain original size
        let zeros_w = Tensor::zeros([b, c, h, 1], &field.device());
        let grad_w = Tensor::cat(vec![grad_w, zeros_w], 3);
        
        (grad_h, grad_w)
    }
    
    /// Compute spatial gradients for 3D field.
    ///
    /// # Arguments
    /// * `field` - Input tensor with shape `[B, D, D, H, W]`
    pub fn spatial_gradient_3d<B: Backend>(field: Tensor<B, 5>) -> (Tensor<B, 5>, Tensor<B, 5>, Tensor<B, 5>) {
        let [b, c, d, h, w] = field.dims();
        
        // Gradient in D direction (depth)
        let front = field.clone().slice([0..b, 0..c, 1..d, 0..h, 0..w]);
        let back = field.clone().slice([0..b, 0..c, 0..(d-1), 0..h, 0..w]);
        let grad_d = front - back;
        let zeros_d = Tensor::zeros([b, c, 1, h, w], &field.device());
        let grad_d = Tensor::cat(vec![grad_d, zeros_d], 2);
        
        // Gradient in H direction (height)
        let top = field.clone().slice([0..b, 0..c, 0..d, 1..h, 0..w]);
        let bottom = field.clone().slice([0..b, 0..c, 0..d, 0..(h-1), 0..w]);
        let grad_h = top - bottom;
        let zeros_h = Tensor::zeros([b, c, d, 1, w], &field.device());
        let grad_h = Tensor::cat(vec![grad_h, zeros_h], 3);
        
        // Gradient in W direction (width)
        let left = field.clone().slice([0..b, 0..c, 0..d, 0..h, 1..w]);
        let right = field.clone().slice([0..b, 0..c, 0..d, 0..h, 0..(w-1)]);
        let grad_w = left - right;
        let zeros_w = Tensor::zeros([b, c, d, h, 1], &field.device());
        let grad_w = Tensor::cat(vec![grad_w, zeros_w], 4);
        
        (grad_d, grad_h, grad_w)
    }
    
    /// Compute second-order spatial derivatives (Laplacian approximation).
    pub fn spatial_laplacian_2d<B: Backend>(field: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = field.dims();
        
        // Central region
        let center = field.clone().slice([0..b, 0..c, 1..(h-1), 1..(w-1)]);
        let left = field.clone().slice([0..b, 0..c, 1..(h-1), 0..(w-2)]);
        let right = field.clone().slice([0..b, 0..c, 1..(h-1), 2..w]);
        let top = field.clone().slice([0..b, 0..c, 0..(h-2), 1..(w-1)]);
        let bottom = field.clone().slice([0..b, 0..c, 2..h, 1..(w-1)]);
        
        let laplacian_center = left + right + top + bottom - center.clone().mul_scalar(4.0);
        
        // Pad back to original size
        let zeros_h = Tensor::zeros([b, c, 1, w-2], &field.device());
        let laplacian = Tensor::cat(vec![zeros_h.clone(), laplacian_center, zeros_h], 2);
        let zeros_w = Tensor::zeros([b, c, h, 1], &field.device());
        let laplacian = Tensor::cat(vec![zeros_w.clone(), laplacian, zeros_w], 3);
        
        laplacian
    }
    
    /// Compute spatial Laplacian for 4D field.
    ///
    /// Alias for spatial_laplacian_2d for simpler naming.
    pub fn laplacian<B: Backend>(field: Tensor<B, 4>) -> Tensor<B, 4> {
        spatial_laplacian_2d(field)
    }
    
    /// Compute spatial Laplacian for 5D field (3D).
    ///
    /// Computes the 3D Laplacian using a 6-point stencil.
    pub fn spatial_laplacian_3d<B: Backend>(field: Tensor<B, 5>) -> Tensor<B, 5> {
        let [b, c, d, h, w] = field.dims();
        
        // Central region (exclude boundaries)
        let center = field.clone().slice([0..b, 0..c, 1..(d-1), 1..(h-1), 1..(w-1)]);
        
        // Neighbors in each dimension
        let front = field.clone().slice([0..b, 0..c, 0..(d-2), 1..(h-1), 1..(w-1)]);
        let back = field.clone().slice([0..b, 0..c, 2..d, 1..(h-1), 1..(w-1)]);
        let top = field.clone().slice([0..b, 0..c, 1..(d-1), 0..(h-2), 1..(w-1)]);
        let bottom = field.clone().slice([0..b, 0..c, 1..(d-1), 2..h, 1..(w-1)]);
        let left = field.clone().slice([0..b, 0..c, 1..(d-1), 1..(h-1), 0..(w-2)]);
        let right = field.clone().slice([0..b, 0..c, 1..(d-1), 1..(h-1), 2..w]);
        
        // 3D Laplacian: sum of second derivatives
        let laplacian_center = front + back + top + bottom + left + right - 
                              center.clone().mul_scalar(6.0);
        
        // Pad back to original size
        let zeros_d = Tensor::zeros([b, c, 1, h-2, w-2], &field.device());
        let laplacian = Tensor::cat(vec![zeros_d.clone(), laplacian_center, zeros_d], 2);
        let zeros_h = Tensor::zeros([b, c, d, 1, w-2], &field.device());
        let laplacian = Tensor::cat(vec![zeros_h.clone(), laplacian, zeros_h.clone()], 3);
        let zeros_w = Tensor::zeros([b, c, d, h, 1], &field.device());
        let laplacian = Tensor::cat(vec![zeros_w.clone(), laplacian, zeros_w], 4);
        
        laplacian
    }
}
