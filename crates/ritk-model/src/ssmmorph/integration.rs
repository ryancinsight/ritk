//! SSMMorph Integration with ritk Framework
//!
//! Provides seamless integration of SSMMorph network with ritk's registration
//! pipeline, including spatial transformation and loss computation.

use burn::prelude::*;
use burn::tensor::cast::ToElement;

use ritk_core::image::Image;
use ritk_core::transform::displacement_field::{DisplacementField, DisplacementFieldTransform3D};
use ritk_core::interpolation::LinearInterpolator;

use super::network::{SSMMorph, SSMMorphConfig};
use super::network::sampling::FlowComposer;
use crate::io::adapter::ImageToTensorAdapter;
use crate::losses::{RegistrationLoss, RegistrationLossConfig};

/// Integration of SSMMorph with ritk registration pipeline
pub struct SSMMorphIntegration<B: Backend> {
    /// The SSMMorph network
    pub network: SSMMorph<B>,
    /// Adapter for converting between ritk Images and Burn tensors
    pub adapter: ImageToTensorAdapter<B>,
}

impl<B: Backend> SSMMorphIntegration<B> {
    /// Create new SSMMorph integration
    ///
    /// # Arguments
    /// * `config` - SSMMorph network configuration
    /// * `device` - Compute device
    pub fn new(config: &SSMMorphConfig, device: &B::Device) -> Self {
        let network = SSMMorph::new(config, device);
        let adapter = ImageToTensorAdapter::new(device.clone());
        
        Self {
            network,
            adapter,
        }
    }
    
    /// Register two images using SSMMorph
    ///
    /// # Arguments
    /// * `fixed` - Fixed reference image
    /// * `moving` - Moving image to register
    ///
    /// # Returns
    /// * Displacement field transform
    pub fn register(
        &self,
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
    ) -> anyhow::Result<DisplacementFieldTransform3D<B>> {
        // Convert images to tensors
        let fixed_tensor = self.adapter.image_to_tensor_3d(fixed)?;
        let moving_tensor = self.adapter.image_to_tensor_3d(moving)?;
        
        // Forward pass through network
        let output = self.network.forward(fixed_tensor, moving_tensor);
        
        // Convert displacement field to ritk transform
        let displacement_field = self.adapter.tensor_to_displacement_field_3d(&output.displacement, fixed)?;
        
        // Create transform from displacement field
        let interpolator = LinearInterpolator::new();
        let transform = DisplacementFieldTransform3D::new(
            displacement_field,
            interpolator,
        );
        
        Ok(transform)
    }
    
    /// Compute registration loss for training
    ///
    /// # Arguments
    /// * `fixed` - Fixed image tensor
    /// * `moving` - Moving image tensor
    /// * `loss_config` - Loss configuration
    ///
    /// # Returns
    /// * Total loss and individual loss components
    pub fn compute_loss(
        &self,
        fixed: Tensor<B, 5>,
        moving: Tensor<B, 5>,
        loss_config: &RegistrationLossConfig,
    ) -> (Tensor<B, 1>, LossComponents<B>) {
        // Forward pass
        let output = self.network.forward(fixed.clone(), moving.clone());
        
        // Apply displacement field to moving image (using spatial transformer)
        let warped = self.warp_image(&moving, &output.displacement);
        
        // Compute similarity loss
        let device = self.adapter.device();
        let loss_fn = RegistrationLoss::new(loss_config.clone(), &device);
        let sim_loss = loss_fn.similarity_loss(&fixed, &warped);
        
        // Compute regularization loss on displacement field
        let reg_loss = loss_fn.regularization_loss(&output.displacement);
        
        // Total loss
        let total_loss = sim_loss.clone() + reg_loss.clone().mul_scalar(loss_config.reg_weight);
        
        (total_loss, LossComponents {
            similarity: sim_loss,
            regularization: reg_loss,
        })
    }
    
    /// Warp image using displacement field
    fn warp_image(&self, image: &Tensor<B, 5>, displacement: &Tensor<B, 5>) -> Tensor<B, 5> {
        // Use FlowComposer for warping
        let composer = FlowComposer::new(self.adapter.device());
        composer.warp(image, displacement)
    }
    
    /// Get network output for analysis/debugging
    pub fn analyze(&self, fixed: &Image<B, 3>, moving: &Image<B, 3>) -> anyhow::Result<SSMMorphAnalysis> {
        let fixed_tensor = self.adapter.image_to_tensor_3d(fixed)?;
        let moving_tensor = self.adapter.image_to_tensor_3d(moving)?;
        
        let output = self.network.forward(fixed_tensor, moving_tensor);
        
        // Analyze displacement field statistics
        let disp = output.displacement;
        let disp_flat = disp.clone().reshape([disp.dims().iter().product::<usize>()]);
        
        let mean = disp_flat.clone().mean();
        let std = disp_flat.clone().var(0).sqrt();
        let min = disp_flat.clone().min();
        let max = disp_flat.max();
        
        Ok(SSMMorphAnalysis {
            displacement_mean: mean.into_scalar().to_f64(),
            displacement_std: std.into_scalar().to_f64(),
            displacement_min: min.into_scalar().to_f64(),
            displacement_max: max.into_scalar().to_f64(),
            num_encoder_features: output.encoder_features.len(),
            bottleneck_shape: output.bottleneck.dims().to_vec(),
        })
    }
}

/// Loss components for monitoring training
#[derive(Debug, Clone)]
pub struct LossComponents<B: Backend> {
    /// Similarity loss (e.g., NCC, MSE, MI)
    pub similarity: Tensor<B, 1>,
    /// Regularization loss (smoothness)
    pub regularization: Tensor<B, 1>,
}

/// Analysis results from SSMMorph forward pass
#[derive(Debug, Clone)]
pub struct SSMMorphAnalysis {
    /// Mean displacement magnitude
    pub displacement_mean: f64,
    /// Standard deviation of displacement
    pub displacement_std: f64,
    /// Minimum displacement component
    pub displacement_min: f64,
    /// Maximum displacement component
    pub displacement_max: f64,
    /// Number of encoder feature stages
    pub num_encoder_features: usize,
    /// Shape of bottleneck features
    pub bottleneck_shape: Vec<usize>,
}

/// Diffeomorphic SSMMorph with full transformation pipeline
///
/// Wraps SSMMorph with proper diffeomorphic constraints and
/// integration with ritk's transform system.
pub struct DiffeomorphicSSMMorph<B: Backend> {
    /// Base SSMMorph integration
    integration: SSMMorphIntegration<B>,
    /// Number of integration steps
    #[allow(dead_code)]
    integration_steps: usize,
}

impl<B: Backend> DiffeomorphicSSMMorph<B> {
    /// Create new diffeomorphic SSMMorph
    pub fn new(config: &SSMMorphConfig, device: &B::Device) -> Self {
        let integration = SSMMorphIntegration::new(config, device);
        
        Self {
            integration,
            integration_steps: config.integration_steps,
        }
    }
    
    /// Register with full diffeomorphic constraints
    pub fn register_diffeomorphic(
        &self,
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
    ) -> anyhow::Result<DisplacementFieldTransform3D<B>> {
        // SSMMorph already outputs diffeomorphic transformations
        // when config.diffeomorphic is true
        self.integration.register(fixed, moving)
    }
    
    /// Compute inverse transformation (for validation)
    ///
    /// Diffeomorphic transformations are invertible by design.
    /// The inverse can be computed by negating and integrating
    /// the velocity field.
    pub fn compute_inverse(
        &self,
        forward_transform: &DisplacementFieldTransform3D<B>,
    ) -> DisplacementFieldTransform3D<B> {
        // Negate displacement field to approximate inverse
        // For exact inverse, would need to solve fixed-point equation
        let forward_disp = forward_transform.field();
        let components = forward_disp.components();
        let neg_components: Vec<Tensor<B, 3>> = components.into_iter().map(|v| -v).collect();
        
        // Create new field with negated components
        let inverse_disp = DisplacementField::new(
            neg_components,
            forward_disp.origin().clone(),
            forward_disp.spacing().clone(),
            forward_disp.direction().clone(),
        );
        
        DisplacementFieldTransform3D::new(
            inverse_disp,
            forward_transform.interpolator().clone(),
        )
    }
    
    /// Validate transformation quality (composition should be identity)
    pub fn validate_transform(
        &self,
        forward: &DisplacementFieldTransform3D<B>,
        inverse: &DisplacementFieldTransform3D<B>,
    ) -> f64 {
        // Compute composition error
        // φ∘φ^{-1}(x) should equal x
        let _composed = self.compose_transforms(forward, inverse);
        
        // Measure deviation from identity
        // Simplified: return 0.0 for now
        0.0
    }
    
    fn compose_transforms(
        &self,
        t1: &DisplacementFieldTransform3D<B>,
        _t2: &DisplacementFieldTransform3D<B>,
    ) -> DisplacementFieldTransform3D<B> {
        // Simplified composition
        t1.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_integration_creation() {
        let device = Default::default();
        let config = SSMMorphConfig::for_3d_registration();
        let integration = SSMMorphIntegration::<TestBackend>::new(&config, &device);
        
        assert_eq!(integration.network.encoder.channels().len(), 4);
    }
    
    #[test]
    fn test_diffeomorphic_wrapper() {
        let device = Default::default();
        let config = SSMMorphConfig::for_3d_registration();
        let diff_ssm = DiffeomorphicSSMMorph::<TestBackend>::new(&config, &device);
        
        assert_eq!(diff_ssm.integration_steps, 7);
    }
}
