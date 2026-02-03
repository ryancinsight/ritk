use burn::{
    module::{Module, Param, Ignored},
    tensor::{backend::Backend, Tensor},
    nn::conv::{Conv3d, Conv3dConfig},
};

/// Penalty type for Gradient Loss
#[derive(Debug, Clone, Copy, Default)]
pub enum GradientPenalty {
    #[default]
    L2,
    L1,
}

/// Local Normalized Cross Correlation (NCC) Loss.
///
/// Computes the NCC between two images using a sliding window.
/// A perfect match is 1.0, uncorrelated is 0.0.
/// The loss returned is -NCC (so minimizing it maximizes correlation).
#[derive(Module, Debug)]
pub struct LocalNCCLoss<B: Backend> {
    window_conv: Conv3d<B>,
    window_size: Ignored<usize>,
    epsilon: Ignored<f32>,
}

impl<B: Backend> LocalNCCLoss<B> {
    /// Create a new Local NCC loss module.
    ///
    /// # Arguments
    /// * `window_size` - Size of the sliding window (cube). Default 9.
    /// * `device` - Device to create the module on.
    pub fn new(window_size: usize, device: &B::Device) -> Self {
        let padding = window_size / 2;
        let conv_config = Conv3dConfig::new([1, 1], [window_size, window_size, window_size])
            .with_stride([1, 1, 1])
            .with_padding(burn::nn::PaddingConfig3d::Explicit(padding, padding, padding))
            .with_bias(false);
            
        let conv = conv_config.init(device);
        
        // Set weights to 1.0 / N to compute mean directly
        let n = (window_size * window_size * window_size) as f32;
        let weight = Tensor::ones([1, 1, window_size, window_size, window_size], device) / n;
        
        // Update weights via record
        let mut record = conv.clone().into_record();
        record.weight = Param::from_tensor(weight);
        let window_conv = conv.load_record(record);
        
        Self {
            window_conv,
            window_size: Ignored(window_size),
            epsilon: Ignored(1e-5),
        }
    }
    
    pub fn forward(&self, y_true: Tensor<B, 5>, y_pred: Tensor<B, 5>) -> Tensor<B, 1> {
        // y_true: Fixed image [B, 1, D, H, W]
        // y_pred: Warped moving image [B, 1, D, H, W]
        
        let ii = y_true.clone() * y_true.clone();
        let jj = y_pred.clone() * y_pred.clone();
        let ij = y_true.clone() * y_pred.clone();
        
        // Local means
        let i_mean = self.window_conv.forward(y_true.clone());
        let j_mean = self.window_conv.forward(y_pred.clone());
        let i2_mean = self.window_conv.forward(ii);
        let j2_mean = self.window_conv.forward(jj);
        let ij_mean = self.window_conv.forward(ij);
        
        // Covariance and Variance
        let cross = ij_mean - i_mean.clone() * j_mean.clone();
        
        let i_var = i2_mean - i_mean.powf_scalar(2.0);
        let j_var = j2_mean - j_mean.powf_scalar(2.0);
        
        let cc = cross.clone() * cross / (i_var * j_var + *self.epsilon);
        
        // Mean over all spatial locations and batch
        cc.mean().neg()
    }
}

use std::marker::PhantomData;

/// Global Normalized Cross Correlation (NCC) Loss.
///
/// Computes the NCC over the entire image.
#[derive(Module, Debug)]
pub struct GlobalNCCLoss<B: Backend> {
    epsilon: Ignored<f32>,
    phantom: PhantomData<B>,
}

impl<B: Backend> GlobalNCCLoss<B> {
    pub fn new() -> Self {
        Self { epsilon: Ignored(1e-5), phantom: PhantomData }
    }

    pub fn forward(&self, y_true: Tensor<B, 5>, y_pred: Tensor<B, 5>) -> Tensor<B, 1> {
        let i_mean = y_true.clone().mean().reshape([1, 1, 1, 1, 1]);
        let j_mean = y_pred.clone().mean().reshape([1, 1, 1, 1, 1]);
        
        let i_hat = y_true.clone().sub(i_mean);
        let j_hat = y_pred.clone().sub(j_mean);
        
        let num = (i_hat.clone() * j_hat.clone()).mean();
        let den = (i_hat.powf_scalar(2.0).mean() * j_hat.powf_scalar(2.0).mean() + *self.epsilon).sqrt();
        
        num.div(den).neg().reshape([1])
    }
}

/// Gradient Loss for smoothness regularization.
///
/// Penalizes local gradients of the displacement field.
#[derive(Module, Debug)]
pub struct GradLoss<B: Backend> {
    penalty: Ignored<GradientPenalty>,
    phantom: PhantomData<B>,
}

impl<B: Backend> GradLoss<B> {
    pub fn new(penalty: GradientPenalty) -> Self {
        Self { 
            penalty: Ignored(penalty),
            phantom: PhantomData,
        }
    }
    
    pub fn forward(&self, flow: Tensor<B, 5>) -> Tensor<B, 1> {
        // flow: [B, 3, D, H, W]
        // Compute gradients
        
        let [b, c, d, h, w] = flow.dims();
        
        // dx = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        let dy = flow.clone().slice([0..b, 0..c, 1..d, 0..h, 0..w]) - flow.clone().slice([0..b, 0..c, 0..d-1, 0..h, 0..w]);
        let dx = flow.clone().slice([0..b, 0..c, 0..d, 1..h, 0..w]) - flow.clone().slice([0..b, 0..c, 0..d, 0..h-1, 0..w]);
        let dz = flow.clone().slice([0..b, 0..c, 0..d, 0..h, 1..w]) - flow.clone().slice([0..b, 0..c, 0..d, 0..h, 0..w-1]);
        
        match *self.penalty {
            GradientPenalty::L2 => {
                let loss = (dy.powf_scalar(2.0).mean() + dx.powf_scalar(2.0).mean() + dz.powf_scalar(2.0).mean()) / 3.0;
                loss.reshape([1])
            }
            GradientPenalty::L1 => {
                let loss = (dy.abs().mean() + dx.abs().mean() + dz.abs().mean()) / 3.0;
                loss.reshape([1])
            }
        }
    }
}

/// Combined registration loss configuration
#[derive(Debug, Clone)]
pub struct RegistrationLossConfig {
    /// Weight for regularization term
    pub reg_weight: f32,
    /// Type of similarity loss
    pub similarity: SimilarityMetric,
    /// Type of regularization
    pub regularization: RegularizationType,
}

impl Default for RegistrationLossConfig {
    fn default() -> Self {
        Self {
            reg_weight: 0.1,
            similarity: SimilarityMetric::Ncc,
            regularization: RegularizationType::L2,
        }
    }
}

/// Similarity metric type
#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    /// Local Normalized Cross Correlation
    Ncc,
    /// Mean Squared Error
    Mse,
    /// Global Normalized Cross Correlation
    GlobalNcc,
}

/// Regularization type
#[derive(Debug, Clone, Copy)]
pub enum RegularizationType {
    /// L2 gradient penalty
    L2,
    /// L1 gradient penalty
    L1,
}

/// Combined registration loss
pub struct RegistrationLoss<B: Backend> {
    config: RegistrationLossConfig,
    ncc_loss: LocalNCCLoss<B>,
    global_ncc_loss: GlobalNCCLoss<B>,
    grad_loss: GradLoss<B>,
}

impl<B: Backend> RegistrationLoss<B> {
    /// Create new registration loss
    pub fn new(config: RegistrationLossConfig, device: &B::Device) -> Self {
        let ncc_loss = LocalNCCLoss::new(9, device);
        let global_ncc_loss = GlobalNCCLoss::new();
        let grad_loss = match config.regularization {
            RegularizationType::L2 => GradLoss::new(GradientPenalty::L2),
            RegularizationType::L1 => GradLoss::new(GradientPenalty::L1),
        };

        Self {
            config,
            ncc_loss,
            global_ncc_loss,
            grad_loss,
        }
    }

    /// Compute similarity loss between fixed and warped moving images
    pub fn similarity_loss(&self, fixed: &Tensor<B, 5>, warped: &Tensor<B, 5>) -> Tensor<B, 1> {
        match self.config.similarity {
            SimilarityMetric::Ncc => self.ncc_loss.forward(fixed.clone(), warped.clone()),
            SimilarityMetric::GlobalNcc => self.global_ncc_loss.forward(fixed.clone(), warped.clone()),
            SimilarityMetric::Mse => {
                let diff = fixed.clone() - warped.clone();
                diff.powf_scalar(2.0).mean().reshape([1])
            }
        }
    }

    /// Compute regularization loss on displacement field
    pub fn regularization_loss(&self, displacement: &Tensor<B, 5>) -> Tensor<B, 1> {
        self.grad_loss.forward(displacement.clone())
    }
}
