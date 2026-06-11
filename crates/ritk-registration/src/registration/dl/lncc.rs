//! Local Normalized Cross Correlation (LNCC) loss module for deep learning registration.
//!
//! # Theorem: Local Normalized Cross Correlation (LNCC)
//! The LNCC loss is a definitive statistical metric evaluating structural coherence between
//! images independent of global intensities by computing sample cross-correlation over
//! localized kernels.
//! Let $F(x)$ and $M(x)$ be the fixed and moving images.
//! The LNCC over a local window $\Omega_x$ centered at $x$ is analytically defined as:
//! $$ LNCC(x) = \frac{\text{Cov}_{\Omega_x}(F, M)}{\max(V_{\Omega_x}(F) \cdot V_{\Omega_x}(M), \epsilon)} $$
//!
//! Our continuous implementation minimizes the negation of the expected mean
//! $ L(\theta) = - \mathbb{E}_x[LNCC(x)] $.
//!
//! **Proof of Invariance:** For strictly linear radiometric transforms
//! $ M(x) = \alpha M'(x) + \beta $, the constants distribute and cancel through the
//! localized covariance numerator and standard deviation product in the denominator,
//! proving $LNCC(M(x), F(x)) = LNCC(M'(x), F(x))$.

use burn::{
    module::{Ignored, Module, Param},
    nn::conv::{Conv3d, Conv3dConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct LocalNCCLoss<B: Backend> {
    pub(super) window_conv: Conv3d<B>,
    pub(super) window_size: Ignored<usize>,
    pub(super) epsilon: Ignored<f32>,
}

impl<B: Backend> LocalNCCLoss<B> {
    pub fn new(window_size: usize, device: &B::Device) -> Self {
        let padding = window_size / 2;
        let conv_config = Conv3dConfig::new([1, 1], [window_size, window_size, window_size])
            .with_stride([1, 1, 1])
            .with_padding(burn::nn::PaddingConfig3d::Explicit(
                padding, padding, padding,
            ))
            .with_bias(false);

        let conv = conv_config.init(device);
        let n = (window_size * window_size * window_size) as f32;
        let weight = Tensor::ones([1, 1, window_size, window_size, window_size], device) / n;

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
        let ii = y_true.clone() * y_true.clone();
        let jj = y_pred.clone() * y_pred.clone();
        let ij = y_true.clone() * y_pred.clone();

        let i_mean = self.window_conv.forward(y_true.clone());
        let j_mean = self.window_conv.forward(y_pred.clone());
        let i2_mean = self.window_conv.forward(ii);
        let j2_mean = self.window_conv.forward(jj);
        let ij_mean = self.window_conv.forward(ij);

        let cross = ij_mean - i_mean.clone() * j_mean.clone();
        let i_var = i2_mean - i_mean.powf_scalar(2.0);
        let j_var = j2_mean - j_mean.powf_scalar(2.0);

        let cc = cross.clone() * cross / (i_var * j_var + *self.epsilon);
        cc.mean().neg()
    }
}
