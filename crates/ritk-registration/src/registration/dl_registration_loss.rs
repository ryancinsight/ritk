use burn::{
    module::{Ignored, Module, Param},
    nn::conv::{Conv3d, Conv3dConfig},
    tensor::{backend::Backend, Tensor},
};
use std::marker::PhantomData;

/// Penalty type for Gradient Loss
#[derive(Debug, Clone, Copy, Default)]
pub enum GradientPenalty {
    #[default]
    L2,
    L1,
}

/// # Theorem: Local Normalized Cross Correlation (LNCC)
/// The LNCC loss is a definitive statistical metric evaluating structural coherence between images 
/// independent of global intensities by computing sample cross-correlation over localized kernels.
/// Let $F(x)$ and $M(x)$ be the fixed and moving images.
/// The LNCC over a local window $\Omega_x$ centered at $x$ is analytically defined as:
/// $$ LNCC(x) = \frac{\text{Cov}_{\Omega_x}(F, M)}{\max(V_{\Omega_x}(F) \cdot V_{\Omega_x}(M), \epsilon)} $$
///
/// Our continuous implementation minimizes the negation of the expected mean $ L(\theta) = - \mathbb{E}_x[LNCC(x)] $.
/// 
/// **Proof of Invariance:** For strictly linear radiometric transforms $ M(x) = \alpha M'(x) + \beta $, the constants distribute and cancel through the localized covariance numerator and standard deviation product in the denominator, proving $LNCC(M(x), F(x)) = LNCC(M'(x), F(x))$.
#[derive(Module, Debug)]
pub struct LocalNCCLoss<B: Backend> {
    window_conv: Conv3d<B>,
    window_size: Ignored<usize>,
    epsilon: Ignored<f32>,
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

/// Global Normalized Cross Correlation (NCC) Loss.
#[derive(Module, Debug)]
pub struct GlobalNCCLoss<B: Backend> {
    epsilon: Ignored<f32>,
    phantom: PhantomData<B>,
}

impl<B: Backend> GlobalNCCLoss<B> {
    pub fn new() -> Self {
        Self {
            epsilon: Ignored(1e-5),
            phantom: PhantomData,
        }
    }

    pub fn forward(&self, y_true: Tensor<B, 5>, y_pred: Tensor<B, 5>) -> Tensor<B, 1> {
        let i_mean = y_true.clone().mean().reshape([1, 1, 1, 1, 1]);
        let j_mean = y_pred.clone().mean().reshape([1, 1, 1, 1, 1]);

        let i_hat = y_true.clone().sub(i_mean);
        let j_hat = y_pred.clone().sub(j_mean);

        let num = (i_hat.clone() * j_hat.clone()).mean();
        let den =
            (i_hat.powf_scalar(2.0).mean() * j_hat.powf_scalar(2.0).mean() + *self.epsilon).sqrt();

        num.div(den).neg().reshape([1])
    }
}

/// # Theorem: First-Order Gradient Regularization (Smoothness)
/// Enforces topological smoothness on the deformation field by penalizing high-frequency
/// spatial gradients. Mathematically, it applies Sobolev space constraints to the
/// unconstrained $L_2$ problem mapping.
/// 
/// Letting $ \phi: \mathbb{R}^3 \to \mathbb{R}^3 $ represent the displacement field vector:
/// $$ R(\phi) = \frac{1}{|\Omega|} \int_{\Omega} \lVert \nabla \phi(x) \rVert_p^p \,dx $$
/// where $\lVert\cdot\rVert_p$ characterizes the spatial penalty (`GradientPenalty`).
/// For `L2`, this is directly analogous to the membrane energy penalty in Tikhonov regularization.
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
        let [b, c, d, h, w] = flow.dims();
        let dy = flow.clone().slice([0..b, 0..c, 1..d, 0..h, 0..w])
            - flow.clone().slice([0..b, 0..c, 0..d - 1, 0..h, 0..w]);
        let dx = flow.clone().slice([0..b, 0..c, 0..d, 1..h, 0..w])
            - flow.clone().slice([0..b, 0..c, 0..d, 0..h - 1, 0..w]);
        let dz = flow.clone().slice([0..b, 0..c, 0..d, 0..h, 1..w])
            - flow.clone().slice([0..b, 0..c, 0..d, 0..h, 0..w - 1]);

        match *self.penalty {
            GradientPenalty::L2 => {
                let loss = (dy.powf_scalar(2.0).mean()
                    + dx.powf_scalar(2.0).mean()
                    + dz.powf_scalar(2.0).mean())
                    / 3.0;
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
    pub reg_weight: f32,
    pub similarity: SimilarityMetric,
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

#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    Ncc,
    Mse,
    GlobalNcc,
}

#[derive(Debug, Clone, Copy)]
pub enum RegularizationType {
    L2,
    L1,
}

pub struct RegistrationLoss<B: Backend> {
    config: RegistrationLossConfig,
    ncc_loss: LocalNCCLoss<B>,
    global_ncc_loss: GlobalNCCLoss<B>,
    grad_loss: GradLoss<B>,
}

impl<B: Backend> RegistrationLoss<B> {
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

    pub fn similarity_loss(&self, fixed: &Tensor<B, 5>, warped: &Tensor<B, 5>) -> Tensor<B, 1> {
        match self.config.similarity {
            SimilarityMetric::Ncc => self.ncc_loss.forward(fixed.clone(), warped.clone()),
            SimilarityMetric::GlobalNcc => {
                self.global_ncc_loss.forward(fixed.clone(), warped.clone())
            }
            SimilarityMetric::Mse => {
                let diff = fixed.clone() - warped.clone();
                diff.powf_scalar(2.0).mean().reshape([1])
            }
        }
    }

    pub fn regularization_loss(&self, displacement: &Tensor<B, 5>) -> Tensor<B, 1> {
        self.grad_loss.forward(displacement.clone())
    }
}
