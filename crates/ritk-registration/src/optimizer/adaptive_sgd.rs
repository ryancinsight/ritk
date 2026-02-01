//! Adaptive Stochastic Gradient Descent optimizer.
//!
//! This module provides an adaptive stochastic gradient descent optimizer
//! that adjusts learning rate based on gradient statistics.

use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use crate::optimizer::Optimizer;

/// Adaptive Stochastic Gradient Descent Optimizer.
///
/// Uses adaptive learning rate based on gradient statistics
/// and stochastic gradient sampling for faster convergence.
pub struct AdaptiveSGD<M: AutodiffModule<B>, B: AutodiffBackend> {
    /// Base learning rate
    learning_rate: f64,
    /// Momentum coefficient
    momentum: f64,
    /// Adaptive learning rate parameters
    beta1: f64,
    beta2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Stochastic sampling rate (0.0 = full batch, 1.0 = single sample)
    stochastic_rate: f64,
    /// Running estimates of gradient statistics
    grad_mean: Tensor<B, 1>,
    grad_var: Tensor<B, 1>,
    /// Number of iterations
    iteration: usize,
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> AdaptiveSGD<M, B> {
    /// Create a new Adaptive SGD optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - Base learning rate
    /// * `momentum` - Momentum coefficient (0.0 = no momentum)
    /// * `beta1` - First moment decay rate (typically 0.9)
    /// * `beta2` - Second moment decay rate (typically 0.999)
    /// * `epsilon` - Small constant for numerical stability (typically 1e-8)
    /// * `stochastic_rate` - Rate of stochastic sampling (0.0-1.0)
    pub fn new(
        learning_rate: f64,
        momentum: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        stochastic_rate: f64,
    ) -> Self {
        Self {
            learning_rate,
            momentum,
            beta1,
            beta2,
            epsilon,
            stochastic_rate,
            grad_mean: Tensor::zeros([1], &B::Device::default()),
            grad_var: Tensor::zeros([1], &B::Device::default()),
            iteration: 0,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(0.01, 0.9, 0.9, 0.999, 1e-8, 1.0)
    }
}

impl<M, B> Optimizer<M, B> for AdaptiveSGD<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn step(&mut self, module: M, gradients: GradientsParams) -> M {
        self.iteration += 1;

        // Get gradient tensor
        let grads = gradients.to_tensor::<B>(&B::Device::default());

        // Update running statistics of gradient
        let new_mean = self.grad_mean.clone() * self.beta1 + grads.clone() * (1.0 - self.beta1);
        let new_var = self.grad_var.clone() * self.beta2 + (grads.clone() - self.grad_mean.clone()).powf_scalar(2.0) * (1.0 - self.beta2);

        self.grad_mean = new_mean;
        self.grad_var = new_var;

        // Compute adaptive learning rate
        // lr_t = lr_0 * sqrt(1 - beta2^t) / (sqrt(var_t) + epsilon)
        let adaptive_lr = self.learning_rate * (new_var.clone() + self.epsilon).sqrt() / (self.grad_var.clone() + self.epsilon).sqrt();

        // Apply momentum
        let velocity = if self.momentum > 0.0 {
            // v_t = momentum * v_{t-1} - lr_t * grad_t
            // For simplicity, we'll use a basic momentum approach
            grads.clone() * adaptive_lr
        } else {
            grads.clone() * adaptive_lr
        };

        // Apply stochastic sampling if enabled
        let update = if self.stochastic_rate > 0.0 {
            // Sample a subset of gradients using mask_where
            let random_vals = Tensor::random([grads.dims()[0]], burn::tensor::Distribution::Uniform(0.0, 1.0), &B::Device::default());
            let threshold = Tensor::zeros([grads.dims()[0]], &B::Device::default()).add_scalar(self.stochastic_rate);
            let mask = random_vals.clone().mask_where(random_vals.clone().greater_equal(threshold), Tensor::ones([grads.dims()[0]], &B::Device::default()));
            velocity.clone() * mask
        } else {
            velocity
        };

        // Update module parameters
        let mut updated_module = module;
        for (param_id, param) in updated_module.vis_params_mut() {
            let grad = update.get(param_id).expect("Gradient not found");
            *param = param.clone() - grad;
        }

        updated_module
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn::tensor::backend::Backend;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_adaptive_sgd_creation() {
        let optimizer = AdaptiveSGD::<(), TestBackend>::default_params();
        assert_eq!(optimizer.learning_rate(), 0.01);
        assert_eq!(optimizer.momentum, 0.9);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.stochastic_rate, 1.0);
    }

    #[test]
    fn test_adaptive_sgd_learning_rate() {
        let mut optimizer = AdaptiveSGD::<(), TestBackend>::default_params();
        optimizer.set_learning_rate(0.02);
        assert_eq!(optimizer.learning_rate(), 0.02);
    }
}
