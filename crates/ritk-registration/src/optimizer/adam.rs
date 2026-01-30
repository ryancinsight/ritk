use crate::optimizer::Optimizer;
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Adam, AdamConfig, Optimizer as BurnOptimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::tensor::backend::AutodiffBackend;

/// Adam optimizer.
///
/// A wrapper around Burn's Adam optimizer.
pub struct AdamOptimizer<M: AutodiffModule<B>, B: AutodiffBackend> {
    optimizer: OptimizerAdaptor<Adam, M, B>,
    learning_rate: f64,
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> AdamOptimizer<M, B> {
    /// Create a new Adam optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate
    pub fn new(learning_rate: f64) -> Self {
        let config = AdamConfig::new();
        Self {
            optimizer: config.init(),
            learning_rate,
        }
    }

    /// Create a new Adam optimizer with custom beta values.
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate
    /// * `beta_1` - Exponential decay rate for the first moment estimates
    /// * `beta_2` - Exponential decay rate for the second moment estimates
    /// * `epsilon` - Small value to prevent division by zero
    pub fn with_config(learning_rate: f64, beta_1: f32, beta_2: f32, epsilon: f32) -> Self {
        let config = AdamConfig::new()
            .with_beta_1(beta_1)
            .with_beta_2(beta_2)
            .with_epsilon(epsilon);
        Self {
            optimizer: config.init(),
            learning_rate,
        }
    }
}

impl<M, B> Optimizer<M, B> for AdamOptimizer<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn step(&mut self, module: M, gradients: GradientsParams) -> M {
        self.optimizer.step(self.learning_rate, module, gradients)
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}
