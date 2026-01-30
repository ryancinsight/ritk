use crate::optimizer::Optimizer;
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Sgd, SgdConfig, Optimizer as BurnOptimizer};
use burn::optim::momentum::MomentumConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::tensor::backend::AutodiffBackend;

/// Momentum optimizer.
///
/// A wrapper around Burn's SGD optimizer with momentum.
pub struct Momentum<M: AutodiffModule<B>, B: AutodiffBackend> {
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, M, B>,
    learning_rate: f64,
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> Momentum<M, B> {
    /// Create a new momentum optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate
    /// * `momentum` - The momentum factor
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        let config = SgdConfig::new().with_momentum(Some(MomentumConfig::new().with_momentum(momentum)));
        Self {
            optimizer: config.init(),
            learning_rate,
        }
    }
}

impl<M, B> Optimizer<M, B> for Momentum<M, B>
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
