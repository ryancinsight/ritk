use crate::optimizer::{Optimizer, OptimizerTelemetry};
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{GradientsParams, Optimizer as BurnOptimizer, Sgd, SgdConfig};
use burn::tensor::backend::AutodiffBackend;

/// Gradient descent optimizer.
///
/// A wrapper around Burn's SGD optimizer.
pub struct GradientDescent<M: AutodiffModule<B>, B: AutodiffBackend> {
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, M, B>,
    learning_rate: f64,
    steps: usize,
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> GradientDescent<M, B> {
    /// Create a new gradient descent optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate
    pub fn new(learning_rate: f64) -> Self {
        let config = SgdConfig::new();
        Self {
            optimizer: config.init(),
            learning_rate,
            steps: 0,
        }
    }
}

impl<M, B> Optimizer<M, B> for GradientDescent<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn step(&mut self, module: M, gradients: GradientsParams) -> M {
        self.steps += 1;
        self.optimizer.step(self.learning_rate, module, gradients)
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn telemetry(&self) -> OptimizerTelemetry {
        OptimizerTelemetry {
            algorithm: "GradientDescent",
            steps: self.steps,
            learning_rate: Some(self.learning_rate),
        }
    }
}
