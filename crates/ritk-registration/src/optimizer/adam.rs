use crate::optimizer::trait_::{update_parameters, ParameterGradients};
use crate::optimizer::{Optimizer, OptimizerAlgorithm, OptimizerTelemetry};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_image::tensor::Backend;
use std::marker::PhantomData;

/// Adam optimizer over a Coeus module parameter inventory.
pub struct AdamOptimizer<M, B> {
    learning_rate: f64,
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    first_moment: Vec<Vec<f32>>,
    second_moment: Vec<Vec<f32>>,
    steps: usize,
    marker: PhantomData<fn() -> (M, B)>,
}

impl<M, B> AdamOptimizer<M, B> {
    pub fn new(learning_rate: f64) -> Self {
        Self::with_config(learning_rate, 0.9, 0.999, 1e-8)
    }

    pub fn with_config(learning_rate: f64, beta_1: f32, beta_2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta_1,
            beta_2,
            epsilon,
            first_moment: Vec::new(),
            second_moment: Vec::new(),
            steps: 0,
            marker: PhantomData,
        }
    }
}

impl<M, B> Optimizer<M, B> for AdamOptimizer<M, B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    M: Module<f32, B>,
{
    fn step(&mut self, module: M, gradients: ParameterGradients<B>) -> M {
        if self.first_moment.is_empty() {
            self.first_moment = gradients.iter().map(|g| vec![0.0; g.numel()]).collect();
            self.second_moment = gradients.iter().map(|g| vec![0.0; g.numel()]).collect();
        }
        self.steps += 1;
        let correction_1 = 1.0 - self.beta_1.powi(self.steps as i32);
        let correction_2 = 1.0 - self.beta_2.powi(self.steps as i32);
        let learning_rate = self.learning_rate as f32;
        update_parameters(module, &gradients, |parameter, element, value, derivative| {
            let first = &mut self.first_moment[parameter][element];
            let second = &mut self.second_moment[parameter][element];
            *first = self.beta_1 * *first + (1.0 - self.beta_1) * derivative;
            *second = self.beta_2 * *second + (1.0 - self.beta_2) * derivative * derivative;
            let estimate = *first / correction_1;
            let variance = *second / correction_2;
            value - learning_rate * estimate / (variance.sqrt() + self.epsilon)
        })
    }

    fn learning_rate(&self) -> f64 { self.learning_rate }
    fn set_learning_rate(&mut self, learning_rate: f64) { self.learning_rate = learning_rate; }
    fn telemetry(&self) -> OptimizerTelemetry {
        OptimizerTelemetry {
            algorithm: OptimizerAlgorithm::Adam,
            steps: self.steps,
            learning_rate: Some(self.learning_rate),
        }
    }
}
