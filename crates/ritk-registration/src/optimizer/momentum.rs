use crate::optimizer::trait_::{update_parameters, ParameterGradients};
use crate::optimizer::{Optimizer, OptimizerAlgorithm, OptimizerTelemetry};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_image::tensor::Backend;
use std::marker::PhantomData;

/// Gradient descent with exponentially accumulated velocity.
pub struct Momentum<M, B> {
    learning_rate: f64,
    momentum: f32,
    velocity: Vec<Vec<f32>>,
    steps: usize,
    marker: PhantomData<fn() -> (M, B)>,
}

impl<M, B> Momentum<M, B> {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum: momentum as f32,
            velocity: Vec::new(),
            steps: 0,
            marker: PhantomData,
        }
    }
}

impl<M, B> Optimizer<M, B> for Momentum<M, B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    M: Module<f32, B>,
{
    fn step(&mut self, module: M, gradients: ParameterGradients<B>) -> M {
        if self.velocity.is_empty() {
            self.velocity = gradients
                .iter()
                .map(|gradient| vec![0.0; gradient.numel()])
                .collect();
        }
        self.steps += 1;
        let learning_rate = self.learning_rate as f32;
        update_parameters(module, &gradients, |parameter, element, value, derivative| {
            let velocity = &mut self.velocity[parameter][element];
            *velocity = self.momentum * *velocity + derivative;
            value - learning_rate * *velocity
        })
    }

    fn learning_rate(&self) -> f64 { self.learning_rate }
    fn set_learning_rate(&mut self, learning_rate: f64) { self.learning_rate = learning_rate; }
    fn telemetry(&self) -> OptimizerTelemetry {
        OptimizerTelemetry {
            algorithm: OptimizerAlgorithm::Momentum,
            steps: self.steps,
            learning_rate: Some(self.learning_rate),
        }
    }
}
