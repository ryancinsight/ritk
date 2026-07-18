use crate::optimizer::trait_::{update_parameters, ParameterGradients};
use crate::optimizer::{Optimizer, OptimizerAlgorithm, OptimizerTelemetry};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_image::tensor::Backend;
use std::marker::PhantomData;

/// Vanilla gradient descent over a Coeus module parameter inventory.
pub struct GradientDescent<M, B> {
    learning_rate: f64,
    steps: usize,
    marker: PhantomData<fn() -> (M, B)>,
}

impl<M, B> GradientDescent<M, B> {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            steps: 0,
            marker: PhantomData,
        }
    }
}

impl<M, B> Optimizer<M, B> for GradientDescent<M, B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    M: Module<f32, B>,
{
    fn step(&mut self, module: M, gradients: ParameterGradients<B>) -> M {
        self.steps += 1;
        let learning_rate = self.learning_rate as f32;
        update_parameters(module, &gradients, |_, _, value, derivative| {
            value - learning_rate * derivative
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    fn telemetry(&self) -> OptimizerTelemetry {
        OptimizerTelemetry {
            algorithm: OptimizerAlgorithm::GradientDescent,
            steps: self.steps,
            learning_rate: Some(self.learning_rate),
        }
    }
}
