//! Regular-step gradient descent over Coeus module parameters.

use super::config::RegularStepGdConfig;
use super::convergence::{ConvergenceFlag, ConvergenceReason};
use crate::optimizer::trait_::{update_parameters, ParameterGradients};
use crate::optimizer::{Optimizer, OptimizerAlgorithm, OptimizerTelemetry};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_image::tensor::Backend;
use std::marker::PhantomData;

pub struct RegularStepGradientDescent<M, B> {
    config: RegularStepGdConfig,
    current_step_length: f64,
    current_loss: Option<f64>,
    previous_loss: Option<f64>,
    steps: usize,
    convergence: ConvergenceFlag,
    convergence_reason: Option<ConvergenceReason>,
    marker: PhantomData<fn() -> (M, B)>,
}

impl<M, B> RegularStepGradientDescent<M, B> {
    pub fn new(config: RegularStepGdConfig) -> Self {
        config.validate().expect("RSGD configuration must be valid");
        Self {
            current_step_length: config.initial_step_length,
            config,
            current_loss: None,
            previous_loss: None,
            steps: 0,
            convergence: ConvergenceFlag::default(),
            convergence_reason: None,
            marker: PhantomData,
        }
    }

    pub fn default_config() -> Self {
        Self::new(RegularStepGdConfig::default())
    }

    pub fn set_loss(&mut self, loss: f64) {
        self.current_loss = Some(loss);
    }

    pub fn converged(&self) -> bool {
        self.convergence == ConvergenceFlag::Converged
    }

    pub fn convergence_reason(&self) -> Option<ConvergenceReason> {
        self.convergence_reason
    }

    pub fn current_step_length(&self) -> f64 {
        self.current_step_length
    }

    pub fn steps(&self) -> usize {
        self.steps
    }
}

impl<M, B> Optimizer<M, B> for RegularStepGradientDescent<M, B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    M: Module<f32, B> + Clone,
{
    fn step(&mut self, module: M, gradients: ParameterGradients<B>) -> M {
        if self.converged() {
            return module;
        }
        let norm = gradients
            .iter()
            .flat_map(|gradient| gradient.as_slice())
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt();
        if norm < self.config.gradient_tolerance {
            self.convergence = ConvergenceFlag::Converged;
            self.convergence_reason = Some(ConvergenceReason::GradientConvergence);
            return module;
        }

        self.current_step_length = self
            .current_step_length
            .min(self.config.maximum_step_length);
        let previous = module.clone();
        let effective_learning_rate = (self.current_step_length / norm) as f32;
        let updated = update_parameters(module, &gradients, |_, _, value, derivative| {
            value - effective_learning_rate * derivative
        });

        if matches!((self.current_loss, self.previous_loss), (Some(current), Some(previous)) if current > previous) {
            self.current_step_length *= self.config.relaxation_factor;
            if self.current_step_length < self.config.minimum_step_length {
                self.convergence = ConvergenceFlag::Converged;
                self.convergence_reason = Some(ConvergenceReason::StepConvergence);
            }
            previous
        } else {
            self.previous_loss = self.current_loss;
            self.steps += 1;
            if self.steps >= self.config.maximum_iterations {
                self.convergence = ConvergenceFlag::Converged;
                self.convergence_reason = Some(ConvergenceReason::MaximumIterations);
            }
            if self.config.learning_rate_decay > 0.0 {
                self.current_step_length = (self.config.initial_step_length
                    / (1.0 + self.config.learning_rate_decay * self.steps as f64))
                    .max(self.config.minimum_step_length);
            }
            updated
        }
    }

    fn learning_rate(&self) -> f64 { self.current_step_length }
    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.current_step_length = learning_rate;
    }
    fn telemetry(&self) -> OptimizerTelemetry {
        OptimizerTelemetry {
            algorithm: OptimizerAlgorithm::RegularStepGradientDescent,
            steps: self.steps,
            learning_rate: Some(self.current_step_length),
        }
    }
}
