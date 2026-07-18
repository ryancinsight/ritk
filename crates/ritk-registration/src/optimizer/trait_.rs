//! Optimizer contracts for registration transform parameters.

use coeus_autograd::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_image::tensor::{Backend, Tensor};

/// Gradient tensors in the same order as [`Module::parameters`].
pub type ParameterGradients<B> = Vec<Tensor<f32, B>>;

/// Discriminant for optimizer algorithm variants used in registration telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerAlgorithm {
    GradientDescent,
    Adam,
    AdaptiveStochasticGradientDescent,
    Momentum,
    RegularStepGradientDescent,
}

impl std::fmt::Display for OptimizerAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Lightweight optimizer telemetry for registration workflows.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerTelemetry {
    pub algorithm: OptimizerAlgorithm,
    pub steps: usize,
    pub learning_rate: Option<f64>,
}

/// Optimizer for a Coeus module's ordered parameter inventory.
pub trait Optimizer<M, B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    M: Module<f32, B>,
{
    fn step(&mut self, module: M, gradients: ParameterGradients<B>) -> M;
    fn learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, learning_rate: f64);
    fn telemetry(&self) -> OptimizerTelemetry;
}

/// Apply an elementwise parameter update and reload the module inventory.
pub(crate) fn update_parameters<M, B, F>(
    mut module: M,
    gradients: &ParameterGradients<B>,
    mut update: F,
) -> M
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    M: Module<f32, B>,
    F: FnMut(usize, usize, f32, f32) -> f32,
{
    let parameters = module.parameters();
    assert_eq!(
        parameters.len(),
        gradients.len(),
        "gradient inventory must match module parameter inventory"
    );
    let backend = B::default();
    let updated = parameters
        .iter()
        .zip(gradients)
        .enumerate()
        .map(|(parameter_index, (parameter, gradient))| {
            assert_eq!(
                parameter.tensor.shape(),
                gradient.shape(),
                "gradient shape must match parameter shape"
            );
            let values = parameter
                .tensor
                .as_slice()
                .iter()
                .zip(gradient.as_slice())
                .enumerate()
                .map(|(element_index, (&value, &derivative))| {
                    update(parameter_index, element_index, value, derivative)
                })
                .collect::<Vec<_>>();
            Var::new(
                Tensor::from_slice_on(parameter.tensor.shape().to_vec(), &values, &backend),
                true,
            )
        })
        .collect::<Vec<_>>();
    module.load_parameters(&updated);
    module
}

/// Central-difference gradient over a module's complete parameter inventory.
///
/// The perturbation is `sqrt(epsilon) * max(1, |x|)`, balancing truncation and
/// rounding error for a second-order central difference in native `f32`.
pub(crate) fn central_difference<M, B, F>(module: &M, mut objective: F) -> ParameterGradients<B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    M: Module<f32, B> + Clone,
    F: FnMut(&M) -> f64,
{
    let backend = B::default();
    let parameters = module.parameters();
    parameters
        .iter()
        .enumerate()
        .map(|(parameter_index, parameter)| {
            let base = parameter.tensor.as_slice();
            let mut derivative = Vec::with_capacity(base.len());
            for element_index in 0..base.len() {
                let step = f32::EPSILON.sqrt() * base[element_index].abs().max(1.0);
                let evaluate = |offset: f32| {
                    let mut perturbed = parameters.clone();
                    let mut values = base.to_vec();
                    values[element_index] += offset;
                    perturbed[parameter_index] = Var::new(
                        Tensor::from_slice_on(
                            parameter.tensor.shape().to_vec(),
                            &values,
                            &backend,
                        ),
                        true,
                    );
                    let mut candidate = module.clone();
                    candidate.load_parameters(&perturbed);
                    objective(&candidate)
                };
                derivative.push(((evaluate(step) - evaluate(-step)) / (2.0 * step as f64)) as f32);
            }
            Tensor::from_slice_on(parameter.tensor.shape().to_vec(), &derivative, &backend)
        })
        .collect()
}

/// Learning-rate schedule.
pub trait LearningRateScheduler: Send + Sync {
    fn get_lr(&self, step: usize, initial_lr: f64) -> f64;
}

/// Step-decay learning-rate schedule.
#[derive(Debug, Clone)]
pub struct StepDecay {
    step_size: usize,
    gamma: f64,
}

impl StepDecay {
    pub fn new(step_size: usize, gamma: f64) -> Self {
        assert!(step_size > 0, "step size must be positive");
        assert!((0.0..=1.0).contains(&gamma) && gamma > 0.0, "gamma must be in (0, 1]");
        Self { step_size, gamma }
    }
}

impl LearningRateScheduler for StepDecay {
    fn get_lr(&self, step: usize, initial_lr: f64) -> f64 {
        initial_lr * self.gamma.powi((step / self.step_size) as i32)
    }
}

#[cfg(test)]
#[path = "tests_trait.rs"]
mod tests;
