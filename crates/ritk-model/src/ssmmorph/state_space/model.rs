//! Selective state-space layer.

use coeus_autograd::{mul, permute, reshape, sigmoid, slice, softplus, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{Dropout, Linear, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

use super::config::SelectiveStateSpaceConfig;
use super::scan::selective_scan;
use crate::ModelError;

/// Input-dependent S6 state-space transformation.
#[derive(Clone)]
pub struct SelectiveStateSpace<B>
where
    B: Backend + BackendOps<f32>,
{
    input_projection: Linear<f32, B>,
    output_projection: Linear<f32, B>,
    step_contraction: Linear<f32, B>,
    step_expansion: Linear<f32, B>,
    input_matrix_projection: Linear<f32, B>,
    output_matrix_projection: Linear<f32, B>,
    state_log: Var<f32, B>,
    skip_scale: Var<f32, B>,
    dropout: Dropout,
    input_dim: usize,
    output_dim: usize,
    state_dim: usize,
    inner_dim: usize }

impl<B> SelectiveStateSpace<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Initialize a selective state-space layer.
    #[must_use]
    pub fn new(config: SelectiveStateSpaceConfig) -> Self {
        let inner_dim = config.input_dim * config.expand_factor;
        let state_log_values: Vec<f32> = (0..inner_dim * config.state_dim)
            .map(|index| -((index % config.state_dim + 1) as f32).ln())
            .collect();
        let mut layer = Self {
            input_projection: Linear::new(config.input_dim, inner_dim * 2, true),
            output_projection: Linear::new(inner_dim, config.output_dim, true),
            step_contraction: Linear::new(inner_dim, config.dt_rank, true),
            step_expansion: Linear::new(config.dt_rank, inner_dim, true),
            input_matrix_projection: Linear::new(inner_dim, config.state_dim, true),
            output_matrix_projection: Linear::new(inner_dim, config.state_dim, true),
            state_log: Var::new(
                Tensor::from_slice_on(
                    [inner_dim * config.state_dim],
                    &state_log_values,
                    &B::default(),
                ),
                true,
            ),
            skip_scale: Var::new(Tensor::ones_on([inner_dim], &B::default()), true),
            dropout: Dropout::new(config.dropout),
            input_dim: config.input_dim,
            output_dim: config.output_dim,
            state_dim: config.state_dim,
            inner_dim };
        crate::initialization::linear(
            &mut layer.input_projection,
            config.input_dim,
            inner_dim * 2,
            301,
        );
        crate::initialization::linear(
            &mut layer.output_projection,
            inner_dim,
            config.output_dim,
            302,
        );
        crate::initialization::linear(&mut layer.step_contraction, inner_dim, config.dt_rank, 303);
        crate::initialization::linear(&mut layer.step_expansion, config.dt_rank, inner_dim, 304);
        crate::initialization::linear(
            &mut layer.input_matrix_projection,
            inner_dim,
            config.state_dim,
            305,
        );
        crate::initialization::linear(
            &mut layer.output_matrix_projection,
            inner_dim,
            config.state_dim,
            306,
        );
        layer
    }

    /// Transform a tensor whose final two axes are sequence and channels.
    pub fn forward(&self, input: &Var<f32, B>) -> Result<Var<f32, B>, ModelError> {
        let shape = input.tensor.shape();
        if shape.len() < 2 || shape[shape.len() - 1] != self.input_dim {
            return Err(ModelError::Shape {
                operation: "SelectiveStateSpace::forward",
                expected: "[..., sequence, input_dim]",
                actual: shape.to_vec() });
        }
        let sequence = shape[shape.len() - 2];
        let batch: usize = shape[..shape.len() - 2].iter().product();
        let projected = self
            .input_projection
            .forward(&reshape(input, [batch, sequence, self.input_dim]));
        let signal = slice(
            &projected,
            &[(0, batch), (0, sequence), (0, self.inner_dim)],
        );
        let gate = slice(
            &projected,
            &[
                (0, batch),
                (0, sequence),
                (self.inner_dim, self.inner_dim * 2),
            ],
        );
        let step = softplus(
            &self
                .step_expansion
                .forward(&self.step_contraction.forward(&signal)),
        );
        let input_matrix = self.input_matrix_projection.forward(&signal);
        let output_matrix = self.output_matrix_projection.forward(&signal);
        let state = selective_scan(
            self.state_dim,
            &self.state_log,
            &signal,
            &step,
            &input_matrix,
            &output_matrix,
        );
        let gated = mul(&mul(&state, &sigmoid(&gate)), &self.skip_scale);
        let output = self
            .dropout
            .forward(&self.output_projection.forward(&gated));
        let mut output_shape = shape.to_vec();
        *output_shape
            .last_mut()
            .expect("invariant: rank checked above") = self.output_dim;
        Ok(reshape(&output, output_shape))
    }

    /// Transform `[batch, channels, depth, height, width]` data.
    pub fn forward_3d(&self, input: &Var<f32, B>) -> Result<Var<f32, B>, ModelError> {
        let shape = input.tensor.shape();
        if shape.len() != 5 || shape[1] != self.input_dim {
            return Err(ModelError::Shape {
                operation: "SelectiveStateSpace::forward_3d",
                expected: "[batch, input_dim, depth, height, width]",
                actual: shape.to_vec() });
        }
        let (batch, depth, height, width) = (shape[0], shape[2], shape[3], shape[4]);
        let sequence = depth * height * width;
        let flat = reshape(
            &permute(input, &[0, 2, 3, 4, 1]),
            [batch, sequence, self.input_dim],
        );
        let output = self.forward(&flat)?;
        Ok(reshape(
            &permute(
                &reshape(&output, [batch, sequence, self.output_dim]),
                &[0, 2, 1],
            ),
            [batch, self.output_dim, depth, height, width],
        ))
    }
}

impl<B> Module<f32, B> for SelectiveStateSpace<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = self.input_projection.parameters();
        parameters.extend(self.output_projection.parameters());
        parameters.extend(self.step_contraction.parameters());
        parameters.extend(self.step_expansion.parameters());
        parameters.extend(self.input_matrix_projection.parameters());
        parameters.extend(self.output_matrix_projection.parameters());
        parameters.push(self.state_log.clone());
        parameters.push(self.skip_scale.clone());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        SelectiveStateSpace::forward(self, input)
            .expect("invariant: module input satisfies selective state-space contract")
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let mut offset = 0;
        for module in [
            &mut self.input_projection,
            &mut self.output_projection,
            &mut self.step_contraction,
            &mut self.step_expansion,
            &mut self.input_matrix_projection,
            &mut self.output_matrix_projection,
        ] {
            let count = module.parameters().len();
            module.load_parameters(&parameters[offset..offset + count]);
            offset += count;
        }
        self.state_log = parameters[offset].clone();
        self.skip_scale = parameters[offset + 1].clone();
    }

    fn train(&mut self, mode: bool) {
        self.dropout.set_training(mode);
    }
}
