//! VMamba convolution/state-space block.

use coeus_autograd::{add, gelu, permute, reshape, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{DepthwiseConv3d, LayerNorm, Linear, Module};
use coeus_ops::BackendOps;

use super::cross_scan::{CrossScan, CrossScanConfig};
use super::policy::ScanDimensionality;
use super::state_space::{SelectiveStateSpace, SelectiveStateSpaceConfig};
use crate::ModelError;

/// VMamba block configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VMambaBlockConfig {
    /// Input and output channel count.
    pub dim: usize,
    /// State-space hidden expansion.
    pub expand_factor: usize,
    /// State dimension.
    pub state_dim: usize,
    /// State-space output dropout probability.
    pub dropout: f64,
    /// Spatial dimensionality of the scan.
    pub dimensionality: ScanDimensionality }

impl VMambaBlockConfig {
    /// Construct the standard volumetric block configuration.
    #[must_use]
    pub const fn new_with_dim(dim: usize) -> Self {
        Self {
            dim,
            expand_factor: 2,
            state_dim: 16,
            dropout: 0.0,
            dimensionality: ScanDimensionality::Scan3d }
    }
}

/// Local depthwise features combined with global directional S6 scans.
#[derive(Clone)]
pub struct VMambaBlock<B>
where
    B: Backend + BackendOps<f32>,
{
    first_norm: LayerNorm<f32, B>,
    second_norm: LayerNorm<f32, B>,
    local_features: DepthwiseConv3d<f32, B>,
    cross_scan: CrossScan,
    state_space: SelectiveStateSpace<B>,
    ffn_expand: Linear<f32, B>,
    ffn_project: Linear<f32, B>,
    channels: usize }

impl<B> VMambaBlock<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Initialize a VMamba block.
    #[must_use]
    pub fn new(config: VMambaBlockConfig) -> Self {
        let state_space = SelectiveStateSpace::new(SelectiveStateSpaceConfig {
            input_dim: config.dim,
            output_dim: config.dim,
            state_dim: config.state_dim,
            expand_factor: config.expand_factor,
            dt_rank: 16,
            dropout: config.dropout });
        let scan_config = match config.dimensionality {
            ScanDimensionality::Scan2d => CrossScanConfig::new_2d(),
            ScanDimensionality::Scan3d => CrossScanConfig::new_3d() };
        let mut block = Self {
            first_norm: LayerNorm::new(config.dim, 1e-5),
            second_norm: LayerNorm::new(config.dim, 1e-5),
            local_features: DepthwiseConv3d::new(config.dim, 3, 1, true),
            cross_scan: CrossScan::new(scan_config),
            state_space,
            ffn_expand: Linear::new(config.dim, config.dim * 4, true),
            ffn_project: Linear::new(config.dim * 4, config.dim, true),
            channels: config.dim };
        crate::initialization::depthwise_convolution(&mut block.local_features, 3, 401);
        crate::initialization::linear(&mut block.ffn_expand, config.dim, config.dim * 4, 402);
        crate::initialization::linear(&mut block.ffn_project, config.dim * 4, config.dim, 403);
        block
    }

    /// Transform `[batch, channels, depth, height, width]` features.
    pub fn forward(&self, input: &Var<f32, B>) -> Result<Var<f32, B>, ModelError> {
        let shape = input.tensor.shape();
        if shape.len() != 5 || shape[1] != self.channels {
            return Err(ModelError::Shape {
                operation: "VMambaBlock::forward",
                expected: "[batch, channels, depth, height, width]",
                actual: shape.to_vec() });
        }
        let (batch, depth, height, width) = (shape[0], shape[2], shape[3], shape[4]);
        let normalized = self
            .first_norm
            .forward_nd(&permute(input, &[0, 2, 3, 4, 1]));
        let local = self
            .local_features
            .forward(&permute(&normalized, &[0, 4, 1, 2, 3]));
        let sequences = self.cross_scan.apply(&local)?;
        let processed = sequences
            .iter()
            .map(|sequence| {
                let channel_last = permute(sequence, &[0, 2, 1]);
                self.state_space
                    .forward(&channel_last)
                    .map(|output| permute(&output, &[0, 2, 1]))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let global = self.cross_scan.merge_3d(&processed, depth, height, width)?;
        let residual = add(input, &global);
        let normalized = self
            .second_norm
            .forward_nd(&permute(&residual, &[0, 2, 3, 4, 1]));
        let flat = reshape(&normalized, [batch * depth * height * width, self.channels]);
        let hidden = gelu(&self.ffn_expand.forward(&flat));
        let projected = self.ffn_project.forward(&hidden);
        let projected = permute(
            &reshape(&projected, [batch, depth, height, width, self.channels]),
            &[0, 4, 1, 2, 3],
        );
        Ok(add(&residual, &projected))
    }
}

impl<B> Module<f32, B> for VMambaBlock<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = self.first_norm.parameters();
        parameters.extend(self.second_norm.parameters());
        parameters.extend(self.local_features.parameters());
        parameters.extend(self.state_space.parameters());
        parameters.extend(self.ffn_expand.parameters());
        parameters.extend(self.ffn_project.parameters());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        VMambaBlock::forward(self, input)
            .expect("invariant: module input satisfies VMamba volumetric contract")
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let mut offset = 0;
        offset = load(&mut self.first_norm, parameters, offset);
        offset = load(&mut self.second_norm, parameters, offset);
        offset = load(&mut self.local_features, parameters, offset);
        offset = load(&mut self.state_space, parameters, offset);
        offset = load(&mut self.ffn_expand, parameters, offset);
        self.ffn_project.load_parameters(&parameters[offset..]);
    }

    fn train(&mut self, mode: bool) {
        self.state_space.train(mode);
    }
}

fn load<B, M>(module: &mut M, parameters: &[Var<f32, B>], offset: usize) -> usize
where
    B: Backend + BackendOps<f32>,
    M: Module<f32, B>,
{
    let count = module.parameters().len();
    module.load_parameters(&parameters[offset..offset + count]);
    offset + count
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn forward_preserves_shape_values_and_gradient_connectivity() {
        let block = VMambaBlock::<MoiraiBackend>::new(VMambaBlockConfig::new_with_dim(2));
        let input = Var::new(
            Tensor::ones_on([1, 2, 2, 2, 2], &MoiraiBackend::new()),
            true,
        );
        let output = block.forward(&input).expect("test shape is valid");
        assert_eq!(output.tensor.shape(), &[1, 2, 2, 2, 2]);
        assert!(output
            .tensor
            .as_slice()
            .iter()
            .all(|value| value.is_finite()));
        output.backward();
        assert!(input.grad().is_some(), "VMamba graph must remain connected");
    }
}
