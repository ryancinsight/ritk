//! One hierarchical VMamba encoder stage.

use coeus_autograd::Var;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{Conv3d, Module};
use coeus_ops::BackendOps;

use super::config::{DownsamplePolicy, EncoderStageConfig};
use crate::ssmmorph::vmamba_block::{VMambaBlock, VMambaBlockConfig};
use crate::ModelError;

/// VMamba feature blocks with optional projection and downsampling.
#[derive(Clone)]
pub struct EncoderStage<B>
where
    B: Backend + BackendOps<f32>,
{
    projection: Option<Conv3d<f32, B>>,
    blocks: Vec<VMambaBlock<B>>,
    downsample: Option<Conv3d<f32, B>>,
    /// Output channel count.
    pub out_channels: usize,
}

/// Encoder-stage outputs before and after optional downsampling.
pub struct EncoderStageOutput<B>
where
    B: Backend,
{
    /// Features retained for the decoder skip connection.
    pub features: Var<f32, B>,
    /// Features passed to the next encoder stage.
    pub continuation: Option<Var<f32, B>>,
}

impl<B> EncoderStage<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Initialize an encoder stage.
    #[must_use]
    pub fn new(config: EncoderStageConfig) -> Self {
        let mut projection = (config.in_channels != config.out_channels).then(|| {
            Conv3d::with_params(config.in_channels, config.out_channels, 3, 1, 1, 1, false)
        });
        if let Some(layer) = &mut projection {
            crate::initialization::convolution(layer, config.in_channels, 3, 501);
        }
        let blocks = (0..config.depth)
            .map(|_| VMambaBlock::new(VMambaBlockConfig::new_with_dim(config.out_channels)))
            .collect();
        let mut downsample = (config.downsample == DownsamplePolicy::Downsample).then(|| {
            Conv3d::with_params(config.out_channels, config.out_channels, 3, 2, 1, 1, false)
        });
        if let Some(layer) = &mut downsample {
            crate::initialization::convolution(layer, config.out_channels, 3, 502);
        }
        Self {
            projection,
            blocks,
            downsample,
            out_channels: config.out_channels,
        }
    }

    /// Return pre-downsample features and an optional downsampled continuation.
    pub fn forward(&self, input: &Var<f32, B>) -> Result<EncoderStageOutput<B>, ModelError> {
        let mut output = self
            .projection
            .as_ref()
            .map_or_else(|| input.clone(), |layer| layer.forward(input));
        for block in &self.blocks {
            output = block.forward(&output)?;
        }
        let continuation = self.downsample.as_ref().map(|layer| layer.forward(&output));
        Ok(EncoderStageOutput {
            features: output,
            continuation,
        })
    }
}

impl<B> Module<f32, B> for EncoderStage<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = Vec::new();
        if let Some(layer) = &self.projection {
            parameters.extend(layer.parameters());
        }
        for block in &self.blocks {
            parameters.extend(block.parameters());
        }
        if let Some(layer) = &self.downsample {
            parameters.extend(layer.parameters());
        }
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        let output = EncoderStage::forward(self, input)
            .expect("invariant: encoder stage receives a valid volume");
        output.continuation.unwrap_or(output.features)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let mut offset = 0;
        if let Some(layer) = &mut self.projection {
            offset = load(layer, parameters, offset);
        }
        for block in &mut self.blocks {
            offset = load(block, parameters, offset);
        }
        if let Some(layer) = &mut self.downsample {
            load(layer, parameters, offset);
        }
    }

    fn train(&mut self, mode: bool) {
        for block in &mut self.blocks {
            block.train(mode);
        }
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
