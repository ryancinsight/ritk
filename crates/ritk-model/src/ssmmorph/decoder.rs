//! Hierarchical VMamba decoder.

use coeus_autograd::{cat, permute, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{Conv3d, ConvTranspose3d, LayerNorm, Module};
use coeus_ops::BackendOps;

use super::vmamba_block::{VMambaBlock, VMambaBlockConfig};
use crate::ModelError;

/// Whether encoder skip connections participate in decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkipConnections {
    /// Decode without encoder features.
    Disabled,
    /// Fuse the corresponding encoder feature at every stage.
    #[default]
    Enabled,
}

/// One decoder stage configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecoderStageConfig {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// VMamba block count.
    pub depth: usize,
}

/// Decoder topology.
#[derive(Debug, Clone, PartialEq)]
pub struct SSMMorphDecoderConfig {
    /// Bottleneck channel count.
    pub bottleneck_channels: usize,
    /// Displacement output channel count.
    pub out_channels: usize,
    /// Decoder stage count.
    pub num_stages: usize,
    /// VMamba blocks per stage.
    pub blocks_per_stage: usize,
    /// Skip-connection policy.
    pub skip_connections: SkipConnections,
}

impl SSMMorphDecoderConfig {
    /// Construct a decoder mirroring encoder channel widths.
    #[must_use]
    pub fn from_encoder(encoder_channels: &[usize], out_channels: usize) -> Self {
        Self {
            bottleneck_channels: encoder_channels.last().copied().unwrap_or(256),
            out_channels,
            num_stages: encoder_channels.len(),
            blocks_per_stage: 2,
            skip_connections: SkipConnections::Enabled,
        }
    }

    /// Derive stage channel contracts in decoding order.
    #[must_use]
    pub fn stage_configs(&self, encoder_channels: &[usize]) -> Vec<DecoderStageConfig> {
        let mut input = self.bottleneck_channels;
        encoder_channels
            .iter()
            .rev()
            .map(|&output| {
                let config = DecoderStageConfig {
                    in_channels: input,
                    out_channels: output,
                    depth: self.blocks_per_stage,
                };
                input = output;
                config
            })
            .collect()
    }
}

/// Upsampling, skip fusion, and VMamba refinement.
#[derive(Clone)]
pub struct DecoderStage<B>
where
    B: Backend + BackendOps<f32>,
{
    upsample: ConvTranspose3d<f32, B>,
    fusion: Option<Conv3d<f32, B>>,
    blocks: Vec<VMambaBlock<B>>,
    norm: LayerNorm<f32, B>,
}

impl<B> DecoderStage<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn new(
        input_channels: usize,
        skip_channels: usize,
        output_channels: usize,
        depth: usize,
        use_skip: bool,
    ) -> Self {
        let upsample =
            ConvTranspose3d::with_params(input_channels, output_channels, 4, 2, 1, 0, 1, false);
        let mut fusion = use_skip.then(|| {
            Conv3d::with_params(
                output_channels + skip_channels,
                output_channels,
                3,
                1,
                1,
                1,
                false,
            )
        });
        if let Some(layer) = &mut fusion {
            crate::initialization::convolution(layer, output_channels + skip_channels, 3, 601);
        }
        Self {
            upsample,
            fusion,
            blocks: (0..depth)
                .map(|_| VMambaBlock::new(VMambaBlockConfig::new_with_dim(output_channels)))
                .collect(),
            norm: LayerNorm::new(output_channels, 1e-5),
        }
    }

    fn forward(
        &self,
        input: &Var<f32, B>,
        skip: Option<&Var<f32, B>>,
    ) -> Result<Var<f32, B>, ModelError> {
        let upsampled = self.upsample.forward(input);
        let fused = match (&self.fusion, skip) {
            (Some(fusion), Some(skip)) => fusion.forward(&cat(&[&upsampled, skip], 1)),
            (None, None) => upsampled,
            _ => {
                return Err(ModelError::Shape {
                    operation: "DecoderStage::forward",
                    expected: "skip input matching decoder skip policy",
                    actual: upsampled.tensor.shape().to_vec(),
                });
            }
        };
        let mut output = permute(
            &self.norm.forward_nd(&permute(&fused, &[0, 2, 3, 4, 1])),
            &[0, 4, 1, 2, 3],
        );
        for block in &self.blocks {
            output = block.forward(&output)?;
        }
        Ok(output)
    }
}

impl<B> Module<f32, B> for DecoderStage<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = self.upsample.parameters();
        if let Some(layer) = &self.fusion {
            parameters.extend(layer.parameters());
        }
        for block in &self.blocks {
            parameters.extend(block.parameters());
        }
        parameters.extend(self.norm.parameters());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.forward(input, None)
            .expect("invariant: standalone decoder stage has skips disabled")
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let mut offset = load(&mut self.upsample, parameters, 0);
        if let Some(layer) = &mut self.fusion {
            offset = load(layer, parameters, offset);
        }
        for block in &mut self.blocks {
            offset = load(block, parameters, offset);
        }
        self.norm.load_parameters(&parameters[offset..]);
    }

    fn train(&mut self, mode: bool) {
        for block in &mut self.blocks {
            block.train(mode);
        }
    }
}

/// Decoder producing a dense displacement field.
#[derive(Clone)]
pub struct SSMMorphDecoder<B>
where
    B: Backend + BackendOps<f32>,
{
    stages: Vec<DecoderStage<B>>,
    output_projection: Conv3d<f32, B>,
    use_skips: bool,
}

impl<B> SSMMorphDecoder<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Initialize a decoder.
    #[must_use]
    pub fn new(config: &SSMMorphDecoderConfig, encoder_channels: &[usize]) -> Self {
        let use_skips = config.skip_connections == SkipConnections::Enabled;
        let stages = config
            .stage_configs(encoder_channels)
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                DecoderStage::new(
                    stage.in_channels,
                    encoder_channels[encoder_channels.len() - 1 - index],
                    stage.out_channels,
                    stage.depth,
                    use_skips,
                )
            })
            .collect();
        let final_channels = encoder_channels.first().copied().unwrap_or(16);
        let mut output_projection =
            Conv3d::with_params(final_channels, config.out_channels, 3, 1, 1, 1, true);
        crate::initialization::zero_convolution(&mut output_projection);
        Self {
            stages,
            output_projection,
            use_skips,
        }
    }

    /// Decode a bottleneck and encoder features into a displacement field.
    pub fn forward(
        &self,
        bottleneck: &Var<f32, B>,
        skip_features: &[Var<f32, B>],
    ) -> Result<Var<f32, B>, ModelError> {
        if self.use_skips && skip_features.len() != self.stages.len() {
            return Err(ModelError::Shape {
                operation: "SSMMorphDecoder::forward",
                expected: "one skip feature per decoder stage",
                actual: vec![skip_features.len()],
            });
        }
        let mut output = bottleneck.clone();
        for (index, stage) in self.stages.iter().enumerate() {
            let skip = self
                .use_skips
                .then(|| &skip_features[skip_features.len() - 1 - index]);
            output = stage.forward(&output, skip)?;
        }
        Ok(self.output_projection.forward(&output))
    }

    /// Return the number of upsampling stages.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

impl<B> Module<f32, B> for SSMMorphDecoder<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters: Vec<_> = self.stages.iter().flat_map(Module::parameters).collect();
        parameters.extend(self.output_projection.parameters());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.output_projection.forward(input)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let mut offset = 0;
        for stage in &mut self.stages {
            offset = load(stage, parameters, offset);
        }
        self.output_projection
            .load_parameters(&parameters[offset..]);
    }

    fn train(&mut self, mode: bool) {
        for stage in &mut self.stages {
            stage.train(mode);
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
