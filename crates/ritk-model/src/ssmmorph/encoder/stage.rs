//! Functional bounds spanning VMamba block executions individually.
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::PaddingConfig3d;
use burn::prelude::*;

use super::config::EncoderStageConfig;
use crate::ssmmorph::vmamba_block::{VMambaBlock, VMambaBlockConfig};

/// Single encoder stage with VMamba blocks and optional downsampling
#[derive(Module, Debug)]
pub struct EncoderStage<B: Backend> {
    /// Initial projection convolution (for channel change)
    pub proj: Option<Conv3d<B>>,
    /// VMamba blocks for feature processing
    pub blocks: Vec<VMambaBlock<B>>,
    /// Downsampling convolution (if applicable)
    pub downsample: Option<Conv3d<B>>,
    /// Output channels
    pub out_channels: usize,
    /// Whether this stage downsamples
    pub has_downsample: bool,
}

impl<B: Backend> EncoderStage<B> {
    /// Create new encoder stage
    pub fn new(config: &EncoderStageConfig, device: &B::Device) -> Self {
        // Initial projection if channels change
        let proj = if config.in_channels != config.out_channels {
            let proj_config =
                Conv3dConfig::new([config.in_channels, config.out_channels], [3, 3, 3])
                    .with_stride([1, 1, 1])
                    .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                    .with_bias(false);
            Some(proj_config.init(device))
        } else {
            None
        };

        // VMamba blocks
        let block_config = VMambaBlockConfig::new_with_dim(config.out_channels);
        let blocks: Vec<_> = (0..config.depth)
            .map(|_| VMambaBlock::new(&block_config, device))
            .collect();

        // Downsampling layer
        let downsample = if config.downsample {
            let ds_config =
                Conv3dConfig::new([config.out_channels, config.out_channels], [3, 3, 3])
                    .with_stride([2, 2, 2])
                    .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                    .with_bias(false);
            Some(ds_config.init(device))
        } else {
            None
        };

        Self {
            proj,
            blocks,
            downsample,
            out_channels: config.out_channels,
            has_downsample: config.downsample,
        }
    }

    /// Forward pass through encoder stage
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, in_channels, D, H, W]
    ///
    /// # Returns
    /// * features: Output features before downsampling [batch, out_channels, D, H, W]
    /// * downsampled: Downsampled features [batch, out_channels, D/2, H/2, W/2]
    pub fn forward(&self, input: Tensor<B, 5>) -> (Tensor<B, 5>, Option<Tensor<B, 5>>) {
        let mut x = input;

        // Initial projection if needed
        if let Some(ref proj) = self.proj {
            x = proj.forward(x);
        }

        // Pass through VMamba blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Store features before downsampling for skip connection
        let features = x.clone();

        // Downsample if configured
        let output = self.downsample.as_ref().map(|ds| ds.forward(x));

        (features, output)
    }
}
