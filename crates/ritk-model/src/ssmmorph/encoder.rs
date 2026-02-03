//! SSMMorph Encoder - Hierarchical Feature Extraction
//!
//! Multi-scale encoder using VMamba blocks for hierarchical feature
//! extraction. The encoder progressively downsamples the input while
//! extracting features at multiple scales for skip connections.
//!
//! # Architecture
//!
//! ```text
//! Input: [batch, 2, D, H, W] (fixed + moving concatenated)
//!          │
//!          ▼
//!    ┌─────────────┐
//!    │   Stage 0   │──► Skip 0: [batch, 32, D, H, W]
//!    │  2 blocks   │
//!    └─────────────┘
//!          │ downsample
//!          ▼
//!    ┌─────────────┐
//!    │   Stage 1   │──► Skip 1: [batch, 64, D/2, H/2, W/2]
//!    │  2 blocks   │
//!    └─────────────┘
//!          │ downsample
//!          ▼
//!    ┌─────────────┐
//!    │   Stage 2   │──► Skip 2: [batch, 128, D/4, H/4, W/4]
//!    │  2 blocks   │
//!    └─────────────┘
//!          │ downsample
//!          ▼
//!    ┌─────────────┐
//!    │   Stage 3   │──► Skip 3: [batch, 256, D/8, H/8, W/8]
//!    │  2 blocks   │
//!    └─────────────┘
//!          │
//!          ▼
//!    Bottleneck: [batch, 256, D/8, H/8, W/8]
//! ```

use burn::prelude::*;
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::PaddingConfig3d;

use super::vmamba_block::{VMambaBlock, VMambaBlockConfig};

/// Configuration for encoder stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncoderStageConfig {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Number of VMamba blocks in this stage
    pub depth: usize,
    /// Whether to downsample at the end of this stage
    pub downsample: bool,
}

/// Configuration for SSMMorph encoder
#[derive(Config, Debug, PartialEq)]
pub struct SSMMorphEncoderConfig {
    /// Number of input channels (2 for fixed + moving)
    #[config(default = "2")]
    pub in_channels: usize,
    /// Base channel dimension for first stage
    #[config(default = "32")]
    pub base_channels: usize,
    /// Channel multiplier between stages
    #[config(default = "2")]
    pub channel_mult: usize,
    /// Number of encoder stages
    #[config(default = "4")]
    pub num_stages: usize,
    /// Number of VMamba blocks per stage
    #[config(default = "2")]
    pub blocks_per_stage: usize,
    /// Use drop path (stochastic depth)
    #[config(default = "false")]
    pub use_drop_path: bool,
}

impl SSMMorphEncoderConfig {
    /// Create configuration for standard 3D registration
    pub fn for_registration() -> Self {
        Self {
            in_channels: 2,
            base_channels: 32,
            channel_mult: 2,
            num_stages: 4,
            blocks_per_stage: 2,
            use_drop_path: false,
        }
    }

    /// Create lightweight configuration
    pub fn lightweight() -> Self {
        Self {
            in_channels: 2,
            base_channels: 16,
            channel_mult: 2,
            num_stages: 3,
            blocks_per_stage: 1,
            use_drop_path: false,
        }
    }

    /// Create high-quality configuration
    pub fn high_quality() -> Self {
        Self {
            in_channels: 2,
            base_channels: 48,
            channel_mult: 2,
            num_stages: 4,
            blocks_per_stage: 3,
            use_drop_path: true,
        }
    }

    /// Get stage configurations
    pub fn stage_configs(&self) -> Vec<EncoderStageConfig> {
        let mut configs = Vec::with_capacity(self.num_stages);
        let mut in_ch = self.in_channels;
        let mut out_ch = self.base_channels;

        for _ in 0..self.num_stages {
            // We downsample at every stage to produce a bottleneck that is 
            // smaller than the last feature map. This ensures the decoder
            // can upsample the bottleneck to match the last skip connection.
            let downsample = true;

            configs.push(EncoderStageConfig {
                in_channels: in_ch,
                out_channels: out_ch,
                depth: self.blocks_per_stage,
                downsample,
            });

            in_ch = out_ch;
            out_ch *= self.channel_mult;
        }

        configs
    }

    /// Get channel dimensions for each stage
    pub fn stage_channels(&self) -> Vec<usize> {
        self.stage_configs()
            .into_iter()
            .map(|c| c.out_channels)
            .collect()
    }
}

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
            let proj_config = Conv3dConfig::new(
                [config.in_channels, config.out_channels],
                [3, 3, 3],
            )
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
            let ds_config = Conv3dConfig::new(
                [config.out_channels, config.out_channels],
                [3, 3, 3],
            )
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

/// SSMMorph Encoder - Hierarchical feature extraction
#[derive(Module, Debug)]
pub struct SSMMorphEncoder<B: Backend> {
    /// Encoder stages
    pub stages: Vec<EncoderStage<B>>,
    /// Number of stages (stored, not config)
    num_stages: usize,
    /// Stage output channels
    #[module(ignore)]
    stage_channels: Vec<usize>,
}

impl<B: Backend> SSMMorphEncoder<B> {
    /// Create new SSMMorph encoder
    pub fn new(config: &SSMMorphEncoderConfig, device: &B::Device) -> Self {
        let stage_configs = config.stage_configs();
        let stage_channels = config.stage_channels();

        let stages: Vec<_> = stage_configs
            .iter()
            .map(|cfg| EncoderStage::new(cfg, device))
            .collect();

        Self {
            stages,
            num_stages: config.num_stages,
            stage_channels,
        }
    }

    /// Get number of channels at each stage
    pub fn channels(&self) -> &[usize] {
        &self.stage_channels
    }

    /// Forward pass through encoder
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, in_channels, D, H, W]
    ///
    /// # Returns
    /// * features: Multi-scale features from each stage (for skip connections)
    /// * bottleneck: Bottleneck features at lowest resolution
    pub fn forward(&self, input: Tensor<B, 5>) -> (Vec<Tensor<B, 5>>, Tensor<B, 5>) {
        let mut x = input;
        let mut features = Vec::with_capacity(self.stages.len());

        for stage in &self.stages {
            let (feat, out) = stage.forward(x);
            features.push(feat.clone());
            x = out.unwrap_or(feat);
        }

        // The final output is the bottleneck
        let bottleneck = x;

        (features, bottleneck)
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Get output channel dimensions for each stage
    pub fn stage_channels(&self) -> &[usize] {
        &self.stage_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_encoder_config() {
        let config = SSMMorphEncoderConfig::for_registration();
        assert_eq!(config.num_stages, 4);
        assert_eq!(config.base_channels, 32);

        let channels = config.stage_channels();
        assert_eq!(channels, vec![32, 64, 128, 256]);
    }

    #[test]
    fn test_encoder_stage_creation() {
        let device = Default::default();
        let config = EncoderStageConfig {
            in_channels: 2,
            out_channels: 32,
            depth: 2,
            downsample: true,
        };

        let stage = EncoderStage::<TestBackend>::new(&config, &device);
        assert_eq!(stage.blocks.len(), 2);
        assert!(stage.downsample.is_some());
        assert!(stage.proj.is_some());
    }

    #[test]
    fn test_encoder_stage_forward() {
        let device = Default::default();
        let config = EncoderStageConfig {
            in_channels: 2,
            out_channels: 32,
            depth: 1,
            downsample: true,
        };

        let stage = EncoderStage::<TestBackend>::new(&config, &device);

        // Input: [batch=1, channels=2, depth=16, height=64, width=64]
        let input = Tensor::<TestBackend, 5>::zeros([1, 2, 16, 64, 64], &device);
        let (features, output) = stage.forward(input);

        // Features should have output channels and same spatial size
        assert_eq!(features.dims(), [1, 32, 16, 64, 64]);

        // Output should be downsampled
        assert!(output.is_some());
        let out = output.unwrap();
        assert_eq!(out.dims(), [1, 32, 8, 32, 32]);
    }

    #[test]
    fn test_encoder_creation() {
        let device = Default::default();
        let config = SSMMorphEncoderConfig::for_registration();
        let encoder = SSMMorphEncoder::<TestBackend>::new(&config, &device);

        assert_eq!(encoder.num_stages(), 4);
        assert_eq!(encoder.stage_channels(), &[32, 64, 128, 256]);
    }

    #[test]
    fn test_encoder_forward() {
        let device = Default::default();
        let config = SSMMorphEncoderConfig::lightweight();
        let encoder = SSMMorphEncoder::<TestBackend>::new(&config, &device);

        // Input: [batch=1, channels=2, depth=16, height=64, width=64]
        let input = Tensor::<TestBackend, 5>::zeros([1, 2, 16, 64, 64], &device);
        let (features, bottleneck) = encoder.forward(input);

        // Should have features from each stage
        assert_eq!(features.len(), 3);

        // Check feature shapes (lightweight config: 3 stages)
        assert_eq!(features[0].dims(), [1, 16, 16, 64, 64]);
        assert_eq!(features[1].dims(), [1, 32, 8, 32, 32]);
        assert_eq!(features[2].dims(), [1, 64, 4, 16, 16]);

        // Bottleneck should be downsampled from last stage
        assert_eq!(bottleneck.dims(), [1, 64, 2, 8, 8]);
    }

    #[test]
    fn test_encoder_presets() {
        let reg_config = SSMMorphEncoderConfig::for_registration();
        assert_eq!(reg_config.num_stages, 4);
        assert_eq!(reg_config.base_channels, 32);

        let lightweight_config = SSMMorphEncoderConfig::lightweight();
        assert_eq!(lightweight_config.num_stages, 3);
        assert_eq!(lightweight_config.base_channels, 16);

        let hq_config = SSMMorphEncoderConfig::high_quality();
        assert_eq!(hq_config.num_stages, 4);
        assert_eq!(hq_config.base_channels, 48);
        assert!(hq_config.use_drop_path);
    }
}

