//! Configuration mappings for hierarchical encodings 
use burn::prelude::*;

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
