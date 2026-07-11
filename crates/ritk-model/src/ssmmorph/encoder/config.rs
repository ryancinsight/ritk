//! Configuration mappings for hierarchical encodings
/// Whether an encoder stage performs spatial downsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DownsamplePolicy {
    /// Preserve spatial resolution through this stage.
    KeepResolution,
    /// Apply strided convolution to halve spatial dimensions.
    #[default]
    Downsample,
}

/// Whether stochastic depth (drop-path) is applied during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum DropPath {
    /// Drop-path is not applied.
    #[default]
    Disabled,
    /// Drop-path is applied at the configured rate.
    Enabled,
}

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
    pub downsample: DownsamplePolicy,
}

/// Configuration for SSMMorph encoder
#[derive(Debug, Clone, PartialEq)]
pub struct SSMMorphEncoderConfig {
    /// Number of input channels (2 for fixed + moving)
    pub in_channels: usize,
    /// Base channel dimension for first stage
    pub base_channels: usize,
    /// Channel multiplier between stages
    pub channel_mult: usize,
    /// Number of encoder stages
    pub num_stages: usize,
    /// Number of VMamba blocks per stage
    pub blocks_per_stage: usize,
    /// Use drop path (stochastic depth)
    pub drop_path: DropPath,
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
            drop_path: DropPath::Disabled,
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
            drop_path: DropPath::Disabled,
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
            drop_path: DropPath::Enabled,
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
            let downsample = DownsamplePolicy::Downsample;

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
