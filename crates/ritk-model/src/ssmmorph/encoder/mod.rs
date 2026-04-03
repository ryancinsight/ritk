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

pub mod config;
pub mod stage;

pub use config::{EncoderStageConfig, SSMMorphEncoderConfig};
pub use stage::EncoderStage;

use burn::prelude::*;

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
mod tests;
