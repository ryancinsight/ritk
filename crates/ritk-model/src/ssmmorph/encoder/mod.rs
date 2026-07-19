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
//!          â”‚
//!          â–¼
//!    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
//!    â”‚   Stage 0   â”‚â”€â”€â–º Skip 0: [batch, 32, D, H, W]
//!    â”‚  2 blocks   â”‚
//!    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
//!          â”‚ downsample
//!          â–¼
//!    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
//!    â”‚   Stage 1   â”‚â”€â”€â–º Skip 1: [batch, 64, D/2, H/2, W/2]
//!    â”‚  2 blocks   â”‚
//!    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
//!          â”‚ downsample
//!          â–¼
//!    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
//!    â”‚   Stage 2   â”‚â”€â”€â–º Skip 2: [batch, 128, D/4, H/4, W/4]
//!    â”‚  2 blocks   â”‚
//!    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
//!          â”‚ downsample
//!          â–¼
//!    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
//!    â”‚   Stage 3   â”‚â”€â”€â–º Skip 3: [batch, 256, D/8, H/8, W/8]
//!    â”‚  2 blocks   â”‚
//!    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
//!          â”‚
//!          â–¼
//!    Bottleneck: [batch, 256, D/8, H/8, W/8]
//! ```

pub mod config;
pub mod stage;

pub use config::{DownsamplePolicy, DropPath, EncoderStageConfig, SSMMorphEncoderConfig};
pub use stage::{EncoderStage, EncoderStageOutput};

use crate::ModelError;
use coeus_autograd::Var;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;

/// SSMMorph Encoder - Hierarchical feature extraction
#[derive(Clone)]
pub struct SSMMorphEncoder<B>
where
    B: Backend + BackendOps<f32>,
{
    /// Encoder stages
    pub stages: Vec<EncoderStage<B>>,
    /// Number of stages (stored, not config)
    num_stages: usize,
    /// Stage output channels
    stage_channels: Vec<usize>,
}

/// Multi-scale encoder result.
pub struct SSMMorphEncoderOutput<B>
where
    B: Backend,
{
    /// Features retained for decoder skip connections.
    pub features: Vec<Var<f32, B>>,
    /// Lowest-resolution representation.
    pub bottleneck: Var<f32, B>,
}

impl<B> SSMMorphEncoder<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Create new SSMMorph encoder
    pub fn new(config: &SSMMorphEncoderConfig) -> Self {
        let stage_configs = config.stage_configs();
        let stage_channels = config.stage_channels();

        let stages: Vec<_> = stage_configs
            .iter()
            .copied()
            .map(EncoderStage::new)
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
    pub fn forward(&self, input: &Var<f32, B>) -> Result<SSMMorphEncoderOutput<B>, ModelError> {
        let mut x = input.clone();
        let mut features = Vec::with_capacity(self.stages.len());

        for stage in &self.stages {
            let output = stage.forward(&x)?;
            features.push(output.features.clone());
            x = output.continuation.unwrap_or(output.features);
        }

        // The final output is the bottleneck
        let bottleneck = x;

        Ok(SSMMorphEncoderOutput {
            features,
            bottleneck,
        })
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

impl<B> Module<f32, B> for SSMMorphEncoder<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        self.stages.iter().flat_map(Module::parameters).collect()
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        SSMMorphEncoder::forward(self, input)
            .expect("invariant: encoder module receives a valid volume")
            .bottleneck
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let mut offset = 0;
        for stage in &mut self.stages {
            let count = stage.parameters().len();
            stage.load_parameters(&parameters[offset..offset + count]);
            offset += count;
        }
    }

    fn train(&mut self, mode: bool) {
        for stage in &mut self.stages {
            stage.train(mode);
        }
    }
}
