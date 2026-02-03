//! SSMMorph Network Architecture - U-Shaped Registration Network
//!
//! Complete end-to-end network combining encoder, bottleneck, and decoder
//! for deformable medical image registration. Based on VMambaMorph and
//! MambaBIR architectures using State Space Models.
//!
//! # Architecture Overview
//!
//! ```text
//! Input: [fixed, moving] (concatenated, 2 channels)
//!          │
//!          ▼
//!     ┌─────────┐
//!     │ Encoder │──► Multi-scale features (skip connections)
//!     └─────────┘         [32, 64, 128, 256] channels
//!          │
//!          ▼
//!     ┌─────────┐
//!     │Bottleneck│
//!     └─────────┘
//!          │
//!          ▼
//!     ┌─────────┐
//!     │ Decoder │◄── Skip connections from encoder
//!     └─────────┘
//!          │
//!          ▼
//!   ┌──────────────┐
//!   │   Optional   │
//!   │ Diffeomorphic│ (velocity field integration)
//!   │  Integration │
//!   └──────────────┘
//!          │
//!          ▼
//!   Displacement Field [batch, 3, D, H, W]
//! ```
//!
//! # Key Features
//!
//! - **Hierarchical Encoding**: Multi-scale feature extraction with VMamba blocks
//! - **Skip Connections**: Preserve spatial detail from encoder to decoder
//! - **Diffeomorphic Option**: Ensures topology-preserving transformations
//! - **3D Native**: Designed for volumetric medical images
//!
//! References:
//! - "VMambaMorph: Multi-Modality Deformable Image Registration based on VSSM"
//! - "MambaBIR: Residual Pyramid Network for Brain Image Registration with SSM"

use burn::prelude::*;

use crate::ssmmorph::encoder::{SSMMorphEncoder, SSMMorphEncoderConfig};
use crate::ssmmorph::decoder::{SSMMorphDecoder, SSMMorphDecoderConfig};

use super::integration::{IntegrationConfig, VelocityFieldIntegrator};
use burn::module::Ignored;

/// Output from SSMMorph forward pass
#[derive(Debug, Clone)]
pub struct SSMMorphOutput<B: Backend> {
    /// Final displacement field [batch, 3, D, H, W]
    pub displacement: Tensor<B, 5>,
    /// Multi-scale encoder features for analysis
    pub encoder_features: Vec<Tensor<B, 5>>,
    /// Bottleneck features
    pub bottleneck: Tensor<B, 5>,
}

/// Configuration for SSMMorph network
#[derive(Config, Debug, PartialEq)]
pub struct SSMMorphConfig {
    /// Encoder configuration
    pub encoder: SSMMorphEncoderConfig,
    /// Decoder configuration (auto-derived if None)
    pub decoder: Option<SSMMorphDecoderConfig>,
    /// Output channels (3 for 3D displacement field)
    #[config(default = "3")]
    pub out_channels: usize,
    /// Use diffeomorphic integration
    #[config(default = "true")]
    pub diffeomorphic: bool,
    /// Integration steps for diffeomorphic transformation
    #[config(default = "7")]
    pub integration_steps: usize,
}

impl SSMMorphConfig {
    /// Create configuration for 3D registration
    pub fn for_3d_registration() -> Self {
        let encoder = SSMMorphEncoderConfig::for_registration();
        Self {
            encoder,
            decoder: None,
            out_channels: 3,
            diffeomorphic: true,
            integration_steps: 7,
        }
    }

    /// Create lightweight configuration for faster inference
    pub fn lightweight() -> Self {
        let encoder = SSMMorphEncoderConfig::lightweight();
        Self {
            encoder,
            decoder: None,
            out_channels: 3,
            diffeomorphic: true,
            integration_steps: 7,
        }
    }

    /// Create high-quality configuration for research
    pub fn high_quality() -> Self {
        let encoder = SSMMorphEncoderConfig::high_quality();
        Self {
            encoder,
            decoder: None,
            out_channels: 3,
            diffeomorphic: true,
            integration_steps: 10,
        }
    }

    /// Disable diffeomorphic integration
    pub fn without_diffeomorphic(mut self) -> Self {
        self.diffeomorphic = false;
        self
    }

    /// Set custom integration steps
    pub fn set_integration_steps(mut self, steps: usize) -> Self {
        self.integration_steps = steps;
        self
    }
}

/// SSMMorph Network for deformable image registration
///
/// U-shaped network that takes fixed and moving images as input
/// and outputs a displacement field for registration.
#[derive(Module, Debug)]
pub struct SSMMorph<B: Backend> {
    /// Hierarchical feature encoder
    pub encoder: SSMMorphEncoder<B>,
    /// Hierarchical decoder with skip connections
    pub decoder: SSMMorphDecoder<B>,
    /// Use diffeomorphic integration
    diffeomorphic: Ignored<bool>,
    /// Integration steps
    integration_steps: Ignored<usize>,
}

impl<B: Backend> SSMMorph<B> {
    /// Create new SSMMorph network
    pub fn new(config: &SSMMorphConfig, device: &B::Device) -> Self {
        // Create encoder
        let encoder = SSMMorphEncoder::new(&config.encoder, device);

        // Derive or use provided decoder config
        let decoder_config = config.decoder.clone().unwrap_or_else(|| {
            SSMMorphDecoderConfig::from_encoder(
                encoder.stage_channels(),
                config.out_channels,
            )
        });

        // Create decoder
        let decoder = SSMMorphDecoder::new(
            &decoder_config,
            encoder.stage_channels(),
            device,
        );

        Self {
            encoder,
            decoder,
            diffeomorphic: Ignored(config.diffeomorphic),
            integration_steps: Ignored(config.integration_steps),
        }
    }

    /// Forward pass through network
    ///
    /// # Arguments
    /// * `fixed` - Fixed/reference image [batch, 1, D, H, W]
    /// * `moving` - Moving image to register [batch, 1, D, H, W]
    ///
    /// # Returns
    /// * Network output containing displacement field and features
    pub fn forward(&self, fixed: Tensor<B, 5>, moving: Tensor<B, 5>) -> SSMMorphOutput<B> {
        // Concatenate along channel dimension
        let input = Tensor::cat(vec![fixed, moving], 1);
        self.forward_combined(input)
    }

    /// Forward with pre-concatenated input
    ///
    /// Useful when input is already prepared by dataloader.
    /// Input should be [batch, 2, D, H, W] with fixed and moving concatenated.
    pub fn forward_combined(&self, input: Tensor<B, 5>) -> SSMMorphOutput<B> {
        // Encode
        let (encoder_features, bottleneck) = self.encoder.forward(input);

        // Decode
        let displacement: Tensor<B, 5> = self.decoder.forward(
            bottleneck.clone(),
            &encoder_features,
        );

        // Apply diffeomorphic integration if enabled
        let displacement = if *self.diffeomorphic {
            let integrator = VelocityFieldIntegrator::new(
                IntegrationConfig::with_steps(*self.integration_steps),
                displacement.device(),
            );
            integrator.integrate(displacement)
        } else {
            displacement
        };

        SSMMorphOutput {
            displacement,
            encoder_features: encoder_features.to_vec(),
            bottleneck,
        }
    }

    /// Get encoder stage channel dimensions
    pub fn encoder_channels(&self) -> &[usize] {
        self.encoder.stage_channels()
    }

    /// Get number of encoder stages
    pub fn num_stages(&self) -> usize {
        self.encoder_channels().len()
    }

    /// Check if network uses diffeomorphic integration
    pub fn is_diffeomorphic(&self) -> bool {
        *self.diffeomorphic
    }
}

/// Preset configurations for common use cases
pub mod presets {
    use super::*;

    /// Configuration for brain MRI registration (T1/T2-weighted)
    ///
    /// Optimized for:
    /// - 256x256x256 isotropic 1mm volumes
    /// - Soft tissue contrast
    /// - High accuracy requirements
    pub fn brain_mri() -> SSMMorphConfig {
        SSMMorphConfig::for_3d_registration()
            .set_integration_steps(10)
    }

    /// Configuration for abdominal CT registration
    ///
    /// Optimized for:
    /// - Larger fields of view
    /// - Variable contrast
    /// - Organ deformation
    pub fn abdominal_ct() -> SSMMorphConfig {
        SSMMorphConfig {
            encoder: SSMMorphEncoderConfig {
                in_channels: 2,
                base_channels: 48,
                channel_mult: 2,
                num_stages: 5,
                blocks_per_stage: 2,
                use_drop_path: false,
            },
            decoder: None,
            out_channels: 3,
            diffeomorphic: true,
            integration_steps: 7,
        }
    }

    /// Configuration for cardiac MRI
    ///
    /// Optimized for:
    /// - Temporal sequences
    /// - Fast inference needs
    /// - Smaller volumes
    pub fn cardiac_mri() -> SSMMorphConfig {
        SSMMorphConfig::lightweight()
            .set_integration_steps(5)
    }

    /// Configuration for real-time applications
    ///
    /// Fastest configuration with acceptable quality.
    /// Disables diffeomorphic integration for speed.
    pub fn realtime() -> SSMMorphConfig {
        SSMMorphConfig::lightweight()
            .without_diffeomorphic()
    }

    /// Configuration for research/high accuracy
    ///
    /// Maximum quality at the cost of computation time.
    pub fn research() -> SSMMorphConfig {
        SSMMorphConfig::high_quality()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_ssmmorph_creation() {
        let device = Default::default();
        let config = SSMMorphConfig::for_3d_registration();
        let network = SSMMorph::<TestBackend>::new(&config, &device);

        assert_eq!(network.encoder_channels(), &[32, 64, 128, 256]);
        assert!(network.is_diffeomorphic());
    }

    #[test]
    fn test_ssmmorph_forward() {
        let device = Default::default();
        let config = SSMMorphConfig::for_3d_registration();
        let network = SSMMorph::<TestBackend>::new(&config, &device);

        // Input images (reduced size for test performance, but depth must be >= 16)
        let fixed = Tensor::<TestBackend, 5>::zeros([1, 1, 16, 16, 16], &device);
        let moving = Tensor::<TestBackend, 5>::zeros([1, 1, 16, 16, 16], &device);

        let output = network.forward(fixed, moving);

        // Output displacement field should have 3 channels
        assert_eq!(output.displacement.dims(), [1, 3, 16, 16, 16]);

        // Should have features from each encoder stage
        assert_eq!(output.encoder_features.len(), 4);
    }

    #[test]
    fn test_ssmmorph_forward_combined() {
        let device = Default::default();
        let config = SSMMorphConfig::for_3d_registration();
        let network = SSMMorph::<TestBackend>::new(&config, &device);

        // Pre-concatenated input (reduced size)
        let input = Tensor::<TestBackend, 5>::zeros([1, 2, 16, 16, 16], &device);
        let output = network.forward_combined(input);

        assert_eq!(output.displacement.dims(), [1, 3, 16, 16, 16]);
    }

    #[test]
    fn test_ssmmorph_presets() {
        let device = Default::default();

        // Brain MRI preset
        let brain_config = presets::brain_mri();
        let brain_network = SSMMorph::<TestBackend>::new(&brain_config, &device);
        assert!(brain_network.is_diffeomorphic());
        assert_eq!(brain_network.encoder_channels().len(), 4);

        // Realtime preset
        let realtime_config = presets::realtime();
        let realtime_network = SSMMorph::<TestBackend>::new(&realtime_config, &device);
        assert!(!realtime_network.is_diffeomorphic());

        // Research preset
        let research_config = presets::research();
        let research_network = SSMMorph::<TestBackend>::new(&research_config, &device);
        assert!(research_network.is_diffeomorphic());
        assert_eq!(research_network.encoder_channels().len(), 4);
    }

    #[test]
    fn test_nondiffeomorphic_mode() {
        let device = Default::default();
        let config = SSMMorphConfig::for_3d_registration()
            .without_diffeomorphic();
        let network = SSMMorph::<TestBackend>::new(&config, &device);

        assert!(!network.is_diffeomorphic());

        let fixed = Tensor::<TestBackend, 5>::zeros([1, 1, 16, 16, 16], &device);
        let moving = Tensor::<TestBackend, 5>::zeros([1, 1, 16, 16, 16], &device);

        let output = network.forward(fixed, moving);
        assert_eq!(output.displacement.dims(), [1, 3, 16, 16, 16]);
    }
}
