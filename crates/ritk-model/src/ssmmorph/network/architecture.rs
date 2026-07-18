//! U-shaped VMamba registration network.

use coeus_autograd::{cat, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;

use crate::ssmmorph::decoder::{SSMMorphDecoder, SSMMorphDecoderConfig};
use crate::ssmmorph::encoder::{DropPath, SSMMorphEncoder, SSMMorphEncoderConfig};
use crate::transmorph::VecInt;
use crate::ModelError;

/// Displacement-field integration policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrationMode {
    /// Return the decoder displacement directly.
    Direct,
    /// Integrate the decoder velocity by scaling and squaring.
    #[default]
    Diffeomorphic }

/// SSMMorph forward result.
#[derive(Clone)]
pub struct SSMMorphOutput<B>
where
    B: Backend + BackendOps<f32>,
{
    /// Final `[batch, 3, depth, height, width]` displacement.
    pub displacement: Var<f32, B>,
    /// Encoder features in increasing channel order.
    pub encoder_features: Vec<Var<f32, B>>,
    /// Lowest-resolution encoder representation.
    pub bottleneck: Var<f32, B> }

/// SSMMorph topology and integration configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SSMMorphConfig {
    /// Encoder configuration.
    pub encoder: SSMMorphEncoderConfig,
    /// Optional explicit decoder configuration.
    pub decoder: Option<SSMMorphDecoderConfig>,
    /// Displacement channel count.
    pub out_channels: usize,
    /// Integration policy.
    pub integration: IntegrationMode,
    /// Scaling-and-squaring stage count.
    pub integration_steps: usize }

impl SSMMorphConfig {
    /// Standard volumetric registration configuration.
    #[must_use]
    pub fn for_3d_registration() -> Self {
        Self {
            encoder: SSMMorphEncoderConfig::for_registration(),
            decoder: None,
            out_channels: 3,
            integration: IntegrationMode::Diffeomorphic,
            integration_steps: 7 }
    }

    /// Reduced-width inference configuration.
    #[must_use]
    pub fn lightweight() -> Self {
        Self {
            encoder: SSMMorphEncoderConfig::lightweight(),
            decoder: None,
            out_channels: 3,
            integration: IntegrationMode::Diffeomorphic,
            integration_steps: 7 }
    }

    /// Wider research configuration.
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            encoder: SSMMorphEncoderConfig::high_quality(),
            decoder: None,
            out_channels: 3,
            integration: IntegrationMode::Diffeomorphic,
            integration_steps: 10 }
    }

    /// Select direct decoder displacement output.
    #[must_use]
    pub const fn without_diffeomorphic(mut self) -> Self {
        self.integration = IntegrationMode::Direct;
        self
    }

    /// Set scaling-and-squaring stages.
    #[must_use]
    pub const fn set_integration_steps(mut self, steps: usize) -> Self {
        self.integration_steps = steps;
        self
    }
}

/// End-to-end VMamba registration model.
#[derive(Clone)]
pub struct SSMMorph<B>
where
    B: Backend + BackendOps<f32>,
{
    encoder: SSMMorphEncoder<B>,
    decoder: SSMMorphDecoder<B>,
    integrator: Option<VecInt> }

impl<B> SSMMorph<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Initialize the registration model.
    #[must_use]
    pub fn new(config: &SSMMorphConfig) -> Self {
        let encoder = SSMMorphEncoder::new(&config.encoder);
        let decoder_config = config.decoder.clone().unwrap_or_else(|| {
            SSMMorphDecoderConfig::from_encoder(encoder.stage_channels(), config.out_channels)
        });
        let decoder = SSMMorphDecoder::new(&decoder_config, encoder.stage_channels());
        Self {
            encoder,
            decoder,
            integrator: (config.integration == IntegrationMode::Diffeomorphic)
                .then(|| VecInt::new(config.integration_steps)) }
    }

    /// Register a moving image to a fixed image.
    pub fn forward(
        &self,
        fixed: &Var<f32, B>,
        moving: &Var<f32, B>,
    ) -> Result<SSMMorphOutput<B>, ModelError> {
        self.forward_combined(&cat(&[fixed, moving], 1))
    }

    /// Evaluate a pre-concatenated `[fixed, moving]` input.
    pub fn forward_combined(&self, input: &Var<f32, B>) -> Result<SSMMorphOutput<B>, ModelError> {
        let encoded = self.encoder.forward(input)?;
        let velocity = self
            .decoder
            .forward(&encoded.bottleneck, &encoded.features)?;
        let displacement = match &self.integrator {
            Some(integrator) => integrator.forward(&velocity),
            None => velocity };
        Ok(SSMMorphOutput {
            displacement,
            encoder_features: encoded.features,
            bottleneck: encoded.bottleneck })
    }

    /// Encoder channel widths.
    #[must_use]
    pub fn encoder_channels(&self) -> &[usize] {
        self.encoder.stage_channels()
    }

    /// Encoder stage count.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.encoder.num_stages()
    }

    /// Whether scaling-and-squaring is active.
    #[must_use]
    pub const fn is_diffeomorphic(&self) -> bool {
        self.integrator.is_some()
    }
}

impl<B> Module<f32, B> for SSMMorph<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = self.encoder.parameters();
        parameters.extend(self.decoder.parameters());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.forward_combined(input)
            .expect("invariant: module input satisfies SSMMorph contract")
            .displacement
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let encoder_count = self.encoder.parameters().len();
        self.encoder.load_parameters(&parameters[..encoder_count]);
        self.decoder.load_parameters(&parameters[encoder_count..]);
    }

    fn train(&mut self, mode: bool) {
        self.encoder.train(mode);
        self.decoder.train(mode);
    }
}

/// Preset configurations.
pub mod presets {
    use super::*;

    /// Brain MRI registration.
    #[must_use]
    pub fn brain_mri() -> SSMMorphConfig {
        SSMMorphConfig::for_3d_registration().set_integration_steps(10)
    }

    /// Abdominal CT registration.
    #[must_use]
    pub fn abdominal_ct() -> SSMMorphConfig {
        SSMMorphConfig {
            encoder: SSMMorphEncoderConfig {
                in_channels: 2,
                base_channels: 48,
                channel_mult: 2,
                num_stages: 5,
                blocks_per_stage: 2,
                drop_path: DropPath::Disabled },
            decoder: None,
            out_channels: 3,
            integration: IntegrationMode::Diffeomorphic,
            integration_steps: 7 }
    }

    /// Cardiac MRI registration.
    #[must_use]
    pub fn cardiac_mri() -> SSMMorphConfig {
        SSMMorphConfig::lightweight().set_integration_steps(5)
    }

    /// Direct-displacement real-time configuration.
    #[must_use]
    pub fn realtime() -> SSMMorphConfig {
        SSMMorphConfig::lightweight().without_diffeomorphic()
    }

    /// High-capacity research configuration.
    #[must_use]
    pub fn research() -> SSMMorphConfig {
        SSMMorphConfig::high_quality()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn complete_graph_restores_resolution_and_gradients() {
        let config = SSMMorphConfig {
            encoder: SSMMorphEncoderConfig {
                in_channels: 2,
                base_channels: 2,
                channel_mult: 2,
                num_stages: 2,
                blocks_per_stage: 1,
                drop_path: DropPath::Disabled },
            decoder: None,
            out_channels: 3,
            integration: IntegrationMode::Direct,
            integration_steps: 2 };
        let model = SSMMorph::<MoiraiBackend>::new(&config);
        let input = Var::new(
            Tensor::ones_on([1, 2, 4, 4, 4], &MoiraiBackend::new()),
            true,
        );

        let output = model
            .forward_combined(&input)
            .expect("test volume satisfies the model contract");

        assert_eq!(output.displacement.tensor.shape(), &[1, 3, 4, 4, 4]);
        assert_eq!(output.encoder_features.len(), 2);
        assert!(output
            .displacement
            .tensor
            .as_slice()
            .iter()
            .all(|&value| value == 0.0));
        output.displacement.backward();
        assert!(
            input.grad().is_some(),
            "complete model graph must remain connected"
        );
    }
}
