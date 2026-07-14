//! TransMorph configuration and builder, Coeus-native.
//!
//! [`TransMorphConfig`] specifies the channel widths and window/integration
//! parameters, and [`TransMorphConfig::init`] instantiates a [`TransMorph`] with
//! Kaiming-uniform-initialized convolutions (fan-in scaled, non-degenerate — the
//! scheme the original relied on; the raw [`Conv3d`] constructor alone leaves
//! weights at ones).

use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Conv3d;
use coeus_ops::BackendOps;

use crate::transmorph::{
    integration::VecInt, model::TransMorph, spatial_transform::SpatialTransformer,
    swin::SwinTransformerBlock,
};

/// Convolution dilation (isotropic, unit).
const DILATION: usize = 1;
/// Attention heads per Swin block.
const NUM_HEADS: usize = 4;
/// MLP hidden-width ratio in each Swin block.
const MLP_RATIO: f64 = 4.0;
/// Base seed for deterministic weight initialization.
const INIT_SEED: u64 = 0x0BAD_F00D;
/// Golden-ratio odd increment decorrelating successive layers' weight draws.
const SEED_STEP: u64 = 0x9E37_79B9_7F4A_7C15;

/// Whether the flow field is integrated (diffeomorphic) or used directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransformIntegration {
    /// Use the prediction directly as displacement.
    Direct,
    /// Flow field is integrated via scaling-and-squaring into a diffeomorphic warp.
    #[default]
    Integrated,
}

/// Configuration for a [`TransMorph`] network.
#[derive(Debug, Clone)]
pub struct TransMorphConfig {
    /// Input channels (e.g. 1 for a single moving volume).
    pub in_channels: usize,
    /// Base embedding dimension (channel width after patch embedding).
    pub embed_dim: usize,
    /// Output flow channels (3 for a 3-D displacement field).
    pub out_channels: usize,
    /// Cubic attention-window side length.
    pub window_size: usize,
    /// Integration mode for the predicted flow field.
    pub integration: TransformIntegration,
    /// Number of scaling-and-squaring steps when integrating.
    pub integration_steps: usize,
}

impl TransMorphConfig {
    /// Construct a config with default window (4) and integration (7 steps).
    #[must_use]
    pub fn new(in_channels: usize, embed_dim: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dim,
            out_channels,
            window_size: 4,
            integration: TransformIntegration::Integrated,
            integration_steps: 7,
        }
    }

    /// Override the attention-window side length.
    #[must_use]
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Override the integration mode.
    #[must_use]
    pub fn with_integration(mut self, integration: TransformIntegration) -> Self {
        self.integration = integration;
        self
    }

    /// Instantiate a [`TransMorph`] over backend `B`.
    pub fn init<B>(&self) -> TransMorph<B>
    where
        B: Backend + BackendOps<f32> + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let seed = std::cell::Cell::new(INIT_SEED);
        let next_seed = || {
            let s = seed.get().wrapping_add(SEED_STEP);
            seed.set(s);
            s
        };
        let conv = |in_ch: usize, out_ch: usize, kernel: usize, stride: usize, pad: usize| {
            let mut layer =
                Conv3d::<f32, B>::with_params(in_ch, out_ch, kernel, stride, pad, DILATION, true);
            let fan_in = in_ch * kernel * kernel * kernel;
            coeus_nn::init::kaiming_uniform_with_seed(&mut layer.weight, fan_in, next_seed());
            layer
        };

        let e = self.embed_dim;
        let ws = self.window_size;
        let shift = ws / 2;
        let block = |dim: usize, shift_size: usize| {
            SwinTransformerBlock::<B>::new(dim, NUM_HEADS, ws, shift_size, MLP_RATIO, next_seed())
        };

        // Encoder: patch embed (stride-4) + four stages, each downsampling by 2.
        let patch_embed = conv(self.in_channels, e, 4, 4, 0);
        let stage1 = vec![block(e, 0), block(e, 0)];
        let down1 = conv(e, e * 2, 2, 2, 0);
        let stage2 = vec![block(e * 2, shift), block(e * 2, 0)];
        let down2 = conv(e * 2, e * 4, 2, 2, 0);
        let stage3 = vec![block(e * 4, shift), block(e * 4, 0)];
        let down3 = conv(e * 4, e * 8, 2, 2, 0);
        let stage4 = vec![block(e * 8, shift), block(e * 8, 0)];

        // Decoder: skip-concatenated 3×3 convolutions.
        let up_conv1 = conv(e * 8 + e * 4, e * 4, 3, 1, 1);
        let up_conv2 = conv(e * 4 + e * 2, e * 2, 3, 1, 1);
        let up_conv3 = conv(e * 2 + e, e, 3, 1, 1);
        let flow_conv = conv(e, self.out_channels, 3, 1, 1);

        let integration = match self.integration {
            TransformIntegration::Integrated => Some(VecInt::new(self.integration_steps)),
            TransformIntegration::Direct => None,
        };
        TransMorph {
            patch_embed,
            stage1,
            down1,
            stage2,
            down2,
            stage3,
            down3,
            stage4,
            up_conv1,
            up_conv2,
            up_conv3,
            flow_conv,
            integration,
            spatial_transform: SpatialTransformer::new(),
        }
    }
}
