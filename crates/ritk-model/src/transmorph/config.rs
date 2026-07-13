//! TransMorph graph configuration.

use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Conv3d;
use coeus_ops::BackendOps;

use crate::transmorph::{
    integration::VecInt, model::TransMorph, spatial_transform::SpatialTransformer,
    swin::SwinTransformerBlockConfig,
};

/// Whether to integrate the predicted velocity field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransformIntegration {
    /// Use the prediction directly as displacement.
    Direct,
    /// Integrate by scaling and squaring.
    #[default]
    Integrated,
}

/// TransMorph architecture configuration.
#[derive(Debug, Clone)]
pub struct TransMorphConfig {
    pub in_channels: usize,
    pub embed_dim: usize,
    pub out_channels: usize,
    pub window_size: usize,
    pub integration: TransformIntegration,
    pub integration_steps: usize,
}

impl TransMorphConfig {
    /// Construct a configuration with the standard window and integration.
    #[must_use]
    pub const fn new(in_channels: usize, embed_dim: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dim,
            out_channels,
            window_size: 4,
            integration: TransformIntegration::Integrated,
            integration_steps: 7,
        }
    }

    /// Set the cubic window width.
    #[must_use]
    pub const fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set flow integration policy.
    #[must_use]
    pub const fn with_integration(mut self, integration: TransformIntegration) -> Self {
        self.integration = integration;
        self
    }

    /// Initialize the graph on backend `B`.
    #[must_use]
    pub fn init<B>(&self) -> TransMorph<B>
    where
        B: Backend + BackendOps<f32>,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let block = |width, shift| {
            SwinTransformerBlockConfig::new(width, 4, self.window_size, shift, 4.0).init()
        };
        let mut model = TransMorph {
            patch_embed: convolution(self.in_channels, self.embed_dim, 4, 4, 0),
            stage1: vec![block(self.embed_dim, 0), block(self.embed_dim, 0)],
            down1: convolution(self.embed_dim, self.embed_dim * 2, 2, 2, 0),
            stage2: vec![
                block(self.embed_dim * 2, self.window_size / 2),
                block(self.embed_dim * 2, 0),
            ],
            down2: convolution(self.embed_dim * 2, self.embed_dim * 4, 2, 2, 0),
            stage3: vec![
                block(self.embed_dim * 4, self.window_size / 2),
                block(self.embed_dim * 4, 0),
            ],
            down3: convolution(self.embed_dim * 4, self.embed_dim * 8, 2, 2, 0),
            stage4: vec![
                block(self.embed_dim * 8, self.window_size / 2),
                block(self.embed_dim * 8, 0),
            ],
            up_conv1: convolution(self.embed_dim * 12, self.embed_dim * 4, 3, 1, 1),
            up_conv2: convolution(self.embed_dim * 6, self.embed_dim * 2, 3, 1, 1),
            up_conv3: convolution(self.embed_dim * 3, self.embed_dim, 3, 1, 1),
            flow_conv: convolution(self.embed_dim, self.out_channels, 3, 1, 1),
            integration: (self.integration == TransformIntegration::Integrated)
                .then(|| VecInt::new(self.integration_steps)),
            spatial_transform: SpatialTransformer::new(),
        };
        for (index, (convolution, input_channels, kernel)) in [
            (&mut model.patch_embed, self.in_channels, 4),
            (&mut model.down1, self.embed_dim, 2),
            (&mut model.down2, self.embed_dim * 2, 2),
            (&mut model.down3, self.embed_dim * 4, 2),
            (&mut model.up_conv1, self.embed_dim * 12, 3),
            (&mut model.up_conv2, self.embed_dim * 6, 3),
            (&mut model.up_conv3, self.embed_dim * 3, 3),
        ]
        .into_iter()
        .enumerate()
        {
            crate::initialization::convolution(
                convolution,
                input_channels,
                kernel,
                300 + index as u64,
            );
        }
        crate::initialization::zero_convolution(&mut model.flow_conv);
        model
    }
}

fn convolution<B>(
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Conv3d<f32, B>
where
    B: Backend + BackendOps<f32>,
{
    Conv3d::with_params(
        input_channels,
        output_channels,
        kernel,
        stride,
        padding,
        1,
        true,
    )
}
