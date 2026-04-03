use burn::tensor::backend::Backend;
use burn::nn::conv::Conv3dConfig;
use burn::nn::PaddingConfig3d;

use crate::transmorph::{
    integration::VecInt,
    spatial_transform::SpatialTransformer,
    swin::SwinTransformerBlockConfig,
    model::TransMorph,
};

#[derive(Debug, Clone)]
pub struct TransMorphConfig {
    pub in_channels: usize,
    pub embed_dim: usize,
    pub out_channels: usize,
    pub window_size: usize,
    pub integrate: bool,
    pub integration_steps: usize,
}

impl TransMorphConfig {
    pub fn new(in_channels: usize, embed_dim: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dim,
            out_channels,
            window_size: 4,
            integrate: true,
            integration_steps: 7,
        }
    }

    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    pub fn with_integrate(mut self, integrate: bool) -> Self {
        self.integrate = integrate;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransMorph<B> {
        // Stage 1
        let patch_embed = Conv3dConfig::new([self.in_channels, self.embed_dim], [4, 4, 4])
            .with_stride([4, 4, 4])
            .init(device);

        let stage1 = vec![
            SwinTransformerBlockConfig::new(self.embed_dim, 4, self.window_size, 0, 4.0)
                .init(device),
            SwinTransformerBlockConfig::new(self.embed_dim, 4, self.window_size, 0, 4.0)
                .init(device),
        ];

        // Stage 2 (Downsample)
        let down1 = Conv3dConfig::new([self.embed_dim, self.embed_dim * 2], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);

        let stage2 = vec![
            SwinTransformerBlockConfig::new(
                self.embed_dim * 2,
                4,
                self.window_size,
                self.window_size / 2,
                4.0,
            )
            .init(device),
            SwinTransformerBlockConfig::new(self.embed_dim * 2, 4, self.window_size, 0, 4.0)
                .init(device),
        ];

        // Stage 3 (Downsample)
        let down2 = Conv3dConfig::new([self.embed_dim * 2, self.embed_dim * 4], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);

        let stage3 = vec![
            SwinTransformerBlockConfig::new(
                self.embed_dim * 4,
                4,
                self.window_size,
                self.window_size / 2,
                4.0,
            )
            .init(device),
            SwinTransformerBlockConfig::new(self.embed_dim * 4, 4, self.window_size, 0, 4.0)
                .init(device),
        ];

        // Stage 4 (Downsample)
        let down3 = Conv3dConfig::new([self.embed_dim * 4, self.embed_dim * 8], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);

        let stage4 = vec![
            SwinTransformerBlockConfig::new(
                self.embed_dim * 8,
                4,
                self.window_size,
                self.window_size / 2,
                4.0,
            )
            .init(device),
            SwinTransformerBlockConfig::new(self.embed_dim * 8, 4, self.window_size, 0, 4.0)
                .init(device),
        ];

        // Decoder
        // Let's use the full implementation logic for decoder layers
        let up_conv1 = Conv3dConfig::new(
            [self.embed_dim * 8 + self.embed_dim * 4, self.embed_dim * 4],
            [3, 3, 3],
        )
        .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
        .init(device);

        let up_conv2 = Conv3dConfig::new(
            [self.embed_dim * 4 + self.embed_dim * 2, self.embed_dim * 2],
            [3, 3, 3],
        )
        .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
        .init(device);

        let up_conv3 = Conv3dConfig::new(
            [self.embed_dim * 2 + self.embed_dim, self.embed_dim],
            [3, 3, 3],
        )
        .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
        .init(device);

        // Final flow
        let flow_conv = Conv3dConfig::new([self.embed_dim, self.out_channels], [3, 3, 3])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);

        let integration = if self.integrate {
            Some(VecInt::new(self.integration_steps))
        } else {
            None
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
