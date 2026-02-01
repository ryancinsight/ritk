pub mod swin;
pub mod spatial_transform;

use burn::{
    nn::{
        conv::{Conv3d, Conv3dConfig, ConvTranspose3d, ConvTranspose3dConfig},
        Gelu, LayerNorm, LayerNormConfig, PaddingConfig3d,
    },
    prelude::*,
};
use swin::{SwinTransformerBlock, SwinTransformerBlockConfig};
pub use spatial_transform::SpatialTransformer;

#[derive(Module, Debug)]
pub struct TransMorph<B: Backend> {
    // Encoder
    patch_embed: Conv3d<B>,
    stage1: Vec<SwinTransformerBlock<B>>,
    down1: Conv3d<B>, // Patch merging 1
    stage2: Vec<SwinTransformerBlock<B>>,
    down2: Conv3d<B>, // Patch merging 2
    stage3: Vec<SwinTransformerBlock<B>>,
    down3: Conv3d<B>, // Patch merging 3
    stage4: Vec<SwinTransformerBlock<B>>,

    // Decoder (Simplified with Conv blocks for now, matching TransMorph-affine or similar variants)
    up3: ConvTranspose3d<B>,
    conv3: ConvBlock<B>,
    up2: ConvTranspose3d<B>,
    conv2: ConvBlock<B>,
    up1: ConvTranspose3d<B>,
    conv1: ConvBlock<B>,
    
    // Head
    flow_head: Conv3d<B>,
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: Conv3d<B>,
    norm1: LayerNorm<B>, // InstanceNorm is preferred in MedIA, but LayerNorm or GroupNorm works too
    act: Gelu,
    conv2: Conv3d<B>,
    norm2: LayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct TransMorphConfig {
    #[config(default = 1)]
    pub in_channels: usize,
    #[config(default = 96)]
    pub embed_dim: usize,
    #[config(default = 3)]
    pub out_channels: usize, // 3 for 3D displacement field
    #[config(default = 4)]
    pub window_size: usize,
}

impl TransMorphConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransMorph<B> {
        let dim = self.embed_dim;
        let ws = self.window_size;

        // Encoder
        // Patch Embed: 2x2x2 patch -> dim channels
        let patch_embed = Conv3dConfig::new([self.in_channels, dim], [4, 4, 4])
            .with_stride([4, 4, 4])
            .init(device);

        // Stage 1
        let stage1 = vec![
            SwinTransformerBlockConfig::new(dim, 4, ws, 0, 4.0).init(device),
            SwinTransformerBlockConfig::new(dim, 4, ws, ws/2, 4.0).init(device),
        ];
        
        // Downsample 1: dim -> 2*dim
        let down1 = Conv3dConfig::new([dim, 2 * dim], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);

        // Stage 2
        let stage2 = vec![
            SwinTransformerBlockConfig::new(2 * dim, 4, ws, 0, 4.0).init(device),
            SwinTransformerBlockConfig::new(2 * dim, 4, ws, ws/2, 4.0).init(device),
        ];

        // Downsample 2: 2*dim -> 4*dim
        let down2 = Conv3dConfig::new([2 * dim, 4 * dim], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);

        // Stage 3
        let stage3 = vec![
            SwinTransformerBlockConfig::new(4 * dim, 8, ws, 0, 4.0).init(device),
            SwinTransformerBlockConfig::new(4 * dim, 8, ws, ws/2, 4.0).init(device),
            SwinTransformerBlockConfig::new(4 * dim, 8, ws, 0, 4.0).init(device),
            SwinTransformerBlockConfig::new(4 * dim, 8, ws, ws/2, 4.0).init(device),
        ];

        // Downsample 3: 4*dim -> 8*dim
        let down3 = Conv3dConfig::new([4 * dim, 8 * dim], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);

        // Stage 4 (Bottleneck)
        let stage4 = vec![
            SwinTransformerBlockConfig::new(8 * dim, 8, ws, 0, 4.0).init(device),
            SwinTransformerBlockConfig::new(8 * dim, 8, ws, ws/2, 4.0).init(device),
        ];

        // Decoder
        // Up 3: 8*dim -> 4*dim
        let up3 = ConvTranspose3dConfig::new([8 * dim, 4 * dim], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);
        // Concatenation happens in forward, input to conv is 4*dim + 4*dim = 8*dim
        let conv3 = ConvBlockConfig::new(8 * dim, 4 * dim).init(device);

        // Up 2: 4*dim -> 2*dim
        let up2 = ConvTranspose3dConfig::new([4 * dim, 2 * dim], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);
        // Concat: 2*dim + 2*dim = 4*dim
        let conv2 = ConvBlockConfig::new(4 * dim, 2 * dim).init(device);

        // Up 1: 2*dim -> dim
        let up1 = ConvTranspose3dConfig::new([2 * dim, dim], [2, 2, 2])
            .with_stride([2, 2, 2])
            .init(device);
        // Concat: dim + dim = 2*dim
        let conv1 = ConvBlockConfig::new(2 * dim, dim).init(device);

        // Final upsampling to original resolution not strictly needed if we want low-res flow field,
        // but typically we upsample to full res.
        // Assuming we want full res flow field:
        // Current res is 1/4 of input (due to initial patch embed 4x4x4).
        // Let's add a final upsampling layer or use interpolation. 
        // For TransMorph, it usually outputs 1/2 or 1/1 resolution.
        // Let's assume we output at 1/4 res and upsample linearly, or add another up layer.
        // For simplicity here, we output at 1/4 resolution (standard for VoxelMorph/TransMorph efficiency).
        // Flow head: dim -> 3
        let flow_head = Conv3dConfig::new([dim, self.out_channels], [3, 3, 3])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);

        TransMorph {
            patch_embed,
            stage1,
            down1,
            stage2,
            down2,
            stage3,
            down3,
            stage4,
            up3,
            conv3,
            up2,
            conv2,
            up1,
            conv1,
            flow_head,
        }
    }
}

impl<B: Backend> TransMorph<B> {
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        // x: [B, C, D, H, W]
        
        // Encoder
        // 1. Patch Embed
        let x0 = self.patch_embed.forward(x); // [B, dim, D/4, H/4, W/4]
        
        // Swin Stage 1
        let mut x1 = x0.clone().permute([0, 2, 3, 4, 1]); // [B, D, H, W, C]
        for block in &self.stage1 {
            x1 = block.forward(x1);
        }
        let x1_out = x1.clone().permute([0, 4, 1, 2, 3]); // Back to [B, C, D, H, W]

        // Down 1
        let x2_in = self.down1.forward(x1_out.clone());
        let mut x2 = x2_in.permute([0, 2, 3, 4, 1]);
        for block in &self.stage2 {
            x2 = block.forward(x2);
        }
        let x2_out = x2.clone().permute([0, 4, 1, 2, 3]);

        // Down 2
        let x3_in = self.down2.forward(x2_out.clone());
        let mut x3 = x3_in.permute([0, 2, 3, 4, 1]);
        for block in &self.stage3 {
            x3 = block.forward(x3);
        }
        let x3_out = x3.clone().permute([0, 4, 1, 2, 3]);

        // Down 3
        let x4_in = self.down3.forward(x3_out.clone());
        let mut x4 = x4_in.permute([0, 2, 3, 4, 1]);
        for block in &self.stage4 {
            x4 = block.forward(x4);
        }
        let x4_out = x4.permute([0, 4, 1, 2, 3]);

        // Decoder
        // Up 3
        let up3 = self.up3.forward(x4_out);
        let cat3 = Tensor::cat(vec![up3, x3_out], 1); // Concat along channels
        let d3 = self.conv3.forward(cat3);

        // Up 2
        let up2 = self.up2.forward(d3);
        let cat2 = Tensor::cat(vec![up2, x2_out], 1);
        let d2 = self.conv2.forward(cat2);

        // Up 1
        let up1 = self.up1.forward(d2);
        let cat1 = Tensor::cat(vec![up1, x1_out], 1);
        let d1 = self.conv1.forward(cat1);

        // Head
        let flow = self.flow_head.forward(d1); // [B, 3, D/4, H/4, W/4]
        
        // Upsample flow to original resolution?
        // Typically handled by `spatial_transform` layer which can take low-res flow.
        // We return the flow field.
        flow
    }
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        ConvBlock {
            conv1: Conv3dConfig::new([self.in_channels, self.out_channels], [3, 3, 3])
                .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                .init(device),
            norm1: LayerNormConfig::new(self.out_channels).init(device),
            act: Gelu::new(),
            conv2: Conv3dConfig::new([self.out_channels, self.out_channels], [3, 3, 3])
                .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                .init(device),
            norm2: LayerNormConfig::new(self.out_channels).init(device),
        }
    }
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let x = self.conv1.forward(x);
        
        // LayerNorm in Burn expects [..., C], but Conv output is [B, C, D, H, W]
        // We need to permute for Norm and back.
        let x = x.permute([0, 2, 3, 4, 1]);
        let x = self.norm1.forward(x);
        let x = x.permute([0, 4, 1, 2, 3]);
        
        let x = self.act.forward(x);
        
        let x = self.conv2.forward(x);
        
        let x = x.permute([0, 2, 3, 4, 1]);
        let x = self.norm2.forward(x);
        let x = x.permute([0, 4, 1, 2, 3]);
        
        self.act.forward(x)
    }
}
