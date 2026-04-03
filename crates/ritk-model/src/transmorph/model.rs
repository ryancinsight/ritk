use burn::{
    module::Module,
    nn::conv::Conv3d,
    tensor::{backend::Backend, Tensor},
};

use crate::transmorph::{
    integration::VecInt,
    spatial_transform::SpatialTransformer,
    swin::SwinTransformerBlock,
};

/// Output from TransMorph forward pass
#[derive(Debug, Clone)]
pub struct TransMorphOutput<B: Backend> {
    /// Warped input image(s) [batch, C, D, H, W]
    pub warped: Tensor<B, 5>,
    /// Final displacement field [batch, 3, D, H, W]
    pub flow: Tensor<B, 5>,
}

#[derive(Module, Debug)]
pub struct TransMorph<B: Backend> {
    pub(crate) patch_embed: Conv3d<B>,
    pub(crate) stage1: Vec<SwinTransformerBlock<B>>,
    pub(crate) down1: Conv3d<B>,
    pub(crate) stage2: Vec<SwinTransformerBlock<B>>,
    pub(crate) down2: Conv3d<B>,
    pub(crate) stage3: Vec<SwinTransformerBlock<B>>,
    pub(crate) down3: Conv3d<B>,
    pub(crate) stage4: Vec<SwinTransformerBlock<B>>,

    // Decoder
    pub(crate) up_conv1: Conv3d<B>,
    pub(crate) up_conv2: Conv3d<B>,
    pub(crate) up_conv3: Conv3d<B>,
    pub(crate) flow_conv: Conv3d<B>,

    pub(crate) integration: Option<VecInt<B>>,
    pub(crate) spatial_transform: SpatialTransformer<B>,
}

impl<B: Backend> TransMorph<B> {
    pub fn forward(&self, x: Tensor<B, 5>) -> TransMorphOutput<B> {
        // Encoder
        // x: [B, C, D, H, W]
        let x0 = self.patch_embed.forward(x.clone()); // [B, 12, D/4, H/4, W/4]

        // Stage 1
        // Permute to [B, D, H, W, C] for Swin Blocks
        let mut x1 = x0.permute([0, 2, 3, 4, 1]);
        for block in &self.stage1 {
            x1 = block.forward(x1);
        }
        // Permute back to [B, C, D, H, W] for downsampling and skip connection
        let x1_out = x1.permute([0, 4, 1, 2, 3]);

        let x1_down = self.down1.forward(x1_out.clone());

        // Stage 2
        let mut x2 = x1_down.permute([0, 2, 3, 4, 1]);
        for block in &self.stage2 {
            x2 = block.forward(x2);
        }
        let x2_out = x2.permute([0, 4, 1, 2, 3]);

        let x2_down = self.down2.forward(x2_out.clone());

        // Stage 3
        let mut x3 = x2_down.permute([0, 2, 3, 4, 1]);
        for block in &self.stage3 {
            x3 = block.forward(x3);
        }
        let x3_out = x3.permute([0, 4, 1, 2, 3]);

        let x3_down = self.down3.forward(x3_out.clone());

        // Stage 4
        let mut x4 = x3_down.permute([0, 2, 3, 4, 1]);
        for block in &self.stage4 {
            x4 = block.forward(x4);
        }
        let x4_out = x4.permute([0, 4, 1, 2, 3]);

        // Decoder (Simple U-Net style with nearest neighbor upsampling)
        // x4_out: [B, 96, D/32, H/32, W/32]

        // Up 1
        let x4_up = self.upsample(x4_out, 2.0); // -> [D/16...]
        let cat1 = Tensor::cat(vec![x4_up, x3_out], 1);
        let d1 = self.up_conv1.forward(cat1);

        // Up 2
        let d1_up = self.upsample(d1, 2.0); // -> [D/8...]
        let cat2 = Tensor::cat(vec![d1_up, x2_out], 1);
        let d2 = self.up_conv2.forward(cat2);

        // Up 3
        let d2_up = self.upsample(d2, 2.0); // -> [D/4...]
        let cat3 = Tensor::cat(vec![d2_up, x1_out], 1);
        let d3 = self.up_conv3.forward(cat3);

        // Up 4 (Final resolution)
        let d3_up = self.upsample(d3, 4.0); // -> [D, H, W]

        // Flow
        let flow = self.flow_conv.forward(d3_up);

        // Integration
        let final_flow = if let Some(integration) = &self.integration {
            integration.forward(flow)
        } else {
            flow
        };

        // Warp
        let warped = self.spatial_transform.forward(x, final_flow.clone());

        TransMorphOutput {
            warped,
            flow: final_flow,
        }
    }

    fn upsample(&self, x: Tensor<B, 5>, scale: f64) -> Tensor<B, 5> {
        let [b, c, d, h, w] = x.dims();
        let scale_int = scale as usize;

        // Naive Nearest Neighbor via iterative repeat to avoid >6 dims
        // [B, C, D, H, W] -> [B, C, D*S, H, W] -> ...

        // 1. D-dimension: [B, C, D, 1, H * W] (5D)
        let x = x.reshape([b, c, d, 1, h * w]);
        let x = x.repeat(&[1, 1, 1, scale_int, 1]);
        let x = x.reshape([b, c, d * scale_int, h, w]);

        // 2. H-dimension: [B, C * D' * S, H, 1, W]
        let x = x.reshape([b, c * d * scale_int, h, 1, w]);
        let x = x.repeat(&[1, 1, 1, scale_int, 1]);
        let x = x.reshape([b, c, d * scale_int, h * scale_int, w]);

        // 3. W-dimension: [B, C * D' * H', W, 1]
        let x = x.reshape([b, c * d * scale_int * h * scale_int, w, 1]);
        let x = x.repeat(&[1, 1, scale_int]);
        x.reshape([b, c, d * scale_int, h * scale_int, w * scale_int])
    }
}
