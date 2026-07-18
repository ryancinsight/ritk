//! TransMorph model, Coeus-native.
//!
//! Encoderâ€“decoder registration network: a Swin-transformer encoder (patch
//! embedding + four hierarchical stages with strided-conv downsampling), a
//! U-Net-style decoder with nearest-neighbor upsampling and skip connections,
//! optional diffeomorphic velocity integration, and a differentiable spatial
//! transformer. Built entirely on [`coeus_nn`]/[`coeus_autograd`] over
//! [`coeus_autograd::Var`]; no Burn tensors, modules, or backends cross this
//! boundary.
//!
//! Reference: Chen et al., "TransMorph: Transformer for unsupervised medical
//! image registration", Medical Image Analysis 82 (2022) 102615.

use coeus_autograd::{cat, permute, reshape, tile, Parameter, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{module::Module, Conv3d};
use coeus_ops::BackendOps;

use crate::transmorph::{
    integration::VecInt, spatial_transform::SpatialTransformer, swin::SwinTransformerBlock };

/// Output of the TransMorph forward pass.
#[derive(Clone)]
pub struct TransMorphOutput<B: Backend + BackendOps<f32> + Default> {
    /// Warped moving image `[B, C, D, H, W]`.
    pub warped: Var<f32, B>,
    /// Final displacement field `[B, 3, D, H, W]`.
    pub flow: Var<f32, B> }

/// TransMorph registration network.
#[derive(Clone)]
pub struct TransMorph<B: Backend + BackendOps<f32> + Default> {
    pub(crate) patch_embed: Conv3d<f32, B>,
    pub(crate) stage1: Vec<SwinTransformerBlock<B>>,
    pub(crate) down1: Conv3d<f32, B>,
    pub(crate) stage2: Vec<SwinTransformerBlock<B>>,
    pub(crate) down2: Conv3d<f32, B>,
    pub(crate) stage3: Vec<SwinTransformerBlock<B>>,
    pub(crate) down3: Conv3d<f32, B>,
    pub(crate) stage4: Vec<SwinTransformerBlock<B>>,
    pub(crate) up_conv1: Conv3d<f32, B>,
    pub(crate) up_conv2: Conv3d<f32, B>,
    pub(crate) up_conv3: Conv3d<f32, B>,
    pub(crate) flow_conv: Conv3d<f32, B>,
    pub(crate) integration: Option<VecInt>,
    pub(crate) spatial_transform: SpatialTransformer }

impl<B> TransMorph<B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Forward pass: register `image` `[B, C, D, H, W]`, producing the warped
    /// image and the displacement field.
    pub fn forward(&self, image: &Var<f32, B>) -> TransMorphOutput<B> {
        // Patch embedding, then four Swin stages in channels-last layout.
        let x0 = self.patch_embed.forward(image);

        let x1_out = self.run_stage(&x0, &self.stage1);
        let x1_down = self.down1.forward(&x1_out);

        let x2_out = self.run_stage(&x1_down, &self.stage2);
        let x2_down = self.down2.forward(&x2_out);

        let x3_out = self.run_stage(&x2_down, &self.stage3);
        let x3_down = self.down3.forward(&x3_out);

        let x4_out = self.run_stage(&x3_down, &self.stage4);

        // U-Net decoder: upsample, concatenate skip, convolve.
        let x4_up = self.upsample(&x4_out, 2);
        let d1 = self.up_conv1.forward(&cat(&[&x4_up, &x3_out], 1));

        let d1_up = self.upsample(&d1, 2);
        let d2 = self.up_conv2.forward(&cat(&[&d1_up, &x2_out], 1));

        let d2_up = self.upsample(&d2, 2);
        let d3 = self.up_conv3.forward(&cat(&[&d2_up, &x1_out], 1));

        let d3_up = self.upsample(&d3, 4);
        let flow = self.flow_conv.forward(&d3_up);

        let flow = match &self.integration {
            Some(integration) => integration.forward(&flow),
            None => flow };

        let warped = self.spatial_transform.forward(image, &flow);
        TransMorphOutput { warped, flow }
    }

    /// Run one Swin stage: `[B, C, D, H, W]` â†’ channels-last blocks â†’ back.
    fn run_stage(&self, x: &Var<f32, B>, blocks: &[SwinTransformerBlock<B>]) -> Var<f32, B> {
        let mut y = permute(x, &[0, 2, 3, 4, 1]);
        for block in blocks {
            y = block.forward(&y);
        }
        permute(&y, &[0, 4, 1, 2, 3])
    }

    /// Nearest-neighbor upsample `[B, C, D, H, W]` by an integer `scale` on each
    /// spatial axis. Each axis inserts a unit dimension, tiles it, and merges,
    /// keeping every intermediate tensor at most rank-5.
    fn upsample(&self, x: &Var<f32, B>, scale: usize) -> Var<f32, B> {
        let sh = x.tensor.shape();
        let (b, c, d, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);

        // D axis.
        let x = reshape(x, [b, c, d, 1, h * w]);
        let x = tile(&x, &[1, 1, 1, scale, 1]);
        let x = reshape(&x, [b, c, d * scale, h, w]);

        // H axis.
        let x = reshape(&x, [b, c * d * scale, h, 1, w]);
        let x = tile(&x, &[1, 1, 1, scale, 1]);
        let x = reshape(&x, [b, c, d * scale, h * scale, w]);

        // W axis.
        let x = reshape(&x, [b, c * d * scale * h * scale, w, 1]);
        let x = tile(&x, &[1, 1, 1, scale]);
        reshape(&x, [b, c, d * scale, h * scale, w * scale])
    }

    /// Trainable parameters in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.patch_embed.parameters();
        let stages = [&self.stage1, &self.stage2, &self.stage3, &self.stage4];
        let downs = [&self.down1, &self.down2, &self.down3];
        for (i, stage) in stages.iter().enumerate() {
            for block in stage.iter() {
                params.extend(block.parameters());
            }
            if let Some(down) = downs.get(i) {
                params.extend(down.parameters());
            }
        }
        params.extend(self.up_conv1.parameters());
        params.extend(self.up_conv2.parameters());
        params.extend(self.up_conv3.parameters());
        params.extend(self.flow_conv.parameters());
        params
    }

    /// Trainable parameters with stable hierarchical names.
    pub fn named_parameters(&self) -> Vec<Parameter<f32, B>> {
        let mut named: Vec<Parameter<f32, B>> = self
            .patch_embed
            .named_parameters()
            .into_iter()
            .map(|p| p.with_prefix("patch_embed"))
            .collect();

        let stages = [
            ("stage1", &self.stage1),
            ("stage2", &self.stage2),
            ("stage3", &self.stage3),
            ("stage4", &self.stage4),
        ];
        for (stage_name, blocks) in stages {
            for (i, block) in blocks.iter().enumerate() {
                let prefix = format!("{stage_name}.{i}");
                named.extend(
                    block
                        .named_parameters()
                        .into_iter()
                        .map(|p| p.with_prefix(&prefix)),
                );
            }
        }
        for (name, conv) in [
            ("down1", &self.down1),
            ("down2", &self.down2),
            ("down3", &self.down3),
            ("up_conv1", &self.up_conv1),
            ("up_conv2", &self.up_conv2),
            ("up_conv3", &self.up_conv3),
            ("flow_conv", &self.flow_conv),
        ] {
            named.extend(
                conv.named_parameters()
                    .into_iter()
                    .map(|p| p.with_prefix(name)),
            );
        }
        named
    }
}
