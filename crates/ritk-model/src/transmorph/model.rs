//! Coeus-backed TransMorph encoder-decoder graph.

use coeus_autograd::{cat, permute, reshape, tile, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{Conv3d, Module};
use coeus_ops::BackendOps;

use crate::{
    transmorph::{
        integration::VecInt, spatial_transform::SpatialTransformer, swin::SwinTransformerBlock,
    },
    ModelError,
};

/// Output of a TransMorph forward pass.
#[derive(Clone)]
pub struct TransMorphOutput<B>
where
    B: Backend + BackendOps<f32>,
{
    /// Warped input volume.
    pub warped: Var<f32, B>,
    /// Final displacement field.
    pub flow: Var<f32, B>,
}

/// Hierarchical shifted-window registration network.
#[derive(Clone)]
pub struct TransMorph<B>
where
    B: Backend + BackendOps<f32>,
{
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
    pub(crate) integration: Option<VecInt<B>>,
    pub(crate) spatial_transform: SpatialTransformer<B>,
}

impl<B> TransMorph<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Evaluate the registration graph.
    pub fn forward(&self, input: &Var<f32, B>) -> Result<TransMorphOutput<B>, ModelError> {
        let x0 = self.patch_embed.forward(input);
        let mut x1 = permute(&x0, &[0, 2, 3, 4, 1]);
        for block in &self.stage1 {
            x1 = block.forward(&x1);
        }
        let x1_out = permute(&x1, &[0, 4, 1, 2, 3]);

        let mut x2 = permute(&self.down1.forward(&x1_out), &[0, 2, 3, 4, 1]);
        for block in &self.stage2 {
            x2 = block.forward(&x2);
        }
        let x2_out = permute(&x2, &[0, 4, 1, 2, 3]);

        let mut x3 = permute(&self.down2.forward(&x2_out), &[0, 2, 3, 4, 1]);
        for block in &self.stage3 {
            x3 = block.forward(&x3);
        }
        let x3_out = permute(&x3, &[0, 4, 1, 2, 3]);

        let mut x4 = permute(&self.down3.forward(&x3_out), &[0, 2, 3, 4, 1]);
        for block in &self.stage4 {
            x4 = block.forward(&x4);
        }
        let x4_out = permute(&x4, &[0, 4, 1, 2, 3]);

        let up1 = upsample_nearest(&x4_out, 2);
        let decoded1 = self.up_conv1.forward(&cat(&[&up1, &x3_out], 1));
        let up2 = upsample_nearest(&decoded1, 2);
        let decoded2 = self.up_conv2.forward(&cat(&[&up2, &x2_out], 1));
        let up3 = upsample_nearest(&decoded2, 2);
        let decoded3 = self.up_conv3.forward(&cat(&[&up3, &x1_out], 1));
        let full_resolution = upsample_nearest(&decoded3, 4);
        let flow = self.flow_conv.forward(&full_resolution);
        let flow = if let Some(integration) = &self.integration {
            integration.forward(&flow)?
        } else {
            flow
        };
        let warped = self.spatial_transform.forward(input, &flow)?;
        Ok(TransMorphOutput { warped, flow })
    }
}

impl<B> Module<f32, B> for TransMorph<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = self.patch_embed.parameters();
        for stage in [&self.stage1, &self.stage2, &self.stage3, &self.stage4] {
            for block in stage {
                parameters.extend(block.parameters());
            }
        }
        for convolution in [
            &self.down1,
            &self.down2,
            &self.down3,
            &self.up_conv1,
            &self.up_conv2,
            &self.up_conv3,
            &self.flow_conv,
        ] {
            parameters.extend(convolution.parameters());
        }
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        TransMorph::forward(self, input)
            .expect("invariant: Module input satisfies the documented TransMorph contract")
            .flow
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let mut offset = 0;
        load_module(&mut self.patch_embed, parameters, &mut offset);
        for stage in [
            &mut self.stage1,
            &mut self.stage2,
            &mut self.stage3,
            &mut self.stage4,
        ] {
            for block in stage {
                load_module(block, parameters, &mut offset);
            }
        }
        for convolution in [
            &mut self.down1,
            &mut self.down2,
            &mut self.down3,
            &mut self.up_conv1,
            &mut self.up_conv2,
            &mut self.up_conv3,
            &mut self.flow_conv,
        ] {
            load_module(convolution, parameters, &mut offset);
        }
        assert_eq!(
            offset,
            parameters.len(),
            "parameter inventory must be exact"
        );
    }

    fn train(&mut self, mode: bool) {
        for stage in [
            &mut self.stage1,
            &mut self.stage2,
            &mut self.stage3,
            &mut self.stage4,
        ] {
            for block in stage {
                block.train(mode);
            }
        }
    }
}

fn load_module<M, B>(module: &mut M, parameters: &[Var<f32, B>], offset: &mut usize)
where
    B: Backend + BackendOps<f32>,
    M: Module<f32, B>,
{
    let count = module.parameters().len();
    module.load_parameters(&parameters[*offset..*offset + count]);
    *offset += count;
}

fn upsample_nearest<B>(input: &Var<f32, B>, scale: usize) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let shape = input.tensor.shape();
    let (batch, channels, depth, height, width) =
        (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let expanded = reshape(input, [batch, channels, depth, 1, height, width]);
    let expanded = tile(&expanded, &[1, 1, 1, scale, 1, 1]);
    let expanded = reshape(&expanded, [batch, channels, depth * scale, height, width]);
    let expanded = reshape(
        &expanded,
        [batch, channels, depth * scale, height, 1, width],
    );
    let expanded = tile(&expanded, &[1, 1, 1, 1, scale, 1]);
    let expanded = reshape(
        &expanded,
        [batch, channels, depth * scale, height * scale, width],
    );
    let expanded = reshape(
        &expanded,
        [batch, channels, depth * scale, height * scale, width, 1],
    );
    let expanded = tile(&expanded, &[1, 1, 1, 1, 1, scale]);
    reshape(
        &expanded,
        [
            batch,
            channels,
            depth * scale,
            height * scale,
            width * scale,
        ],
    )
}
