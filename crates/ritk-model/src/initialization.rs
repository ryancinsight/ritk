//! Registration-model parameter initialization policies.

use coeus_core::Backend;
use coeus_nn::{init, Conv3d, DepthwiseConv3d, Linear};
use coeus_ops::BackendOps;

pub(crate) fn linear<B>(layer: &mut Linear<f32, B>, fan_in: usize, fan_out: usize, seed: u64)
where
    B: Backend + BackendOps<f32>,
{
    init::xavier_uniform_with_seed(&mut layer.weight, fan_in, fan_out, seed);
    if let Some(bias) = &mut layer.bias {
        init::zeros(bias);
    }
}

pub(crate) fn depthwise_convolution<B>(
    layer: &mut DepthwiseConv3d<f32, B>,
    kernel: usize,
    seed: u64,
) where
    B: Backend + BackendOps<f32>,
{
    init::kaiming_uniform_with_seed(&mut layer.weight, kernel * kernel * kernel, seed);
    if let Some(bias) = &mut layer.bias {
        init::zeros(bias);
    }
}

pub(crate) fn convolution<B>(
    layer: &mut Conv3d<f32, B>,
    input_channels: usize,
    kernel: usize,
    seed: u64,
) where
    B: Backend + BackendOps<f32>,
{
    let fan_in = input_channels * kernel * kernel * kernel;
    init::kaiming_uniform_with_seed(&mut layer.weight, fan_in, seed);
    if let Some(bias) = &mut layer.bias {
        init::zeros(bias);
    }
}

pub(crate) fn zero_convolution<B>(layer: &mut Conv3d<f32, B>)
where
    B: Backend + BackendOps<f32>,
{
    init::zeros(&mut layer.weight);
    if let Some(bias) = &mut layer.bias {
        init::zeros(bias);
    }
}
