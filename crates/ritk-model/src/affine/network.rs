//! Coeus-backed affine parameter regressor.

use coeus_autograd::{add, mean_axis, reshape, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{Conv3d, InstanceNorm3d, Linear, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Convolutional affine-regression network.
#[derive(Clone)]
pub struct AffineNetwork<B>
where
    B: Backend + BackendOps<f32>,
{
    convolutions: [Conv3d<f32, B>; 5],
    normalizations: [InstanceNorm3d<f32, B>; 5],
    projection: Linear<f32, B>,
}

/// Affine-network channel configuration.
#[derive(Debug, Clone)]
pub struct AffineNetworkConfig {
    pub channels: [usize; 5],
}

impl Default for AffineNetworkConfig {
    fn default() -> Self {
        Self {
            channels: [16, 32, 64, 128, 256],
        }
    }
}

impl AffineNetworkConfig {
    /// Initialize the graph on backend `B`.
    #[must_use]
    pub fn init<B>(&self) -> AffineNetwork<B>
    where
        B: Backend + BackendOps<f32>,
    {
        let channels = self.channels;
        let mut network = AffineNetwork {
            convolutions: [
                convolution(2, channels[0]),
                convolution(channels[0], channels[1]),
                convolution(channels[1], channels[2]),
                convolution(channels[2], channels[3]),
                convolution(channels[3], channels[4]),
            ],
            normalizations: channels.map(|width| InstanceNorm3d::new(width, 1e-5)),
            projection: Linear::new(channels[4], 12, true),
        };
        for (index, convolution) in network.convolutions.iter_mut().enumerate() {
            let input_channels = if index == 0 { 2 } else { channels[index - 1] };
            crate::initialization::convolution(convolution, input_channels, 3, 100 + index as u64);
        }
        coeus_nn::init::zeros(&mut network.projection.weight);
        if let Some(bias) = &mut network.projection.bias {
            coeus_nn::init::zeros(bias);
        }
        network
    }
}

impl<B> Module<f32, B> for AffineNetwork<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = Vec::new();
        for (convolution, normalization) in self.convolutions.iter().zip(self.normalizations.iter())
        {
            parameters.extend(convolution.parameters());
            parameters.extend(normalization.parameters());
        }
        parameters.extend(self.projection.parameters());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        let mut activation = input.clone();
        for (convolution, normalization) in self.convolutions.iter().zip(self.normalizations.iter())
        {
            activation = convolution.forward(&activation);
            activation = normalization.forward(&activation);
            activation = coeus_autograd::relu(&activation);
        }
        let shape = activation.tensor.shape();
        let (batch, channels) = (shape[0], shape[1]);
        let flattened = reshape(
            &activation,
            [batch, channels, shape[2] * shape[3] * shape[4]],
        );
        let pooled = mean_axis(&flattened, 2);
        let pooled = reshape(&pooled, [batch, channels]);
        let parameters = self.projection.forward(&pooled);
        let identity = Var::new(
            Tensor::from_slice_on(
                [1, 12],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                &B::default(),
            ),
            false,
        );
        add(&parameters, &identity)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let mut offset = 0;
        for (convolution, normalization) in self
            .convolutions
            .iter_mut()
            .zip(self.normalizations.iter_mut())
        {
            load_module(convolution, parameters, &mut offset);
            load_module(normalization, parameters, &mut offset);
        }
        load_module(&mut self.projection, parameters, &mut offset);
        assert_eq!(
            offset,
            parameters.len(),
            "parameter inventory must be exact"
        );
    }
}

fn convolution<B>(input_channels: usize, output_channels: usize) -> Conv3d<f32, B>
where
    B: Backend + BackendOps<f32>,
{
    Conv3d::with_params(input_channels, output_channels, 3, 2, 1, 1, true)
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
