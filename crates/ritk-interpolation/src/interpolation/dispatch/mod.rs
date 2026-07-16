//! Runtime dimension dispatch for interpolators.

use coeus_core::Backend;
use coeus_tensor::Tensor;

use super::kernel::{linear, nearest};
use super::shared::OutOfBoundsMode;

#[inline]
pub fn dispatch_linear<B>(
    data: &Tensor<f32, B>,
    indices: Tensor<f32, B>,
    mode: OutOfBoundsMode,
) -> Tensor<f32, B>
where
    B: Backend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    match data.ndim() {
        1 => linear::interpolate_linear_host(data, indices, mode),
        2 => linear::interpolate_linear_host(data, indices, mode),
        3 => linear::interpolate_linear_host(data, indices, mode),
        4 => linear::interpolate_linear_host(data, indices, mode),
        rank => panic!("Linear interpolation only supports ranks 1-4, got {rank}"),
    }
}

#[inline]
pub fn dispatch_nearest<B>(
    data: &Tensor<f32, B>,
    indices: Tensor<f32, B>,
    mode: OutOfBoundsMode,
) -> Tensor<f32, B>
where
    B: Backend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    match data.ndim() {
        1 => nearest::interpolate_nearest_host(data, indices, mode),
        2 => nearest::interpolate_nearest_host(data, indices, mode),
        3 => nearest::interpolate_nearest_host(data, indices, mode),
        4 => nearest::interpolate_nearest_host(data, indices, mode),
        rank => panic!("Nearest-neighbor interpolation only supports ranks 1-4, got {rank}"),
    }
}

pub use dispatch_linear as dispatch_for_shape;
pub use dispatch_nearest as dispatch_nearest_for_shape;
