//! Fused transform → world-to-index → linear interpolation.

use coeus_core::Backend;
use coeus_tensor::Tensor;

use crate::interpolation::shared::compute_oob_mask;
use crate::interpolation::LinearInterpolator;
use ritk_core::interpolation::Interpolator;
use ritk_core::transform::Transform;
use ritk_image::types::Image;

#[cfg(test)]
pub(crate) fn is_identity_direction<const D: usize>(direction: &ritk_spatial::Direction<D>) -> bool {
    let id: ritk_spatial::Direction<D> = ritk_spatial::Direction::identity();
    for r in 0..D {
        for c in 0..D {
            if (direction[(r, c)] - id[(r, c)]).abs() > 1e-9 {
                return false;
            }
        }
    }
    true
}

pub struct FusedInterpolationResult<B: Backend> {
    pub values: Tensor<f32, B>,
    pub oob_mask: Option<Tensor<f32, B>>,
}

pub fn transform_and_interpolate<B, T, const D: usize>(
    fixed_points: Tensor<f32, B>,
    transform: &T,
    moving: &Image<f32, B, D>,
    interpolator: &LinearInterpolator,
) -> FusedInterpolationResult<B>
where
    B: Backend + Default + coeus_ops::BackendOps<f32>,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    T: Transform<B, D>,
{
    let moving_world = transform.transform_points(fixed_points);
    let indices = moving.world_to_index_tensor(moving_world);
    let oob_mask = compute_oob_mask(&indices, &moving.shape());
    let values = interpolator.interpolate(moving.data(), indices);
    FusedInterpolationResult {
        values,
        oob_mask: Some(oob_mask),
    }
}

#[cfg(test)]
#[path = "tests/fused.rs"]
mod tests;
