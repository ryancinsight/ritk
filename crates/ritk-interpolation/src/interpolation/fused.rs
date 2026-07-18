//! Fused transform -> world-to-index -> linear interpolation.
//!
//! Combines three operations that normally produce intermediate `[N, D]` tensors:
//! 1. Transform application (fixed world -> moving world)
//! 2. World-to-index conversion (moving world -> moving index)
//! 3. Linear interpolation (moving index -> intensity values)
//!
//! This inlines world-to-index construction and avoids repeated call-site
//! allocation around `Image::world_to_index_tensor` while preserving the
//! existing transform and interpolation contracts.

use coeus_core::{Backend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use ritk_core::interpolation::Interpolator;
use ritk_core::transform::Transform;
use ritk_image::native::Image;

use crate::interpolation::kernel::linear::LinearInterpolator;
use crate::interpolation::shared::compute_oob_mask;

/// Result of fused transform -> world-to-index -> interpolation.
///
/// Returns both the interpolated intensity values and an optional
/// out-of-bounds mask, allowing callers to zero out OOB contributions
/// in downstream histogram computations without recomputing indices.
pub struct FusedInterpolationResult<B: Backend> {
    /// Interpolated intensity values `[N]`.
    pub values: Tensor<f32, B>,
    /// Out-of-bounds mask `[N]`: `1.0` = in-bounds, `0.0` = out-of-bounds.
    pub oob_mask: Option<Tensor<f32, B>>,
}

/// Fused transform -> world-to-index -> linear interpolation.
///
/// Combines three operations that normally produce intermediate `[N, D]` tensors:
/// 1. Transform application (fixed world -> moving world)
/// 2. World-to-index conversion (moving world -> moving index)  — **inlined, no separate allocation**
/// 3. Linear interpolation (moving index -> intensity values)
///
/// Additionally computes an out-of-bounds mask from the inlined indices,
/// avoiding the need for the caller to retain the `moving_indices` tensor
/// just for OOB masking.
///
/// # Arguments
/// * `fixed_points` — `[N, D]` world-space points from the fixed image
/// * `transform` — spatial transform (fixed world -> moving world)
/// * `moving` — moving image (provides origin, spacing, direction, data)
/// * `interpolator` — linear interpolation method
///
/// # Returns
/// [`FusedInterpolationResult`] containing interpolated values `[N]` and
/// an OOB mask `[N]` (`1.0` = in-bounds, `0.0` = out-of-bounds).
pub fn transform_and_interpolate<B, T, const D: usize>(
    fixed_points: Tensor<f32, B>,
    transform: &T,
    moving: &Image<f32, B, D>,
    interpolator: &LinearInterpolator,
) -> FusedInterpolationResult<B>
where
    B: Backend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    T: Transform<B, D>,
{
    // Step 1: transform fixed-world -> moving-world
    let moving_world = transform.transform_points(fixed_points);

    // Step 2: world-to-index using the moving image's native method
    let moving_indices = moving.world_to_index_native(&moving_world);

    // Step 2b: compute OOB mask from indices before interpolation consumes them
    let oob_mask = compute_oob_mask(&moving_indices, &moving.shape());

    // Step 3: interpolate
    let values = interpolator.interpolate(moving.data(), moving_indices);

    FusedInterpolationResult {
        values,
        oob_mask: Some(oob_mask),
    }
}
