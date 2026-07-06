//! Fused transform → world-to-index → linear interpolation.
//!
//! Combines three operations that normally produce intermediate `[N, D]` tensors:
//! 1. Transform application (fixed world → moving world)
//! 2. World-to-index conversion (moving world → moving index)
//! 3. Linear interpolation (moving index → intensity values)
//!
//! This inlines world-to-index construction and avoids repeated call-site
//! allocation around `Image::world_to_index_tensor` while preserving the
//! existing transform and interpolation contracts.

use ritk_image::tensor::Backend;
use ritk_image::tensor::{Tensor, TensorData};

use crate::interpolation::shared::compute_oob_mask;
use crate::interpolation::LinearInterpolator;
use ritk_core::interpolation::Interpolator;
use ritk_core::transform::Transform;
use ritk_image::Image;

/// Check whether a direction matrix is the identity (within tolerance).
#[cfg(test)]
pub(crate) fn is_identity_direction<const D: usize>(
    direction: &ritk_spatial::Direction<D>,
) -> bool {
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

/// Result of fused transform → world-to-index → interpolation.
///
/// Returns both the interpolated intensity values and an optional
/// out-of-bounds mask, allowing callers to zero out OOB contributions
/// in downstream histogram computations without recomputing indices.
pub struct FusedInterpolationResult<B: Backend> {
    /// Interpolated intensity values `[N]`.
    pub values: Tensor<B, 1>,
    /// Out-of-bounds mask `[N]`: `1.0` = in-bounds, `0.0` = out-of-bounds.
    /// Currently populated by this fused path.
    pub oob_mask: Option<Tensor<B, 1>>,
}

/// Fused transform → world-to-index → linear interpolation.
///
/// Combines three operations that normally produce intermediate `[N, D]` tensors:
/// 1. Transform application (fixed world → moving world)
/// 2. World-to-index conversion (moving world → moving index)  — **inlined, no separate allocation**
/// 3. Linear interpolation (moving index → intensity values)
///
/// Additionally computes an out-of-bounds mask from the inlined indices,
/// avoiding the need for the caller to retain the `moving_indices` tensor
/// just for OOB masking.
///
/// # Arguments
/// * `fixed_points` — `[N, D]` world-space points from the fixed image
/// * `transform` — spatial transform (fixed world → moving world)
/// * `moving` — moving image (provides origin, spacing, direction, data)
/// * `interpolator` — linear interpolation method
///
/// # Returns
/// [`FusedInterpolationResult`] containing interpolated values `[N]` and
/// an OOB mask `[N]` (`1.0` = in-bounds, `0.0` = out-of-bounds).
pub fn transform_and_interpolate<B: Backend, T: Transform<B, D>, const D: usize>(
    fixed_points: Tensor<B, 2>,
    transform: &T,
    moving: &Image<B, D>,
    interpolator: &LinearInterpolator,
) -> FusedInterpolationResult<B> {
    let [n_points, _] = fixed_points.dims();
    let device = fixed_points.device();

    // ---- Step 1: transform fixed-world → moving-world ----
    // This allocation is unavoidable for a general transform.
    let moving_world = transform.transform_points(fixed_points);

    // ---- Step 2: world-to-index (fused, inlined) ----
    let origin_data: Vec<f32> = (0..D).map(|i| moving.origin()[i] as f32).collect();
    let origin_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(origin_data, ritk_image::tensor::Shape::new([D])),
        &device,
    );

    let inv_dir = moving
        .direction()
        .try_inverse()
        .expect("Direction matrix must be invertible");
    let mut t_data = Vec::with_capacity(D * D);
    for r in 0..D {
        for c in 0..D {
            let axis = D - 1 - c;
            let val = (inv_dir[(axis, r)] / moving.spacing()[axis]) as f32;
            t_data.push(val);
        }
    }
    let t_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(t_data, ritk_image::tensor::Shape::new([D, D])),
        &device,
    );

    let indices =
        compute_general_indices_chunked::<B, D>(moving_world, origin_tensor, t_tensor, n_points);

    // ---- Step 2b: compute OOB mask from indices before interpolation consumes them ----
    let oob_mask = compute_oob_mask(&indices, &moving.shape());

    // ---- Step 3: interpolate ----
    let values = interpolator.interpolate(moving.data(), indices);

    FusedInterpolationResult {
        values,
        oob_mask: Some(oob_mask),
    }
}

/// Compute `(world - origin) @ T` with chunking for WGPU dispatch limits.
///
/// For general-direction images, this inlines the world-to-index matmul.
fn compute_general_indices_chunked<B: Backend, const D: usize>(
    world: Tensor<B, 2>,
    origin: Tensor<B, 1>,
    t: Tensor<B, 2>,
    n_points: usize,
) -> Tensor<B, 2> {
    let origin = origin.reshape([1usize, D]);

    if n_points <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
        let diff = world - origin;
        diff.matmul(t)
    } else {
        let num_chunks = n_points.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
        let mut chunks = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
            let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n_points);
            let chunk_range = start..end;
            // slice() consumes self; tensor clones are refcounted handle
            // copies, not data copies. t is [3,3] — cheap to clone.
            let chunk_world = world.clone().slice([chunk_range]);
            let diff = chunk_world - origin.clone();
            let result = diff.matmul(t.clone());
            chunks.push(result);
        }
        Tensor::cat(chunks, 0)
    }
}

#[cfg(test)]
#[path = "tests/fused.rs"]
mod tests;
