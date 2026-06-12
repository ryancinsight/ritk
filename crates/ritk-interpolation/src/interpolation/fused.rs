//! Fused transform → world-to-index → linear interpolation for 3D images.
//!
//! Combines three operations that normally produce intermediate `[N, 3]` tensors:
//! 1. Transform application (fixed world → moving world)
//! 2. World-to-index conversion (moving world → moving index)
//! 3. Linear interpolation (moving index → intensity values)
//!
//! For the special case where the moving image has an identity direction matrix,
//! steps 2 and 3 are fused into a single pass, eliminating the intermediate
//! `moving_indices` tensor allocation (a `[N, 3]` tensor).
//!
//! For the general (non-identity direction) case, this still provides benefit by
//! inlining the world-to-index computation and passing indices directly to
//! interpolation, avoiding the method-call overhead and potential extra
//! allocations inside `world_to_index_tensor`.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::interpolation::shared::compute_oob_mask_3d;
use crate::interpolation::LinearInterpolator;
use ritk_core::interpolation::Interpolator;
use ritk_core::transform::Transform;
use ritk_image::Image;

/// Check whether a direction matrix is the identity (within tolerance).
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
    /// `None` when the moving image is not 3D or OOB masking is not needed.
    pub oob_mask: Option<Tensor<B, 1>>,
}

/// Fused transform → world-to-index → linear interpolation for 3D images.
///
/// Combines three operations that normally produce intermediate `[N, 3]` tensors:
/// 1. Transform application (fixed world → moving world)
/// 2. World-to-index conversion (moving world → moving index)  — **inlined, no separate allocation**
/// 3. Linear interpolation (moving index → intensity values)
///
/// Additionally computes an out-of-bounds mask from the inlined indices,
/// avoiding the need for the caller to retain the `moving_indices` tensor
/// just for OOB masking. This eliminates one `[N, 3]` intermediate allocation
/// compared to the unfused path (`transform_points` → `world_to_index_tensor`
/// → `compute_oob_mask_3d` → `interpolate`).
///
/// For the special case where the moving image has an identity direction matrix,
/// the world-to-index conversion uses element-wise `(world - origin) * inv_spacing`
/// instead of a matmul, further reducing compute cost.
///
/// # Arguments
/// * `fixed_points` — `[N, 3]` world-space points from the fixed image
/// * `transform` — spatial transform (fixed world → moving world)
/// * `moving` — moving image (provides origin, spacing, direction, data)
/// * `interpolator` — linear interpolation method
///
/// # Returns
/// [`FusedInterpolationResult`] containing interpolated values `[N]` and
/// an OOB mask `[N]` (`1.0` = in-bounds, `0.0` = out-of-bounds).
pub fn transform_and_interpolate_3d<B: Backend, T: Transform<B, 3>>(
    fixed_points: Tensor<B, 2>,
    transform: &T,
    moving: &Image<B, 3>,
    interpolator: &LinearInterpolator,
) -> FusedInterpolationResult<B> {
    let [n_points, _] = fixed_points.dims();
    let device = fixed_points.device();

    // ---- Step 1: transform fixed-world → moving-world ----
    // This allocation is unavoidable for a general transform.
    let moving_world = transform.transform_points(fixed_points);

    // ---- Step 2: world-to-index (fused, inlined) ----
    let indices = if is_identity_direction(moving.direction()) {
        // Fast path: identity direction means
        // index = (world - origin) / spacing
        // which can be computed with a simple broadcast subtraction and
        // element-wise multiply (by precomputed 1/spacing), avoiding the
        // matmul entirely.

        let origin_data: Vec<f32> = (0..3).map(|i| moving.origin()[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_data, burn::tensor::Shape::new([3])),
            &device,
        );

        let inv_spacing_data: Vec<f32> =
            (0..3).map(|i| (1.0 / moving.spacing()[i]) as f32).collect();
        let inv_spacing_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(inv_spacing_data, burn::tensor::Shape::new([3])),
            &device,
        );

        compute_identity_indices_chunked(moving_world, origin_tensor, inv_spacing_tensor, n_points)
    } else {
        // General path: inline the world-to-index matmul
        // index = (world - origin) @ T  where T = (D^-1)^T * S^-1

        let origin_data: Vec<f32> = (0..3).map(|i| moving.origin()[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_data, burn::tensor::Shape::new([3])),
            &device,
        );

        let inv_dir = moving
            .direction()
            .try_inverse()
            .expect("Direction matrix must be invertible");
        let mut t_data = Vec::with_capacity(9);
        for r in 0..3 {
            for c in 0..3 {
                let val = (inv_dir[(c, r)] / moving.spacing()[c]) as f32;
                t_data.push(val);
            }
        }
        let t_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(t_data, burn::tensor::Shape::new([3, 3])),
            &device,
        );

        compute_general_indices_chunked(moving_world, origin_tensor, t_tensor, n_points)
    };

    // ---- Step 2b: compute OOB mask from indices before interpolation consumes them ----
    let oob_mask = compute_oob_mask_3d(&indices, &moving.shape());

    // ---- Step 3: interpolate ----
    let values = interpolator.interpolate(moving.data(), indices);

    FusedInterpolationResult {
        values,
        oob_mask: Some(oob_mask),
    }
}

/// Compute `(world - origin) * inv_spacing` with chunking for WGPU dispatch limits.
///
/// For identity-direction images, this replaces the matmul with a simple
/// element-wise multiply by the precomputed inverse spacing.
fn compute_identity_indices_chunked<B: Backend>(
    world: Tensor<B, 2>,
    origin: Tensor<B, 1>,      // shape [3], will be reshaped to [1, 3] per chunk
    inv_spacing: Tensor<B, 1>, // shape [3], will be reshaped to [1, 3] per chunk
    n_points: usize,
) -> Tensor<B, 2> {
    let origin = origin.reshape([1usize, 3]);
    let inv_spacing = inv_spacing.reshape([1usize, 3]);

    if n_points <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
        (world - origin) * inv_spacing
    } else {
        let num_chunks = n_points.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
        let mut chunks = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
            let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n_points);
            let chunk_range = start..end;
            // slice() consumes self; tensor clones are refcounted handle
            // copies, not data copies, so cloning `world` per chunk is cheap.
            // The [1,3] broadcast tensors are likewise cheap to clone.
            let chunk_world = world.clone().slice([chunk_range]);
            let result = (chunk_world - origin.clone()) * inv_spacing.clone();
            chunks.push(result);
        }
        Tensor::cat(chunks, 0)
    }
}

/// Compute `(world - origin) @ T` with chunking for WGPU dispatch limits.
///
/// For general-direction images, this inlines the world-to-index matmul.
fn compute_general_indices_chunked<B: Backend>(
    world: Tensor<B, 2>,
    origin: Tensor<B, 1>, // shape [3], will be reshaped to [1, 3] per chunk
    t: Tensor<B, 2>,      // shape [3, 3]
    n_points: usize,
) -> Tensor<B, 2> {
    let origin = origin.reshape([1usize, 3]);

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
