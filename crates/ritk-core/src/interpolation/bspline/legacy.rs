//! Legacy tensor-based B-Spline interpolation functions.
//!
//! These implementations use tensor operations (slice, clone, mul_scalar) and
//! are retained for reference but are not used in production — the flat-slice
//! variants in [`super::flat`] are preferred for performance.

use super::cubic_bspline;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// 3D B-Spline interpolation for a single point (legacy version using tensor operations).
///
/// The data tensor layout is `[dim0, dim1, dim2]` and `coords` are
/// `[coord0, coord1, coord2]` indexing the respective dimensions.
///
/// When `zero_pad` is `true` and `floor(coord_d)` lies outside `[0, dim_d - 1]`
/// for any dimension `d`, the function returns `0.0` immediately.
#[allow(dead_code)]
pub(super) fn interpolate_point_3d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    coords: &[f32],
    dims: &[usize],
    device: &B::Device,
    zero_pad: bool,
) -> Tensor<B, 1> {
    let x = coords[0];
    let y = coords[1];
    let z = coords[2];

    // Zero-pad early exit: if the query coordinate itself is outside the volume,
    // return 0.0 immediately. This mirrors the Linear and NearestNeighbor
    // zero-pad semantics where `floor(coord) == clamp(floor(coord), 0, dim-1)`
    // is the in-bounds criterion.
    if zero_pad {
        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        let zf = z.floor() as isize;
        if xf < 0
            || xf >= dims[0] as isize
            || yf < 0
            || yf >= dims[1] as isize
            || zf < 0
            || zf >= dims[2] as isize
        {
            return Tensor::zeros([1], device);
        }
    }

    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;
    let z0 = z.floor() as isize - 1;

    let mut result = Tensor::zeros([1], device);
    let mut weight_sum = 0.0f32;

    // Sample 4x4x4 neighborhood
    for dz in 0..4 {
        for dy in 0..4 {
            for dx in 0..4 {
                let xi = x0 + dx;
                let yi = y0 + dy;
                let zi = z0 + dz;

                // Compute B-Spline weights
                let wx = cubic_bspline(x - xi as f32);
                let wy = cubic_bspline(y - yi as f32);
                let wz = cubic_bspline(z - zi as f32);
                let weight = wx * wy * wz;

                // Check bounds and sample
                // Performance: use slice without clone to avoid O(volume_size) allocation.
                // The slice operation creates a view, and we only need to clone the single
                // element we're sampling, not the entire volume.
                if xi >= 0
                    && xi < dims[0] as isize
                    && yi >= 0
                    && yi < dims[1] as isize
                    && zi >= 0
                    && zi < dims[2] as isize
                {
                    let sample = data.clone().slice([
                        xi as usize..xi as usize + 1,
                        yi as usize..yi as usize + 1,
                        zi as usize..zi as usize + 1,
                    ]);
                    let sample_scalar = sample.reshape([1]);
                    result = result.add(sample_scalar.mul_scalar(weight));
                    weight_sum += weight;
                }
            }
        }
    }

    // Normalize by weight sum (handles boundary renormalization when neighbors are OOB)
    if weight_sum > 0.0 {
        result = result.div_scalar(weight_sum);
    }
    result
}

/// 2D B-Spline interpolation for a single point (legacy version using tensor operations).
///
/// When `zero_pad` is `true` and `floor(coord_d)` lies outside `[0, dim_d - 1]`
/// for any dimension `d`, the function returns `0.0` immediately.
#[allow(dead_code)]
pub(super) fn interpolate_point_2d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    coords: &[f32],
    dims: &[usize],
    device: &B::Device,
    zero_pad: bool,
) -> Tensor<B, 1> {
    let x = coords[0];
    let y = coords[1];

    // Zero-pad early exit: if the query coordinate itself is outside the image.
    if zero_pad {
        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        if xf < 0 || xf >= dims[0] as isize || yf < 0 || yf >= dims[1] as isize {
            return Tensor::zeros([1], device);
        }
    }

    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;

    let mut result = Tensor::zeros([1], device);
    let mut weight_sum = 0.0f32;

    // Sample 4x4 neighborhood
    for dy in 0..4 {
        for dx in 0..4 {
            let xi = x0 + dx;
            let yi = y0 + dy;

            // Compute B-Spline weights
            let wx = cubic_bspline(x - xi as f32);
            let wy = cubic_bspline(y - yi as f32);
            let weight = wx * wy;

            // Check bounds and sample
            // Performance: use slice without clone to avoid O(image_size) allocation.
            if xi >= 0 && xi < dims[0] as isize && yi >= 0 && yi < dims[1] as isize {
                let sample = data
                    .clone()
                    .slice([xi as usize..xi as usize + 1, yi as usize..yi as usize + 1]);
                let sample_scalar = sample.reshape([1]);
                result = result.add(sample_scalar.mul_scalar(weight));
                weight_sum += weight;
            }
        }
    }

    // Normalize by weight sum
    if weight_sum > 0.0 {
        result = result.div_scalar(weight_sum);
    }
    result
}
