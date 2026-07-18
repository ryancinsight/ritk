//! Gradient computation for SLIC center perturbation.

use super::coords::{decode_coords, encode_coords};

/// Compute gradient magnitude using central differences.
///
/// For each voxel, the gradient is the L2 norm of the partial derivatives
/// approximated by central differences along each axis. Boundary voxels
/// use forward/backward differences.
///
/// Dispatches to a const-generic implementation for D âˆˆ {1, 2, 3} to
/// eliminate per-voxel heap allocations.
///
/// # Panics
/// Panics if `ndim` is not in {1, 2, 3}. SLIC is only meaningfully
/// defined for 2-D and 3-D images in medical imaging contexts.
pub fn compute_gradient(data: &[f32], shape: &[usize], ndim: usize, scale: f32) -> Vec<f32> {
    match ndim {
        1 => compute_gradient_impl::<1>(data, shape, scale),
        2 => compute_gradient_impl::<2>(data, shape, scale),
        3 => compute_gradient_impl::<3>(data, shape, scale),
        _ => panic!("compute_gradient: unsupported dimensionality {}", ndim),
    }
}

/// Const-generic gradient computation.
///
/// `decode_coords` returns `[usize; D]` (stack-allocated, `Copy`),
/// so neighbour-coordinate mutations use cheap copies instead of
/// `Vec::clone()` â€” eliminating ~67M allocations for a 256Â³ image.
fn compute_gradient_impl<const D: usize>(data: &[f32], shape: &[usize], scale: f32) -> Vec<f32> {
    let shape_arr: [usize; D] = {
        let mut arr = [0usize; D];
        arr.copy_from_slice(shape);
        arr
    };
    let n: usize = shape_arr.iter().product();
    let mut grad = vec![0.0_f32; n];

    for i in 0..n {
        let coords = decode_coords(i, shape_arr);
        let mut sum_sq = 0.0_f32;
        for d in 0..D {
            let diff = if coords[d] > 0 && coords[d] < shape_arr[d] - 1 {
                let mut hi = coords;
                hi[d] += 1;
                let mut lo = coords;
                lo[d] -= 1;
                data[encode_coords(&hi, shape_arr)] - data[encode_coords(&lo, shape_arr)]
            } else if coords[d] == 0 && shape_arr[d] > 1 {
                let mut hi = coords;
                hi[d] += 1;
                data[encode_coords(&hi, shape_arr)] - data[i]
            } else if coords[d] == shape_arr[d] - 1 && shape_arr[d] > 1 {
                let mut lo = coords;
                lo[d] -= 1;
                data[i] - data[encode_coords(&lo, shape_arr)]
            } else {
                0.0
            };
            let normalized = diff / scale;
            sum_sq += normalized * normalized;
        }
        grad[i] = sum_sq.sqrt();
    }

    grad
}
