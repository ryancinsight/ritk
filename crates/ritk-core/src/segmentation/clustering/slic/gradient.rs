//! Gradient computation for SLIC center perturbation.

use super::coords::{decode_coords, encode_coords};

/// Compute gradient magnitude using central differences.
///
/// For each voxel, the gradient is the L2 norm of the partial derivatives
/// approximated by central differences along each axis. Boundary voxels
/// use forward/backward differences.
pub fn compute_gradient(data: &[f64], shape: &[usize], ndim: usize) -> Vec<f64> {
    let n: usize = shape.iter().product();
    let mut grad = vec![0.0_f64; n];

    for i in 0..n {
        let coords = decode_coords(i, shape);
        let mut sum_sq = 0.0_f64;
        for d in 0..ndim {
            let diff = if coords[d] > 0 && coords[d] < shape[d] - 1 {
                let mut hi = coords.clone();
                hi[d] += 1;
                let mut lo = coords.clone();
                lo[d] -= 1;
                data[encode_coords(&hi, shape)] - data[encode_coords(&lo, shape)]
            } else if coords[d] == 0 && shape[d] > 1 {
                let mut hi = coords.clone();
                hi[d] += 1;
                data[encode_coords(&hi, shape)] - data[i]
            } else if coords[d] == shape[d] - 1 && shape[d] > 1 {
                let mut lo = coords.clone();
                lo[d] -= 1;
                data[i] - data[encode_coords(&lo, shape)]
            } else {
                0.0
            };
            sum_sq += diff * diff;
        }
        grad[i] = sum_sq.sqrt();
    }

    grad
}
