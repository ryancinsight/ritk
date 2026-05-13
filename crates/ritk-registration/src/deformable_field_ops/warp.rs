//! Image warping and streaming MSE under displacement fields.

use super::{flat, trilinear_interpolate};

/// Warp `moving` by the displacement field into a caller-provided buffer.
///
/// For each voxel `p = (iz, iy, ix)`:
///   `output[p] = moving(iz + dz[p], iy + dy[p], ix + dx[p])`
/// sampled with trilinear interpolation and clamp-to-border BC.
/// `output` must have length `dims[0] * dims[1] * dims[2]`.
pub(crate) fn warp_image_into(
    moving: &[f32],
    dims: [usize; 3],
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
    output: &mut [f32],
) {
    let [nz, ny, nx] = dims;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let wz = iz as f32 + dz[fi];
                let wy = iy as f32 + dy[fi];
                let wx = ix as f32 + dx[fi];
                output[fi] = trilinear_interpolate(moving, dims, wz, wy, wx);
            }
        }
    }
}

/// Warp `moving` by the displacement field `(dz, dy, dx)`.
///
/// For each voxel `p = (iz, iy, ix)`:
///   `warped(p) = moving(iz + dz[p], iy + dy[p], ix + dx[p])`
/// sampled with trilinear interpolation and clamp-to-border BC.
pub(crate) fn warp_image(
    moving: &[f32],
    dims: [usize; 3],
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    let mut warped = vec![0.0_f32; n];
    warp_image_into(moving, dims, dz, dy, dx, &mut warped);
    warped
}

/// Compute `mean((fixed − warp(moving, D))²)` without materialising a warped buffer.
///
/// Streams trilinear samples of `moving` under displacement `D = (dz, dy, dx)` directly
/// into a squared-error accumulator. No intermediate `Vec<f32>` is allocated.
///
/// Returns the mean squared error as `f64`.
pub(crate) fn compute_mse_streaming(
    fixed: &[f32],
    moving: &[f32],
    dims: [usize; 3],
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
) -> f64 {
    let [nz, ny, nx] = dims;
    let mut sum = 0.0_f64;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);
                let wz = iz as f32 + dz[fi];
                let wy = iy as f32 + dy[fi];
                let wx = ix as f32 + dx[fi];
                let warped = trilinear_interpolate(moving, dims, wz, wy, wx);
                let diff = (fixed[fi] - warped) as f64;
                sum += diff * diff;
            }
        }
    }
    sum / fixed.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Warp of a constant image is constant regardless of displacement.
    #[test]
    fn warp_constant_image() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let data = vec![5.0_f32; n];
        let dz = vec![0.5_f32; n];
        let dy = vec![-0.3_f32; n];
        let dx = vec![1.1_f32; n];
        let out = warp_image(&data, dims, &dz, &dy, &dx);
        for &v in &out {
            assert!((v - 5.0).abs() < 1e-5, "expected 5.0, got {v}");
        }
    }

    /// Warp with zero displacement must return the original image.
    #[test]
    fn warp_identity_displacement() {
        let dims = [6usize, 6, 6];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let zeros = vec![0.0_f32; n];
        let out = warp_image(&data, dims, &zeros, &zeros, &zeros);
        for (i, (&orig, &warped)) in data.iter().zip(out.iter()).enumerate() {
            assert!(
                (orig - warped).abs() < 1e-5,
                "voxel {i}: expected {orig}, got {warped}"
            );
        }
    }

    /// Streaming MSE of identical images with zero displacement is zero.
    #[test]
    fn streaming_mse_identical_is_zero() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let zeros = vec![0.0_f32; n];
        let mse = compute_mse_streaming(&data, &data, dims, &zeros, &zeros, &zeros);
        assert!(mse < 1e-10, "expected ~0, got {mse}");
    }
}
