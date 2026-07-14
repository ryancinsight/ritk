//! Image gradient computation via finite differences.

use super::{flat, VelocityField};
use ritk_spatial::VolumeDims;

/// Write the gradient of `data` into caller-provided buffers.
///
/// Uses central differences at interior voxels and one-sided first-order
/// differences at boundaries. No allocation occurs; all results are written
/// into `gz`, `gy`, `gx`, each of length `dims[0] * dims[1] * dims[2]`.
pub(crate) fn compute_gradient_into(
    data: &[f32],
    dims: VolumeDims,
    spacing: [f64; 3],
    gz: &mut [f32],
    gy: &mut [f32],
    gx: &mut [f32],
) {
    let [nz, ny, nx] = dims.0;
    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

    let slice_len = ny * nx;
    moirai::for_each_chunk_triple_mut_enumerated_with::<moirai::Adaptive, _, _, _, _>(
        gz,
        gy,
        gx,
        slice_len,
        |iz, gz_slice, gy_slice, gx_slice| {
            for iy in 0..ny {
                for ix in 0..nx {
                    let local = iy * nx + ix;
                    let fi = iz * slice_len + local;

                    gz_slice[local] = if nz == 1 {
                        0.0
                    } else if iz == 0 {
                        (data[flat(1, iy, ix, ny, nx)] - data[fi]) / sz
                    } else if iz == nz - 1 {
                        (data[fi] - data[flat(nz - 2, iy, ix, ny, nx)]) / sz
                    } else {
                        (data[flat(iz + 1, iy, ix, ny, nx)] - data[flat(iz - 1, iy, ix, ny, nx)])
                            / (2.0 * sz)
                    };

                    gy_slice[local] = if ny == 1 {
                        0.0
                    } else if iy == 0 {
                        (data[flat(iz, 1, ix, ny, nx)] - data[fi]) / sy
                    } else if iy == ny - 1 {
                        (data[fi] - data[flat(iz, ny - 2, ix, ny, nx)]) / sy
                    } else {
                        (data[flat(iz, iy + 1, ix, ny, nx)] - data[flat(iz, iy - 1, ix, ny, nx)])
                            / (2.0 * sy)
                    };

                    gx_slice[local] = if nx == 1 {
                        0.0
                    } else if ix == 0 {
                        (data[flat(iz, iy, 1, ny, nx)] - data[fi]) / sx
                    } else if ix == nx - 1 {
                        (data[fi] - data[flat(iz, iy, nx - 2, ny, nx)]) / sx
                    } else {
                        (data[flat(iz, iy, ix + 1, ny, nx)] - data[flat(iz, iy, ix - 1, ny, nx)])
                            / (2.0 * sx)
                    };
                }
            }
        },
    );
}

/// Compute the gradient of `data` via central differences at interior voxels
/// and one-sided first-order differences at boundaries.
///
/// Each component is divided by the corresponding physical `spacing` so that
/// the result is in (intensity / length) units.
///
/// # Returns
/// `(gz, gy, gx)` — three flat `Vec<f32>` of length `nz * ny * nx`.
pub(crate) fn compute_gradient(data: &[f32], dims: VolumeDims, spacing: [f64; 3]) -> VelocityField {
    let n = dims.total_voxels();
    let mut gz = vec![0.0_f32; n];
    let mut gy = vec![0.0_f32; n];
    let mut gx = vec![0.0_f32; n];
    compute_gradient_into(data, dims, spacing, &mut gz, &mut gy, &mut gx);
    VelocityField {
        z: gz,
        y: gy,
        x: gx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deformable_field_ops::flat;

    /// Gradient of a linear ramp I[z,y,x] = x should be (0, 0, 1/sx).
    #[test]
    fn gradient_linear_ramp_x() {
        let dims = VolumeDims::new([4, 4, 8]);
        let [nz, ny, nx] = dims.0;
        let data: Vec<f32> = (0..nz * ny * nx).map(|fi| (fi % nx) as f32).collect();
        let spacing = [1.0, 1.0, 1.0];
        let grad = compute_gradient(&data, dims, spacing);

        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let fi = flat(iz, iy, ix, ny, nx);
                    assert!(
                        (grad.z[fi]).abs() < 1e-5,
                        "gz should be 0, got {}",
                        grad.z[fi]
                    );
                    assert!(
                        (grad.y[fi]).abs() < 1e-5,
                        "gy should be 0, got {}",
                        grad.y[fi]
                    );
                    assert!(
                        (grad.x[fi] - 1.0).abs() < 1e-5,
                        "gx should be 1, got {}",
                        grad.x[fi]
                    );
                }
            }
        }
    }

    /// Gradient of a constant field is zero everywhere.
    #[test]
    fn gradient_constant_field_is_zero() {
        let dims = VolumeDims::new([4, 4, 4]);
        let [nz, ny, nx] = dims.0;
        let data = vec![5.0_f32; nz * ny * nx];
        let grad = compute_gradient(&data, dims, [1.0; 3]);
        for i in 0..nz * ny * nx {
            assert_eq!(grad.z[i], 0.0, "gz[{i}] should be 0");
            assert_eq!(grad.y[i], 0.0, "gy[{i}] should be 0");
            assert_eq!(grad.x[i], 0.0, "gx[{i}] should be 0");
        }
    }
}
