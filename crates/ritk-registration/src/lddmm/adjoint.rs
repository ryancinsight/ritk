//! EPDiff coadjoint operator ad*_v(m).

use crate::deformable_field_ops::flat;

/// Write the EPDiff coadjoint operator ad\*\_v(m) into caller-provided buffers.
///
/// Performs zero heap allocation. All output buffers must have length `nz*ny*nx`.
pub(super) fn epdiff_adjoint_into(
    vz: &[f32],
    vy: &[f32],
    vx: &[f32],
    mz: &[f32],
    my: &[f32],
    mx: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    ad_z: &mut [f32],
    ad_y: &mut [f32],
    ad_x: &mut [f32],
) {
    let [nz, ny, nx] = dims;
    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = flat(iz, iy, ix, ny, nx);

                let ddz = |f: &[f32]| -> f32 {
                    if nz == 1 {
                        0.0
                    } else if iz == 0 {
                        (f[flat(1, iy, ix, ny, nx)] - f[fi]) / sz
                    } else if iz == nz - 1 {
                        (f[fi] - f[flat(nz - 2, iy, ix, ny, nx)]) / sz
                    } else {
                        (f[flat(iz + 1, iy, ix, ny, nx)] - f[flat(iz - 1, iy, ix, ny, nx)])
                            / (2.0 * sz)
                    }
                };
                let ddy = |f: &[f32]| -> f32 {
                    if ny == 1 {
                        0.0
                    } else if iy == 0 {
                        (f[flat(iz, 1, ix, ny, nx)] - f[fi]) / sy
                    } else if iy == ny - 1 {
                        (f[fi] - f[flat(iz, ny - 2, ix, ny, nx)]) / sy
                    } else {
                        (f[flat(iz, iy + 1, ix, ny, nx)] - f[flat(iz, iy - 1, ix, ny, nx)])
                            / (2.0 * sy)
                    }
                };
                let ddx = |f: &[f32]| -> f32 {
                    if nx == 1 {
                        0.0
                    } else if ix == 0 {
                        (f[flat(iz, iy, 1, ny, nx)] - f[fi]) / sx
                    } else if ix == nx - 1 {
                        (f[fi] - f[flat(iz, iy, nx - 2, ny, nx)]) / sx
                    } else {
                        (f[flat(iz, iy, ix + 1, ny, nx)] - f[flat(iz, iy, ix - 1, ny, nx)])
                            / (2.0 * sx)
                    }
                };

                let div_v = ddz(vz) + ddy(vy) + ddx(vx);

                ad_z[fi] = vz[fi] * ddz(mz)
                    + vy[fi] * ddy(mz)
                    + vx[fi] * ddx(mz)
                    + mz[fi] * ddz(vz)
                    + my[fi] * ddy(vz)
                    + mx[fi] * ddx(vz)
                    + mz[fi] * div_v;

                ad_y[fi] = vz[fi] * ddz(my)
                    + vy[fi] * ddy(my)
                    + vx[fi] * ddx(my)
                    + mz[fi] * ddz(vy)
                    + my[fi] * ddy(vy)
                    + mx[fi] * ddx(vy)
                    + my[fi] * div_v;

                ad_x[fi] = vz[fi] * ddz(mx)
                    + vy[fi] * ddy(mx)
                    + vx[fi] * ddx(mx)
                    + mz[fi] * ddz(vx)
                    + my[fi] * ddy(vx)
                    + mx[fi] * ddx(vx)
                    + mx[fi] * div_v;
            }
        }
    }
}

/// Compute the EPDiff coadjoint operator ad\*_v(m) (allocating convenience wrapper).
#[cfg(test)]
pub(super) fn epdiff_adjoint(
    vz: &[f32],
    vy: &[f32],
    vx: &[f32],
    mz: &[f32],
    my: &[f32],
    mx: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = dims[0] * dims[1] * dims[2];
    let mut ad_z = vec![0.0_f32; n];
    let mut ad_y = vec![0.0_f32; n];
    let mut ad_x = vec![0.0_f32; n];
    epdiff_adjoint_into(
        vz, vy, vx, mz, my, mx, dims, spacing, &mut ad_z, &mut ad_y, &mut ad_x,
    );
    (ad_z, ad_y, ad_x)
}
