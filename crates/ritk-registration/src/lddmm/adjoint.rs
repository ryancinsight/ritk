//! EPDiff coadjoint operator ad*_v(m).

#[cfg(test)]
use crate::deformable_field_ops::VelocityField;
use crate::deformable_field_ops::{flat, VectorField, VectorFieldMut};

/// Compute the EPDiff coadjoint operator ad\*\_v(m).
///
/// For each spatial component i âˆˆ {z, y, x}:
///
///   (ad\*\_v m)\_i = Î£\_j \[v\_j Â· âˆ‚m\_i/âˆ‚x\_j + m\_j Â· âˆ‚v\_i/âˆ‚x\_j\] + m\_i Â· div(v)
///
/// Derivatives use central differences at interior voxels and one-sided
/// differences at boundaries, consistent with [`compute_gradient`].
#[cfg(test)]
pub(super) fn epdiff_adjoint(
    v: VectorField<'_>,
    m: VectorField<'_>,
    dims: [usize; 3],
    spacing: [f64; 3],
) -> VelocityField {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut ad_z = vec![0.0_f32; n];
    let mut ad_y = vec![0.0_f32; n];
    let mut ad_x = vec![0.0_f32; n];
    epdiff_adjoint_into(
        v,
        m,
        dims,
        spacing,
        VectorFieldMut {
            z: &mut ad_z,
            y: &mut ad_y,
            x: &mut ad_x },
    );
    VelocityField {
        z: ad_z,
        y: ad_y,
        x: ad_x }
}

/// Zero-allocation variant of `epdiff_adjoint`.
///
/// Writes the result directly into `out` instead of allocating new `Vec`s.
/// `out` must have the same length as `v.z` (i.e. `dims[0]*dims[1]*dims[2]`).
pub(super) fn epdiff_adjoint_into(
    v: VectorField<'_>,
    m: VectorField<'_>,
    dims: [usize; 3],
    spacing: [f64; 3],
    out: VectorFieldMut<'_>,
) {
    let VectorField {
        z: vz,
        y: vy,
        x: vx } = v;
    let VectorField {
        z: mz,
        y: my,
        x: mx } = m;
    let VectorFieldMut {
        z: ad_z,
        y: ad_y,
        x: ad_x } = out;
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
