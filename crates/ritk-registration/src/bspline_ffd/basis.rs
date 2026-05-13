//! Cubic B-spline basis functions and control grid evaluation.
//!
//! Implements the Rueckert (1999) uniform cubic B-spline basis:
//!
//! ```text
//! β₃₀(t) = (1 − t)³ / 6
//! β₃₁(t) = (3t³ − 6t² + 4) / 6
//! β₃₂(t) = (−3t³ + 3t² + 3t + 1) / 6
//! β₃₃(t) = t³ / 6
//! ```

use crate::deformable_field_ops::flat;

/// Evaluate the four cubic B-spline basis values at parameter `t ∈ [0, 1]`.
///
/// Returns `[β₃₀(t), β₃₁(t), β₃₂(t), β₃₃(t)]`. These sum to 1.0 (partition
/// of unity) and are non-negative on `[0, 1]`.
#[inline]
pub(super) fn cubic_bspline_1d(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt3 = omt * omt * omt;

    [
        omt3 / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// Evaluate the four cubic B-spline basis *derivatives* at parameter `t ∈ [0, 1]`.
///
/// Returns `d/dt [β₃₀(t), β₃₁(t), β₃₂(t), β₃₃(t)]`:
///
/// ```text
/// β₃₀'(t) = −(1 − t)² / 2
/// β₃₁'(t) = (3t² − 4t) / 2
/// β₃₂'(t) = (−3t² + 2t + 1) / 2
/// β₃₃'(t) = t² / 2
/// ```
#[inline]
#[allow(dead_code)]
pub(super) fn cubic_bspline_1d_deriv(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let omt = 1.0 - t;
    [
        -omt * omt / 2.0,
        (3.0 * t2 - 4.0 * t) / 2.0,
        (-3.0 * t2 + 2.0 * t + 1.0) / 2.0,
        t2 / 2.0,
    ]
}

/// Compute control-grid dimensions from image dimensions and control spacing.
///
/// The control lattice extends one extra control point beyond each boundary
/// to ensure full support coverage. Grid dimension along axis `d`:
///
/// ```text
/// n_ctrl[d] = ceil(dims[d] / spacing[d]) + 3
/// ```
///
/// The `+3` accounts for one point before the domain origin and two points
/// after the far boundary, providing the four-point support stencil at every
/// image voxel.
pub(super) fn init_control_grid(dims: [usize; 3], ctrl_spacing: &[f64; 3]) -> [usize; 3] {
    let mut ctrl_dims = [0usize; 3];
    for d in 0..3 {
        ctrl_dims[d] = (dims[d] as f64 / ctrl_spacing[d]).ceil() as usize + 3;
    }
    ctrl_dims
}

/// Evaluate the dense displacement field from B-spline control points.
///
/// For each image voxel `(iz, iy, ix)`, computes the displacement as the
/// tensor-product of 1D cubic B-spline bases evaluated over the 4×4×4
/// neighborhood of control points.
///
/// # Returns
/// `(dz, dy, dx)` — displacement components in voxel units, each of length
/// `dims[0] * dims[1] * dims[2]`.
pub(super) fn evaluate_bspline_displacement(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    dims: [usize; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let [cnz, cny, cnx] = *ctrl_dims;

    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];

    for iz in 0..nz {
        // Map image coordinate to control-grid parameter space.
        // The control grid origin is at index 1 (one point of padding before
        // the domain), so: u = iz / spacing + 1.
        let uz = iz as f64 / ctrl_spacing[0] + 1.0;
        let kz = uz.floor() as isize - 1;
        let tz = uz - (kz + 1) as f64;
        let bz = cubic_bspline_1d(tz);

        for iy in 0..ny {
            let uy = iy as f64 / ctrl_spacing[1] + 1.0;
            let ky = uy.floor() as isize - 1;
            let ty = uy - (ky + 1) as f64;
            let by = cubic_bspline_1d(ty);

            for ix in 0..nx {
                let ux = ix as f64 / ctrl_spacing[2] + 1.0;
                let kx = ux.floor() as isize - 1;
                let tx = ux - (kx + 1) as f64;
                let bx = cubic_bspline_1d(tx);

                let fi = flat(iz, iy, ix, ny, nx);
                let mut sum_z = 0.0_f64;
                let mut sum_y = 0.0_f64;
                let mut sum_x = 0.0_f64;

                for az in 0..4isize {
                    let ciz = kz + az;
                    if ciz < 0 || ciz >= cnz as isize {
                        continue;
                    }
                    let ciz = ciz as usize;
                    let wz = bz[az as usize];

                    for ay in 0..4isize {
                        let ciy = ky + ay;
                        if ciy < 0 || ciy >= cny as isize {
                            continue;
                        }
                        let ciy = ciy as usize;
                        let wzy = wz * by[ay as usize];

                        for ax in 0..4isize {
                            let cix = kx + ax;
                            if cix < 0 || cix >= cnx as isize {
                                continue;
                            }
                            let cix = cix as usize;
                            let w = wzy * bx[ax as usize];

                            let ci = flat(ciz, ciy, cix, cny, cnx);
                            sum_z += w * cp_z[ci] as f64;
                            sum_y += w * cp_y[ci] as f64;
                            sum_x += w * cp_x[ci] as f64;
                        }
                    }
                }

                dz[fi] = sum_z as f32;
                dy[fi] = sum_y as f32;
                dx[fi] = sum_x as f32;
            }
        }
    }

    (dz, dy, dx)
}
