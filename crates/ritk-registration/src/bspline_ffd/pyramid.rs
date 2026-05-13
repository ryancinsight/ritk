//! Multi-resolution control grid refinement via B-spline subdivision.
//!
//! Implements the cubic B-spline subdivision rule (1D):
//!
//! ```text
//! Q[2i]     = (P[i-1] + 6·P[i] + P[i+1]) / 8
//! Q[2i + 1] = (P[i] + P[i+1]) / 2
//! ```
//!
//! Applied as three sequential separable 1D passes to refine a 3D control lattice.

use crate::deformable_field_ops::flat;

/// Double the control-grid resolution via B-spline subdivision.
///
/// Each control-point displacement is subdivided using the cubic B-spline
/// refinement mask so that the represented displacement field is preserved
/// exactly (to within floating-point precision). The control spacing is halved.
#[allow(clippy::type_complexity)]
pub(super) fn refine_control_grid(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, [usize; 3], [f64; 3]) {
    let [cnz, cny, cnx] = *ctrl_dims;

    let new_spacing = [
        ctrl_spacing[0] / 2.0,
        ctrl_spacing[1] / 2.0,
        ctrl_spacing[2] / 2.0,
    ];

    // new_dim = 2 * old_dim - 1 preserves the displacement field exactly.
    let new_dims = [2 * cnz - 1, 2 * cny - 1, 2 * cnx - 1];
    let [nnz, nny, nnx] = new_dims;
    let nn = nnz * nny * nnx;

    let mut new_z = vec![0.0_f32; nn];
    let mut new_y = vec![0.0_f32; nn];
    let mut new_x = vec![0.0_f32; nn];

    for comp_pair in [(cp_z, &mut new_z), (cp_y, &mut new_y), (cp_x, &mut new_x)] {
        let (old, new) = comp_pair;
        refine_component_3d(old, new, [cnz, cny, cnx], [nnz, nny, nnx]);
    }

    (new_z, new_y, new_x, new_dims, new_spacing)
}

/// Apply 3D B-spline subdivision to a single displacement component via three
/// sequential separable 1D passes.
fn refine_component_3d(old: &[f32], new: &mut [f32], old_dims: [usize; 3], new_dims: [usize; 3]) {
    let [oz, oy, ox] = old_dims;
    let [nz, ny, nx] = new_dims;

    // Pass 1: subdivide along X.  old [oz, oy, ox] -> tmp1 [oz, oy, nx]
    let mut tmp1 = vec![0.0_f32; oz * oy * nx];
    for iz in 0..oz {
        for iy in 0..oy {
            for jx in 0..nx {
                let ix = jx / 2;
                let v = if jx % 2 == 0 {
                    let pm = if ix > 0 {
                        old[flat(iz, iy, ix - 1, oy, ox)]
                    } else {
                        old[flat(iz, iy, 0, oy, ox)]
                    };
                    let p0 = old[flat(iz, iy, ix, oy, ox)];
                    let pp = if ix + 1 < ox {
                        old[flat(iz, iy, ix + 1, oy, ox)]
                    } else {
                        old[flat(iz, iy, ox - 1, oy, ox)]
                    };
                    (pm + 6.0 * p0 + pp) / 8.0
                } else {
                    let p0 = old[flat(iz, iy, ix, oy, ox)];
                    let pp = if ix + 1 < ox {
                        old[flat(iz, iy, ix + 1, oy, ox)]
                    } else {
                        old[flat(iz, iy, ox - 1, oy, ox)]
                    };
                    (p0 + pp) / 2.0
                };
                tmp1[iz * oy * nx + iy * nx + jx] = v;
            }
        }
    }

    // Pass 2: subdivide along Y.  tmp1 [oz, oy, nx] -> tmp2 [oz, ny, nx]
    let mut tmp2 = vec![0.0_f32; oz * ny * nx];
    for iz in 0..oz {
        for jy in 0..ny {
            let iy = jy / 2;
            for jx in 0..nx {
                let v = if jy % 2 == 0 {
                    let pm = if iy > 0 {
                        tmp1[iz * oy * nx + (iy - 1) * nx + jx]
                    } else {
                        tmp1[iz * oy * nx + jx]
                    };
                    let p0 = tmp1[iz * oy * nx + iy * nx + jx];
                    let pp = if iy + 1 < oy {
                        tmp1[iz * oy * nx + (iy + 1) * nx + jx]
                    } else {
                        tmp1[iz * oy * nx + (oy - 1) * nx + jx]
                    };
                    (pm + 6.0 * p0 + pp) / 8.0
                } else {
                    let p0 = tmp1[iz * oy * nx + iy * nx + jx];
                    let pp = if iy + 1 < oy {
                        tmp1[iz * oy * nx + (iy + 1) * nx + jx]
                    } else {
                        tmp1[iz * oy * nx + (oy - 1) * nx + jx]
                    };
                    (p0 + pp) / 2.0
                };
                tmp2[iz * ny * nx + jy * nx + jx] = v;
            }
        }
    }

    // Pass 3: subdivide along Z.  tmp2 [oz, ny, nx] -> new [nz, ny, nx]
    for jz in 0..nz {
        let iz = jz / 2;
        for jy in 0..ny {
            for jx in 0..nx {
                let v = if jz % 2 == 0 {
                    let pm = if iz > 0 {
                        tmp2[(iz - 1) * ny * nx + jy * nx + jx]
                    } else {
                        tmp2[jy * nx + jx]
                    };
                    let p0 = tmp2[iz * ny * nx + jy * nx + jx];
                    let pp = if iz + 1 < oz {
                        tmp2[(iz + 1) * ny * nx + jy * nx + jx]
                    } else {
                        tmp2[(oz - 1) * ny * nx + jy * nx + jx]
                    };
                    (pm + 6.0 * p0 + pp) / 8.0
                } else {
                    let p0 = tmp2[iz * ny * nx + jy * nx + jx];
                    let pp = if iz + 1 < oz {
                        tmp2[(iz + 1) * ny * nx + jy * nx + jx]
                    } else {
                        tmp2[(oz - 1) * ny * nx + jy * nx + jx]
                    };
                    (p0 + pp) / 2.0
                };
                new[jz * ny * nx + jy * nx + jx] = v;
            }
        }
    }
}
