//! Local cross-correlation (CC) metric primitives for BSplineSyN.
//!
//! # Algorithm
//! Local CC metric (Avants 2008, eq. 10): for each voxel p with window W of
//! radius `r`:
//!
//!   force_scale(p) = (J_w(p)−μ_J) / √(σ_I²·σ_J²) − CC · (I_w(p)−μ_I) / σ_I²
//!   f_k(p)        = force_scale(p) · ∇_k I_w(p)

use crate::deformable_field_ops::flat;

/// Compute local CC gradient forces (Avants 2008, eq. 10).
pub(crate) fn cc_forces(
    i_w: &[f32],
    j_w: &[f32],
    gi_z: &[f32],
    gi_y: &[f32],
    gi_x: &[f32],
    dims: [usize; 3],
    radius: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut fz = vec![0.0_f32; n];
    let mut fy = vec![0.0_f32; n];
    let mut fx = vec![0.0_f32; n];
    let r = radius as isize;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let (mut si, mut sj, mut cnt) = (0.0_f64, 0.0_f64, 0u32);
                for dz in -r..=r {
                    let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            let qi = flat(qz, qy, qx, ny, nx);
                            si += i_w[qi] as f64;
                            sj += j_w[qi] as f64;
                            cnt += 1;
                        }
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let (mu_i, mu_j) = (si / cnt as f64, sj / cnt as f64);

                let (mut num, mut vi, mut vj) = (0.0_f64, 0.0_f64, 0.0_f64);
                for dz in -r..=r {
                    let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            let qi = flat(qz, qy, qx, ny, nx);
                            let di = i_w[qi] as f64 - mu_i;
                            let dj = j_w[qi] as f64 - mu_j;
                            num += di * dj;
                            vi += di * di;
                            vj += dj * dj;
                        }
                    }
                }
                if vi < 1e-10 {
                    continue;
                }

                let fi = flat(iz, iy, ix, ny, nx);
                let iw_c = i_w[fi] as f64 - mu_i;
                let jw_c = j_w[fi] as f64 - mu_j;
                // Avants 2008, eq. 10 — gradient ascent on local CC.
                // ∂CC/∂v₁_k(x) = [(J_w−μ_J)/√(σ_I²·σ_J²) − CC·(I_w−μ_I)/σ_I²] · ∇_k I_w
                let denom = (vi * vj).sqrt() + 1e-10;
                let cc = num / denom;
                let force_scale = jw_c / denom - cc * iw_c / (vi + 1e-10);
                fz[fi] = (force_scale * gi_z[fi] as f64) as f32;
                fy[fi] = (force_scale * gi_y[fi] as f64) as f32;
                fx[fi] = (force_scale * gi_x[fi] as f64) as f32;
            }
        }
    }
    (fz, fy, fx)
}

/// Compute mean local CC over all voxels (same window radius as CC forces).
pub(crate) fn mean_local_cc(i_w: &[f32], j_w: &[f32], dims: [usize; 3], radius: usize) -> f64 {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let (mut total, mut count) = (0.0_f64, 0u64);

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let (mut si, mut sj, mut nw) = (0.0_f64, 0.0_f64, 0u32);
                for dz in -r..=r {
                    let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            let qi = flat(qz, qy, qx, ny, nx);
                            si += i_w[qi] as f64;
                            sj += j_w[qi] as f64;
                            nw += 1;
                        }
                    }
                }
                let (mi, mj) = (si / nw as f64, sj / nw as f64);

                let (mut num, mut di2, mut dj2) = (0.0_f64, 0.0_f64, 0.0_f64);
                for dz in -r..=r {
                    let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            let qi = flat(qz, qy, qx, ny, nx);
                            let a = i_w[qi] as f64 - mi;
                            let b = j_w[qi] as f64 - mj;
                            num += a * b;
                            di2 += a * a;
                            dj2 += b * b;
                        }
                    }
                }
                let denom = (di2 * dj2).sqrt();
                if denom > 1e-10 {
                    total += num / denom;
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}
