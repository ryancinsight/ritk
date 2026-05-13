//! Local cross-correlation (CC) metric primitives for MultiResSyN.
//!
//! # Algorithm
//! Local CC metric (Avants 2008, eq. 10): for each voxel p with window W of
//! radius `r`:
//!
//!   force_scale(p) = (J_w(p)−μ_J) / √(σ_I²·σ_J²) − CC · (I_w(p)−μ_I) / σ_I²
//!   f_k(p)        = force_scale(p) · ∇_k I_w(p)

use crate::deformable_field_ops::flat;

/// Local CC window statistics at voxel `(iz, iy, ix)` with radius `r`.
///
/// Returns `(mu_i, mu_j, cc_num, var_i, var_j, count)`.
#[inline]
pub(crate) fn window_cc_stats(
    i_w: &[f32],
    j_w: &[f32],
    dims: [usize; 3],
    iz: usize,
    iy: usize,
    ix: usize,
    r: isize,
) -> (f64, f64, f64, f64, f64, u32) {
    let [nz, ny, nx] = dims;
    let (mut si, mut sj, mut cnt) = (0.0_f64, 0.0_f64, 0u32);
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let qi = flat(
                    (iz as isize + dz).max(0).min(nz as isize - 1) as usize,
                    (iy as isize + dy).max(0).min(ny as isize - 1) as usize,
                    (ix as isize + dx).max(0).min(nx as isize - 1) as usize,
                    ny,
                    nx,
                );
                si += i_w[qi] as f64;
                sj += j_w[qi] as f64;
                cnt += 1;
            }
        }
    }
    let (mu_i, mu_j) = (si / cnt as f64, sj / cnt as f64);
    let (mut num, mut vi, mut vj) = (0.0_f64, 0.0_f64, 0.0_f64);
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let qi = flat(
                    (iz as isize + dz).max(0).min(nz as isize - 1) as usize,
                    (iy as isize + dy).max(0).min(ny as isize - 1) as usize,
                    (ix as isize + dx).max(0).min(nx as isize - 1) as usize,
                    ny,
                    nx,
                );
                let di = i_w[qi] as f64 - mu_i;
                let dj = j_w[qi] as f64 - mu_j;
                num += di * dj;
                vi += di * di;
                vj += dj * dj;
            }
        }
    }
    (mu_i, mu_j, num, vi, vj, cnt)
}

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
    let (mut fz, mut fy, mut fx) = (vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]);
    let r = radius as isize;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let (mu_i, mu_j, num, vi, vj, _) =
                    window_cc_stats(i_w, j_w, dims, iz, iy, ix, r);
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

/// Compute mean local CC over all voxels.
pub(crate) fn mean_local_cc(i_w: &[f32], j_w: &[f32], dims: [usize; 3], radius: usize) -> f64 {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let (mut total, mut count) = (0.0_f64, 0u64);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let (_, _, num, di2, dj2, _) = window_cc_stats(i_w, j_w, dims, iz, iy, ix, r);
                let d = (di2 * dj2).sqrt();
                if d > 1e-10 {
                    total += num / d;
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
