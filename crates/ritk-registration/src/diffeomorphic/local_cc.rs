//! Local cross-correlation (CC) primitives for SyN registration.
//!
//! # Algorithm
//! Local CC metric (Avants 2008, eq. 10): for each voxel p with window W of
//! radius `r`:
//!
//!   cc(p) = Σ_{q∈W}(I(q)−μ_I)(J(q)−μ_J)
//!           / √(Σ_{q∈W}(I(q)−μ_I)² · Σ_{q∈W}(J(q)−μ_J)²)
//!
//! Force for velocity field v₁ (fixed→midpoint):
//!
//!   f_k(p) = [(J_w(p)−μ_J)/√(σ_I²σ_J²) − CC·(I_w(p)−μ_I)/σ_I²] · ∇_k I_w(p)
//!
//! Both `cc_forces` and `mean_local_cc` parallelize the outer voxel loop via
//! Rayon, since each voxel's computation is independent (read-only window
//! accesses, no writes to shared memory).

use rayon::prelude::*;

use crate::deformable_field_ops::flat;

// ── Force computation ────────────────────────────────────────────────────────

/// Compute local CC gradient forces for SyN (Avants 2008, eq. 10).
///
/// For each voxel p the local window W = {q : |q−p|_∞ ≤ r} yields:
///   fz[p] = force_scale · gIz[p]
///   where force_scale = (J_w(p)−μ_J)/denom − CC·(I_w(p)−μ_I)/(σ_I²+ε)
///   and   denom = √(σ_I²·σ_J²) + ε
///
/// Parallelized over voxels via Rayon; each voxel's window reads are
/// independent, producing no data race.
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
    let r = radius as isize;

    let forces: Vec<(f32, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|fi| {
            let ix = fi % nx;
            let iy = (fi / nx) % ny;
            let iz = fi / (ny * nx);

            // First pass: window means.
            let mut sum_i = 0.0_f64;
            let mut sum_j = 0.0_f64;
            let mut count = 0usize;
            for dz in -r..=r {
                let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                for dy in -r..=r {
                    let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                    for dx in -r..=r {
                        let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                        let qi = flat(qz, qy, qx, ny, nx);
                        sum_i += i_w[qi] as f64;
                        sum_j += j_w[qi] as f64;
                        count += 1;
                    }
                }
            }
            if count == 0 {
                return (0.0_f32, 0.0_f32, 0.0_f32);
            }
            let mu_i = sum_i / count as f64;
            let mu_j = sum_j / count as f64;

            // Second pass: covariance and variances.
            let mut cc_num = 0.0_f64;
            let mut var_i = 0.0_f64;
            let mut var_j = 0.0_f64;
            for dz in -r..=r {
                let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                for dy in -r..=r {
                    let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                    for dx in -r..=r {
                        let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                        let qi = flat(qz, qy, qx, ny, nx);
                        let di = i_w[qi] as f64 - mu_i;
                        let dj = j_w[qi] as f64 - mu_j;
                        cc_num += di * dj;
                        var_i += di * di;
                        var_j += dj * dj;
                    }
                }
            }
            if var_i < 1e-10 {
                return (0.0_f32, 0.0_f32, 0.0_f32);
            }

            let iw_c = i_w[fi] as f64 - mu_i;
            let jw_c = j_w[fi] as f64 - mu_j;
            let denom = (var_i * var_j).sqrt() + 1e-10;
            let cc = cc_num / denom;
            let force_scale = jw_c / denom - cc * iw_c / (var_i + 1e-10);

            (
                (force_scale * gi_z[fi] as f64) as f32,
                (force_scale * gi_y[fi] as f64) as f32,
                (force_scale * gi_x[fi] as f64) as f32,
            )
        })
        .collect();

    let mut fz = Vec::with_capacity(n);
    let mut fy = Vec::with_capacity(n);
    let mut fx = Vec::with_capacity(n);
    for (z, y, x) in forces {
        fz.push(z);
        fy.push(y);
        fx.push(x);
    }
    (fz, fy, fx)
}

// ── Convergence metric ───────────────────────────────────────────────────────

/// Compute mean local CC over all voxels for convergence monitoring.
///
/// Parallelized over voxels via Rayon; each voxel's reads are independent.
pub(crate) fn mean_local_cc(
    i_w: &[f32],
    j_w: &[f32],
    dims: [usize; 3],
    radius: usize,
) -> f64 {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let r = radius as isize;

    let (total_cc, count) = (0..n)
        .into_par_iter()
        .map(|fi| {
            let ix = fi % nx;
            let iy = (fi / nx) % ny;
            let iz = fi / (ny * nx);

            let mut sum_i = 0.0_f64;
            let mut sum_j = 0.0_f64;
            let mut n_w = 0usize;
            for dz in -r..=r {
                let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                for dy in -r..=r {
                    let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                    for dx in -r..=r {
                        let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                        let qi = flat(qz, qy, qx, ny, nx);
                        sum_i += i_w[qi] as f64;
                        sum_j += j_w[qi] as f64;
                        n_w += 1;
                    }
                }
            }
            let mu_i = sum_i / n_w as f64;
            let mu_j = sum_j / n_w as f64;

            let mut num = 0.0_f64;
            let mut den_i = 0.0_f64;
            let mut den_j = 0.0_f64;
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
                        den_i += di * di;
                        den_j += dj * dj;
                    }
                }
            }
            let denom = (den_i * den_j).sqrt();
            if denom > 1e-10 {
                (num / denom, 1usize)
            } else {
                (0.0_f64, 0usize)
            }
        })
        .reduce(|| (0.0_f64, 0usize), |(a, b), (c, d)| (a + c, b + d));

    if count == 0 {
        0.0
    } else {
        total_cc / count as f64
    }
}

// ── Utility ──────────────────────────────────────────────────────────────────

/// RMS magnitude of a displacement field component.
pub(crate) fn field_rms(v: &[f32]) -> f64 {
    let ss: f64 = v.iter().map(|&x| (x as f64).powi(2)).sum();
    (ss / v.len() as f64).sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_local_cc_constant_images_safe() {
        // Constant images have zero variance → CC should be 0, not NaN.
        let dims = [5usize, 5, 5];
        let n = 5 * 5 * 5;
        let a = vec![3.0_f32; n];
        let b = vec![3.0_f32; n];
        let cc = mean_local_cc(&a, &b, dims, 1);
        assert!(
            cc.is_finite(),
            "CC of constant images must be finite, got {cc}"
        );
        assert!(cc.abs() < 1e-10, "CC of constant images must be 0, got {cc}");
    }

    #[test]
    fn cc_forces_zero_on_constant_images() {
        // var_i < 1e-10 guard must return zero forces for constant I.
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let a = vec![5.0_f32; n];
        let b = vec![3.0_f32; n];
        let gi = vec![1.0_f32; n];
        let (fz, fy, fx) = cc_forces(&a, &b, &gi, &gi, &gi, dims, 1);
        for &v in fz.iter().chain(fy.iter()).chain(fx.iter()) {
            assert!(
                v.abs() < 1e-6,
                "constant-I force must be zero, got {v}"
            );
        }
    }

    #[test]
    fn cc_forces_nonzero_for_shifted_images() {
        // A linearly increasing image vs its shifted version: forces must be
        // non-trivially large (algorithm sees the local intensity gradient).
        let dims = [8usize, 8, 10];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let fixed: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let shifted: Vec<f32> = (0..n).map(|i| (i + nx) as f32).collect();
        let gi_x: Vec<f32> = vec![1.0_f32; n];
        let gi_zero: Vec<f32> = vec![0.0_f32; n];
        let (_, _, fx) = cc_forces(&fixed, &shifted, &gi_zero, &gi_zero, &gi_x, dims, 1);
        let rms_fx: f64 = field_rms(&fx);
        assert!(rms_fx > 0.0, "x-forces must be non-zero for an x-gradient image");
    }

    #[test]
    fn mean_local_cc_identical_images_returns_one() {
        let dims = [6usize, 6, 6];
        let n = 6 * 6 * 6;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cc = mean_local_cc(&a, &a, dims, 1);
        assert!(
            (cc - 1.0).abs() < 1e-8,
            "CC of identical images must be 1.0, got {cc}"
        );
    }
}
