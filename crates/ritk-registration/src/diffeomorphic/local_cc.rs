//! Canonical local cross-correlation (CC) primitives for SyN registration.
//!
//! Single authoritative implementation of the Avants 2008 eq. 10 local CC
//! metric, shared by `syn_core`, `bspline_syn`, and `multires_syn`.
//!
//! # Algorithm
//! Local CC metric (Avants 2008, eq. 10): for each voxel p with window W of
//! radius `r`:
//!
//! cc(p) = Σ_{q∈W}(I(q)−μ_I)(J(q)−μ_J)
//!       / √(Σ_{q∈W}(I(q)−μ_I)² · Σ_{q∈W}(J(q)−μ_J)²)
//!
//! Force for velocity field v₁ (fixed→midpoint):
//!
//! f_k(p) = [(J_w(p)−μ_J)/√(σ_I²σ_J²) − CC·(I_w(p)−μ_I)/σ_I²] · ∇_k I_w(p)
//!
//! All outer voxel loops are parallelized via Rayon, since each voxel's
//! window reads are independent (read-only access, no data races).

use crate::deformable_field_ops::flat;
use rayon::prelude::*;

// ── Window statistics ─────────────────────────────────────────────────────────

/// Local CC window statistics at voxel `(iz, iy, ix)` with radius `r`.
///
/// Returns `(mu_i, mu_j, cc_numerator, var_i, var_j, count)`.
///
/// Two-pass computation: first pass computes window means, second pass
/// computes covariance and variances using the means (numerically stable
/// for the Avants 2008 formula).
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

// ── Force computation ────────────────────────────────────────────────────────

/// Compute local CC gradient forces (Avants 2008, eq. 10).
///
/// For each voxel p the local window W = {q : |q−p|_∞ ≤ r} yields:
///   fz[p] = force_scale · gIz[p]
/// where force_scale = (J_w(p)−μ_J)/denom − CC·(I_w(p)−μ_I)/(σ_I²+ε)
/// and denom = √(σ_I²·σ_J²) + ε
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

            let (mu_i, mu_j, num, vi, vj, _) = window_cc_stats(i_w, j_w, dims, iz, iy, ix, r);

            if vi < 1e-10 {
                return (0.0_f32, 0.0_f32, 0.0_f32);
            }

            let iw_c = i_w[fi] as f64 - mu_i;
            let jw_c = j_w[fi] as f64 - mu_j;

            // Avants 2008, eq. 10 — gradient ascent on local CC.
            // ∂CC/∂v₁_k(x) = [(J_w−μ_J)/√(σ_I²·σ_J²) − CC·(I_w−μ_I)/σ_I²] · ∇_k I_w
            let denom = (vi * vj).sqrt() + 1e-10;
            let cc = num / denom;
            let force_scale = jw_c / denom - cc * iw_c / (vi + 1e-10);

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
pub(crate) fn mean_local_cc(i_w: &[f32], j_w: &[f32], dims: [usize; 3], radius: usize) -> f64 {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let r = radius as isize;

    let (total_cc, count) = (0..n)
        .into_par_iter()
        .map(|fi| {
            let ix = fi % nx;
            let iy = (fi / nx) % ny;
            let iz = fi / (ny * nx);

            let (_, _, num, di2, dj2, _) = window_cc_stats(i_w, j_w, dims, iz, iy, ix, r);

            let d = (di2 * dj2).sqrt();
            if d > 1e-10 {
                (num / d, 1usize)
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
///
/// Used by registration engine test modules for field-quality assertions.
#[allow(dead_code)]
pub(crate) fn field_rms(v: &[f32]) -> f64 {
    let ss: f64 = v.iter().map(|&x| (x as f64).powi(2)).sum();
    (ss / v.len() as f64).sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- Window statistics tests --

    #[test]
    fn window_cc_stats_constant_images() {
        let dims = [5usize, 5, 5];
        let a = vec![3.0_f32; 125];
        let b = vec![7.0_f32; 125];
        let (mu_i, mu_j, num, vi, vj, cnt) = window_cc_stats(&a, &b, dims, 2, 2, 2, 1);
        assert!((mu_i - 3.0).abs() < 1e-10, "mu_i = {mu_i}");
        assert!((mu_j - 7.0).abs() < 1e-10, "mu_j = {mu_j}");
        assert!(num.abs() < 1e-10, "num = {num}");
        assert!(vi.abs() < 1e-10, "var_i = {vi}");
        assert!(vj.abs() < 1e-10, "var_j = {vj}");
        assert!(cnt > 0, "count = {cnt}");
    }

    #[test]
    fn window_cc_stats_identical_non_constant() {
        let dims = [6usize, 6, 6];
        let image: Vec<f32> = (0..216).map(|i| i as f32).collect();
        let (_, _, num, di2, dj2, _) = window_cc_stats(&image, &image, dims, 3, 3, 3, 1);
        // For identical images: num = var_i = var_j, so CC = 1.0
        let d = (di2 * dj2).sqrt();
        assert!(d > 1e-10, "denom = {d}");
        let cc = num / d;
        assert!(
            (cc - 1.0).abs() < 1e-10,
            "CC of identical local patches = {cc}"
        );
    }

    // -- Force computation tests --

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
        assert!(
            cc.abs() < 1e-10,
            "CC of constant images must be 0, got {cc}"
        );
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
            assert!(v.abs() < 1e-6, "constant-I force must be zero, got {v}");
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
        assert!(
            rms_fx > 0.0,
            "x-forces must be non-zero for an x-gradient image"
        );
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

    #[test]
    fn cc_forces_identical_images_bounded() {
        // CC forces on identical images are bounded (at optimum, gradient is small).
        let dims = [6usize, 6, 6];
        let n = 216;
        let image: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let (gz, gy, gx) =
            crate::deformable_field_ops::compute_gradient(&image, dims, [1.0, 1.0, 1.0]);
        let (fz, fy, fx) = cc_forces(&image, &image, &gz, &gy, &gx, dims, 1);
        let rms = |f: &[f32]| -> f64 {
            (f.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt()
        };
        assert!(
            rms(&fz) < 10.0 && rms(&fy) < 10.0 && rms(&fx) < 10.0,
            "CC forces on identical images should be bounded"
        );
    }

    #[test]
    fn mean_local_cc_identical_non_constant_images() {
        let dims = [6usize, 6, 6];
        let image: Vec<f32> = (0..216).map(|i| i as f32).collect();
        let cc = mean_local_cc(&image, &image, dims, 1);
        assert!(
            cc > 0.99,
            "CC of identical non-constant images should be ≈ 1.0, got {cc}"
        );
    }

    #[test]
    fn mean_local_cc_constant_images_is_zero() {
        let dims = [5usize, 5, 5];
        let a = vec![3.0_f32; 125];
        let cc = mean_local_cc(&a, &a, dims, 1);
        assert!(cc.is_finite(), "CC must be finite, got {cc}");
        assert!(
            cc.abs() < 1e-6,
            "CC of constant images should be 0, got {cc}"
        );
    }
}
