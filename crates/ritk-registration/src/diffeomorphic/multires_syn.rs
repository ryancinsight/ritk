//! Multi-Resolution Symmetric Normalization (SyN) registration.
//!
//! # Mathematical Specification
//!
//! Multi-resolution SyN executes the SyN optimization at multiple image
//! resolutions in a coarse-to-fine hierarchy. At level `l` ∈ {0, …, L−1}
//! (0 = coarsest):
//!
//! 1. Compute downsample factor `f = 2^(L − l − 1)`
//! 2. Downsample fixed `F` and moving `M` by factor `f` via average pooling
//! 3. If `l > 0`, upsample velocity fields `v₁, v₂` from level `l−1` to
//!    current resolution via trilinear interpolation with displacement scaling
//! 4. Run SyN iterations at this level (max = `iterations_per_level[l]`)
//! 5. Optionally enforce inverse consistency: `v₁ ← (v₁ − compose(v₁,v₂))/2`
//!
//! ## Downsampling
//!
//! Average pooling with stride `f` in each dimension:
//!   `out[oz,oy,ox] = mean(in[oz·f .. min(oz·f+f, D), ...])`
//! Output dimension per axis: `new_d = max(1, d / f)`.
//!
//! ## Upsampling
//!
//! Trilinear interpolation to target dimensions. Displacement component `d` is
//! scaled by `new_dims[d] / old_dims[d]` to preserve physical displacement
//! magnitude across voxel-size changes.
//!
//! ## Local CC Gradient (Avants 2008, eq. 10)
//!
//!   `f_z[p] = −2 · cc_num / (var_I · var_J + ε) · (J_w[p] − μ_J) · ∇I_z[p]`
//!
//! where sums are over a local window of radius `r` centred at `p`.
//!
//! ## Inverse Consistency Enforcement
//!
//! After each iteration (when enabled), both velocity fields are nudged toward
//! mutual inverse consistency:
//!   `c₁ = compose(v₁, v₂);  c₂ = compose(v₂, v₁)`
//!   `v₁ ← (v₁ − c₁) / 2;   v₂ ← (v₂ − c₂) / 2`
//! Both corrections are computed from the pre-update fields to maintain symmetry.
//!
//! # References
//! - Avants, B. B., Epstein, C. L., Grossman, M. & Gee, J. C. (2008).
//!   Symmetric diffeomorphic image registration with cross-correlation.
//!   *Medical Image Analysis* 12(1):26–41.

use std::collections::VecDeque;

use crate::deformable_field_ops::{
    compose_fields, compute_gradient, flat, gaussian_smooth_inplace, scaling_and_squaring,
    trilinear_interpolate, warp_image,
};
use crate::diffeomorphic::SyNResult;
use crate::error::RegistrationError;

/// Configuration for multi-resolution SyN registration.
#[derive(Debug, Clone)]
pub struct MultiResSyNConfig {
    /// Number of resolution levels (e.g., 3 → factors 4×, 2×, 1×).
    pub num_levels: usize,
    /// Maximum iterations at each level. Length must equal `num_levels`.
    pub iterations_per_level: Vec<usize>,
    /// Gaussian regularisation σ (voxels) applied to velocity fields.
    pub sigma_smooth: f64,
    /// Stop when CC variance over the convergence window falls below this.
    pub convergence_threshold: f64,
    /// Number of recent CC values for convergence checking.
    pub convergence_window: usize,
    /// Number of scaling-and-squaring steps for exp(v).
    pub n_squarings: usize,
    /// Radius of local CC window (voxels).
    pub cc_window_radius: usize,
    /// Maximum per-step displacement (voxels) used to normalise the CC gradient
    /// before accumulating into the velocity field.  Mirrors the ANTs
    /// `gradientStep` parameter.  Default: 0.25.
    pub gradient_step: f64,
    /// Enforce inverse consistency via `v ← (v − compose(v₁,v₂)) / 2`.
    pub enforce_inverse_consistency: bool,
}

/// Multi-resolution SyN registration engine.
#[derive(Debug, Clone)]
pub struct MultiResSyNRegistration {
    pub config: MultiResSyNConfig,
}

impl MultiResSyNRegistration {
    pub fn new(config: MultiResSyNConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` using multi-resolution SyN with local CC.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<SyNResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        if fixed.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "fixed length {} != dims product {}",
                fixed.len(),
                n
            )));
        }
        if moving.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "moving length {} != dims product {}",
                moving.len(),
                n
            )));
        }
        if self.config.iterations_per_level.len() != self.config.num_levels {
            return Err(RegistrationError::InvalidConfiguration(format!(
                "iterations_per_level length {} != num_levels {}",
                self.config.iterations_per_level.len(),
                self.config.num_levels
            )));
        }
        if self.config.num_levels == 0 {
            return Err(RegistrationError::InvalidConfiguration(
                "num_levels must be >= 1".into(),
            ));
        }

        let mut prev: Option<(
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            [usize; 3],
        )> = None;
        let mut final_cc = 0.0_f64;
        let mut total_iter = 0usize;

        for level in 0..self.config.num_levels {
            let factor = 1usize << (self.config.num_levels - level - 1);
            let ld = [
                (nz / factor).max(1),
                (ny / factor).max(1),
                (nx / factor).max(1),
            ];
            let ls = [
                spacing[0] * factor as f64,
                spacing[1] * factor as f64,
                spacing[2] * factor as f64,
            ];
            let ln = ld[0] * ld[1] * ld[2];

            let f_ds = if factor > 1 {
                downsample(fixed, dims, factor)
            } else {
                fixed.to_vec()
            };
            let m_ds = if factor > 1 {
                downsample(moving, dims, factor)
            } else {
                moving.to_vec()
            };

            let (mut v1z, mut v1y, mut v1x, mut v2z, mut v2y, mut v2x) =
                if let Some((pz, py, px, qz, qy, qx, pd)) = prev.take() {
                    (
                        upsample_field(&pz, pd, ld, 0),
                        upsample_field(&py, pd, ld, 1),
                        upsample_field(&px, pd, ld, 2),
                        upsample_field(&qz, pd, ld, 0),
                        upsample_field(&qy, pd, ld, 1),
                        upsample_field(&qx, pd, ld, 2),
                    )
                } else {
                    (
                        vec![0.0_f32; ln],
                        vec![0.0_f32; ln],
                        vec![0.0_f32; ln],
                        vec![0.0_f32; ln],
                        vec![0.0_f32; ln],
                        vec![0.0_f32; ln],
                    )
                };

            let mut cc_hist: VecDeque<f64> = VecDeque::new();
            let r = self.config.cc_window_radius;

            for _ in 0..self.config.iterations_per_level[level] {
                total_iter += 1;
                let (p1z, p1y, p1x) =
                    scaling_and_squaring(&v1z, &v1y, &v1x, ld, self.config.n_squarings);
                let (p2z, p2y, p2x) =
                    scaling_and_squaring(&v2z, &v2y, &v2x, ld, self.config.n_squarings);
                let i_w = warp_image(&f_ds, ld, &p1z, &p1y, &p1x);
                let j_w = warp_image(&m_ds, ld, &p2z, &p2y, &p2x);
                let (giz, giy, gix) = compute_gradient(&i_w, ld, ls);
                let (gjz, gjy, gjx) = compute_gradient(&j_w, ld, ls);
                let (u1z, u1y, u1x) = cc_forces(&i_w, &j_w, &giz, &giy, &gix, ld, r);
                let (u2z, u2y, u2x) = cc_forces(&j_w, &i_w, &gjz, &gjy, &gjx, ld, r);

                // Normalise per-step displacement to `gradient_step` voxels (inf-norm).
                let max_u1 = u1z
                    .iter()
                    .chain(u1y.iter())
                    .chain(u1x.iter())
                    .map(|&v| (v as f64).abs())
                    .fold(0.0_f64, f64::max);
                let (mut u1z, mut u1y, mut u1x) = (u1z, u1y, u1x);
                if max_u1 > 1e-10 {
                    let s = (self.config.gradient_step / max_u1) as f32;
                    u1z.iter_mut().for_each(|v| *v *= s);
                    u1y.iter_mut().for_each(|v| *v *= s);
                    u1x.iter_mut().for_each(|v| *v *= s);
                }
                let max_u2 = u2z
                    .iter()
                    .chain(u2y.iter())
                    .chain(u2x.iter())
                    .map(|&v| (v as f64).abs())
                    .fold(0.0_f64, f64::max);
                let (mut u2z, mut u2y, mut u2x) = (u2z, u2y, u2x);
                if max_u2 > 1e-10 {
                    let s = (self.config.gradient_step / max_u2) as f32;
                    u2z.iter_mut().for_each(|v| *v *= s);
                    u2y.iter_mut().for_each(|v| *v *= s);
                    u2x.iter_mut().for_each(|v| *v *= s);
                }

                for i in 0..ln {
                    v1z[i] += u1z[i];
                    v1y[i] += u1y[i];
                    v1x[i] += u1x[i];
                    v2z[i] += u2z[i];
                    v2y[i] += u2y[i];
                    v2x[i] += u2x[i];
                }
                if self.config.sigma_smooth > 0.0 {
                    gaussian_smooth_inplace(&mut v1z, ld, self.config.sigma_smooth);
                    gaussian_smooth_inplace(&mut v1y, ld, self.config.sigma_smooth);
                    gaussian_smooth_inplace(&mut v1x, ld, self.config.sigma_smooth);
                    gaussian_smooth_inplace(&mut v2z, ld, self.config.sigma_smooth);
                    gaussian_smooth_inplace(&mut v2y, ld, self.config.sigma_smooth);
                    gaussian_smooth_inplace(&mut v2x, ld, self.config.sigma_smooth);
                }
                if self.config.enforce_inverse_consistency {
                    let (c1z, c1y, c1x) = compose_fields(&v1z, &v1y, &v1x, &v2z, &v2y, &v2x, ld);
                    let (c2z, c2y, c2x) = compose_fields(&v2z, &v2y, &v2x, &v1z, &v1y, &v1x, ld);
                    for i in 0..ln {
                        v1z[i] = (v1z[i] - c1z[i]) * 0.5;
                        v1y[i] = (v1y[i] - c1y[i]) * 0.5;
                        v1x[i] = (v1x[i] - c1x[i]) * 0.5;
                        v2z[i] = (v2z[i] - c2z[i]) * 0.5;
                        v2y[i] = (v2y[i] - c2y[i]) * 0.5;
                        v2x[i] = (v2x[i] - c2x[i]) * 0.5;
                    }
                }
                final_cc = mean_local_cc(&i_w, &j_w, ld, r);
                cc_hist.push_back(final_cc);
                if cc_hist.len() > self.config.convergence_window {
                    cc_hist.pop_front();
                }
                if cc_hist.len() == self.config.convergence_window {
                    let mu = cc_hist.iter().sum::<f64>() / cc_hist.len() as f64;
                    let var = cc_hist.iter().map(|&v| (v - mu).powi(2)).sum::<f64>()
                        / cc_hist.len() as f64;
                    if var < self.config.convergence_threshold {
                        break;
                    }
                }
            }
            prev = Some((v1z, v1y, v1x, v2z, v2y, v2x, ld));
        }

        let (v1z, v1y, v1x, v2z, v2y, v2x, _) = prev.unwrap();
        let (p1z, p1y, p1x) = scaling_and_squaring(&v1z, &v1y, &v1x, dims, self.config.n_squarings);
        let (p2z, p2y, p2x) = scaling_and_squaring(&v2z, &v2y, &v2x, dims, self.config.n_squarings);
        Ok(SyNResult {
            forward_field: (v1z, v1y, v1x),
            inverse_field: (v2z, v2y, v2x),
            warped_fixed: warp_image(fixed, dims, &p1z, &p1y, &p1x),
            warped_moving: warp_image(moving, dims, &p2z, &p2y, &p2x),
            final_cc,
            num_iterations: total_iter,
        })
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Downsample a 3-D image by factor `f` via average pooling with stride `f`.
fn downsample(image: &[f32], dims: [usize; 3], factor: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let od = [
        (nz / factor).max(1),
        (ny / factor).max(1),
        (nx / factor).max(1),
    ];
    let mut out = vec![0.0_f32; od[0] * od[1] * od[2]];
    for oz in 0..od[0] {
        for oy in 0..od[1] {
            for ox in 0..od[2] {
                let (mut sum, mut cnt) = (0.0_f64, 0u32);
                for dz in 0..factor {
                    let iz = oz * factor + dz;
                    if iz >= nz {
                        break;
                    }
                    for dy in 0..factor {
                        let iy = oy * factor + dy;
                        if iy >= ny {
                            break;
                        }
                        for dx in 0..factor {
                            let ix = ox * factor + dx;
                            if ix >= nx {
                                break;
                            }
                            sum += image[flat(iz, iy, ix, ny, nx)] as f64;
                            cnt += 1;
                        }
                    }
                }
                out[flat(oz, oy, ox, od[1], od[2])] = (sum / cnt as f64) as f32;
            }
        }
    }
    out
}

/// Upsample a single displacement-field component via trilinear interpolation.
/// `component` ∈ {0,1,2} selects the axis whose ratio scales displacement values.
fn upsample_field(field: &[f32], old: [usize; 3], new: [usize; 3], component: usize) -> Vec<f32> {
    let nn = new[0] * new[1] * new[2];
    let scale = if old[component] > 0 {
        new[component] as f32 / old[component] as f32
    } else {
        1.0
    };
    let mut out = vec![0.0_f32; nn];
    let map = |n_new: usize, n_old: usize, idx: usize| -> f32 {
        if n_new > 1 && n_old > 1 {
            idx as f32 * (n_old - 1) as f32 / (n_new - 1) as f32
        } else {
            0.0
        }
    };
    for iz in 0..new[0] {
        let oz = map(new[0], old[0], iz);
        for iy in 0..new[1] {
            let oy = map(new[1], old[1], iy);
            for ix in 0..new[2] {
                let ox = map(new[2], old[2], ix);
                out[flat(iz, iy, ix, new[1], new[2])] =
                    trilinear_interpolate(field, old, oz, oy, ox) * scale;
            }
        }
    }
    out
}

// ── CC metric primitives ──────────────────────────────────────────────────────

/// Local CC window statistics at voxel `(iz, iy, ix)` with radius `r`.
/// Returns `(mu_i, mu_j, cc_num, var_i, var_j, count)`.
#[inline]
fn window_cc_stats(
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

/// Local CC gradient forces (Avants 2008, eq. 10).
fn cc_forces(
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
                let (mu_i, mu_j, num, vi, vj, _) = window_cc_stats(i_w, j_w, dims, iz, iy, ix, r);
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

/// Mean local CC over all voxels.
fn mean_local_cc(i_w: &[f32], j_w: &[f32], dims: [usize; 3], radius: usize) -> f64 {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        cc_forces, downsample, mean_local_cc, upsample_field, MultiResSyNConfig,
        MultiResSyNRegistration,
    };

    /// `I[z,y,x] = sin(π·z/nz) · cos(π·y/ny) · (x + 1)`.
    /// Analytically non-trivial gradients in all three axes.
    fn make_test_image(dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        (0..nz * ny * nx)
            .map(|fi| {
                let ix = fi % nx;
                let iy = (fi / nx) % ny;
                let iz = fi / (ny * nx);
                let sz = std::f32::consts::PI * iz as f32 / nz as f32;
                let sy = std::f32::consts::PI * iy as f32 / ny as f32;
                sz.sin() * sy.cos() * (ix as f32 + 1.0)
            })
            .collect()
    }

    fn translate_x(data: &[f32], dims: [usize; 3], shift: usize) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let mut out = vec![0.0_f32; nz * ny * nx];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in shift..nx {
                    out[iz * ny * nx + iy * nx + ix] = data[iz * ny * nx + iy * nx + (ix - shift)];
                }
            }
        }
        out
    }

    fn make_config(num_levels: usize, iters: Vec<usize>, ic: bool) -> MultiResSyNConfig {
        MultiResSyNConfig {
            num_levels,
            iterations_per_level: iters,
            sigma_smooth: 2.0,
            convergence_threshold: 1e-7,
            convergence_window: 10,
            n_squarings: 6,
            cc_window_radius: 2,
            gradient_step: 0.25,
            enforce_inverse_consistency: ic,
        }
    }

    // ── Downsample / upsample ─────────────────────────────────────────────────

    /// Average-pool of constant field preserves value (analytical: mean(c) = c).
    #[test]
    fn downsample_constant_preserves_value() {
        let dims = [8, 8, 8];
        let image = vec![7.0_f32; 8 * 8 * 8];
        let ds = downsample(&image, dims, 2);
        assert_eq!(ds.len(), 4 * 4 * 4);
        for &v in &ds {
            assert!((v - 7.0).abs() < 1e-6, "expected 7.0, got {v}");
        }
    }

    /// Upsample constant field with component=0: output = value × (new_nz / old_nz).
    /// Analytical: 3.0 × (8/4) = 6.0.
    #[test]
    fn upsample_constant_field_scales_correctly() {
        let old = [4, 4, 4];
        let new = [8, 8, 8];
        let field = vec![3.0_f32; 4 * 4 * 4];
        let up = upsample_field(&field, old, new, 0);
        assert_eq!(up.len(), 8 * 8 * 8);
        for &v in &up {
            assert!((v - 6.0).abs() < 1e-4, "expected 6.0, got {v}");
        }
    }

    // ── Registration ──────────────────────────────────────────────────────────

    /// Identical images → CC > 0.9 (analytical: perfect correlation).
    #[test]
    fn identity_registration_high_cc() {
        let dims = [10, 10, 10];
        let image = make_test_image(dims);
        let reg = MultiResSyNRegistration::new(make_config(2, vec![10, 10], false));
        let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.final_cc > 0.9,
            "identity CC should be > 0.9, got {}",
            result.final_cc
        );
    }

    /// Single-level (num_levels=1) equivalent to standard SyN.
    #[test]
    fn single_level_equivalent_to_syn() {
        let dims = [8, 8, 8];
        let image = make_test_image(dims);
        let reg = MultiResSyNRegistration::new(make_config(1, vec![15], false));
        let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.final_cc > 0.9,
            "single-level CC should be > 0.9, got {}",
            result.final_cc
        );
    }

    /// Multi-res on translated pair: non-divergence, non-trivial fields.
    #[test]
    fn multires_registration_non_divergence() {
        let dims = [12, 12, 16];
        let n = dims[0] * dims[1] * dims[2];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);
        let reg = MultiResSyNRegistration::new(make_config(2, vec![10, 15], false));
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        let rms = |f: &[f32]| -> f64 {
            (f.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt()
        };
        let fwd_x = rms(&result.forward_field.2);
        let inv_x = rms(&result.inverse_field.2);
        assert!(
            fwd_x > 0.001 || inv_x > 0.001,
            "x-field must be non-trivial: fwd={fwd_x:.6} inv={inv_x:.6}"
        );
        assert!(
            result.final_cc > 0.8,
            "CC must be > 0.8, got {}",
            result.final_cc
        );
        for &v in result
            .forward_field
            .0
            .iter()
            .chain(result.forward_field.1.iter())
            .chain(result.forward_field.2.iter())
            .chain(result.inverse_field.0.iter())
            .chain(result.inverse_field.1.iter())
            .chain(result.inverse_field.2.iter())
        {
            assert!(v.is_finite(), "non-finite value: {v}");
        }
    }

    /// Inverse consistency produces finite fields with high CC.
    #[test]
    fn inverse_consistency_produces_finite_fields() {
        let dims = [10, 10, 12];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);
        let reg = MultiResSyNRegistration::new(make_config(2, vec![8, 12], true));
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();
        for &v in result
            .forward_field
            .0
            .iter()
            .chain(result.forward_field.1.iter())
            .chain(result.forward_field.2.iter())
            .chain(result.inverse_field.0.iter())
            .chain(result.inverse_field.1.iter())
            .chain(result.inverse_field.2.iter())
        {
            assert!(v.is_finite(), "IC field non-finite: {v}");
        }
        assert!(
            result.final_cc > 0.8,
            "IC CC should be > 0.8, got {}",
            result.final_cc
        );
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn mismatched_fixed_length_returns_error() {
        let dims = [4, 4, 4];
        let reg = MultiResSyNRegistration::new(make_config(1, vec![5], false));
        let err = reg.register(&vec![0.0_f32; 80], &vec![0.0_f32; 64], dims, [1.0; 3]);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("fixed length"));
    }

    #[test]
    fn mismatched_moving_length_returns_error() {
        let dims = [4, 4, 4];
        let reg = MultiResSyNRegistration::new(make_config(1, vec![5], false));
        let err = reg.register(&vec![0.0_f32; 64], &vec![0.0_f32; 80], dims, [1.0; 3]);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("moving length"));
    }

    #[test]
    fn invalid_iterations_per_level_returns_error() {
        let dims = [4, 4, 4];
        let img = vec![0.0_f32; 64];
        let reg = MultiResSyNRegistration::new(make_config(3, vec![5, 5], false));
        let err = reg.register(&img, &img, dims, [1.0; 3]);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("iterations_per_level"));
    }

    #[test]
    fn zero_levels_returns_error() {
        let dims = [4, 4, 4];
        let img = vec![0.0_f32; 64];
        let reg = MultiResSyNRegistration::new(make_config(0, vec![], false));
        assert!(reg.register(&img, &img, dims, [1.0; 3]).is_err());
    }

    // ── CC primitive tests ────────────────────────────────────────────────────

    /// Identical non-constant images → CC ≈ 1.0 (analytical: perfect correlation).
    #[test]
    fn mean_local_cc_identical_images() {
        let dims = [6, 6, 6];
        let image = make_test_image(dims);
        let cc = mean_local_cc(&image, &image, dims, 1);
        assert!(
            cc > 0.99,
            "CC of identical images should be ≈ 1.0, got {cc}"
        );
    }

    /// Constant images → CC = 0 (zero variance, degenerate).
    #[test]
    fn mean_local_cc_constant_images_is_zero() {
        let dims = [5, 5, 5];
        let a = vec![3.0_f32; 125];
        let cc = mean_local_cc(&a, &a, dims, 1);
        assert!(cc.is_finite(), "CC must be finite, got {cc}");
        assert!(
            cc.abs() < 1e-6,
            "CC of constant images should be 0, got {cc}"
        );
    }

    /// CC forces on identical images are bounded (at optimum, gradient is small).
    #[test]
    fn cc_forces_identical_images_bounded() {
        let dims = [6, 6, 6];
        let n = 216;
        let image = make_test_image(dims);
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
}
