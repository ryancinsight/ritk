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

use std::borrow::Cow;
use std::collections::VecDeque;

use crate::deformable_field_ops::{
    compose_fields_into, compute_gradient_into, gaussian_smooth_field_inplace,
    normalize_forces_into, scaling_and_squaring, scaling_and_squaring_into, warp_image,
    warp_image_into, VectorField3D, VectorFieldMut3D, VelocityField,
};
use crate::diffeomorphic::SyNResult;
use crate::error::RegistrationError;

use self::pyramid::{downsample, upsample_field};
use super::local_cc::{cc_forces_into, mean_local_cc};

pub(crate) mod pyramid;

/// Inverse-consistency enforcement policy for SyN velocity field updates.
///
/// Replaces the former `enforce_inverse_consistency: bool` field, eliminating
/// boolean blindness at call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InverseConsistency {
    /// No inverse-consistency enforcement (relaxed update).
    #[default]
    Relaxed,
    /// Enforce inverse consistency via `v ← (v − compose(v₁,v₂)) / 2`.
    Enforced,
}

/// Velocity fields and dimensions carried between resolution levels.
struct PrevLevelState {
    /// Forward velocity field v₁ (fixed→midpoint) from the previous level.
    forward: VelocityField,
    /// Inverse velocity field v₂ (moving→midpoint) from the previous level.
    inverse: VelocityField,
    /// Image dimensions `[nz, ny, nx]` at the previous level.
    dims: [usize; 3],
}
#[cfg(test)]
mod tests;

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
    /// before accumulating into the velocity field. Mirrors the ANTs
    /// `gradientStep` parameter. Default: 0.25.
    pub gradient_step: f64,
    /// Inverse-consistency enforcement policy.
    /// Default: [`InverseConsistency::Relaxed`].
    pub enforce_inverse_consistency: InverseConsistency,
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

        let mut prev: Option<PrevLevelState> = None;
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
            let mut u1z = vec![0.0_f32; ln];
            let mut u1y = vec![0.0_f32; ln];
            let mut u1x = vec![0.0_f32; ln];
            let mut u2z = vec![0.0_f32; ln];
            let mut u2y = vec![0.0_f32; ln];
            let mut u2x = vec![0.0_f32; ln];
            let mut c1z = vec![0.0_f32; ln];
            let mut c1y = vec![0.0_f32; ln];
            let mut c1x = vec![0.0_f32; ln];
            let mut c2z = vec![0.0_f32; ln];
            let mut c2y = vec![0.0_f32; ln];
            let mut c2x = vec![0.0_f32; ln];

            let f_ds: Cow<'_, [f32]> = if factor > 1 {
                Cow::Owned(downsample(fixed, dims, factor))
            } else {
                Cow::Borrowed(fixed)
            };
            let m_ds: Cow<'_, [f32]> = if factor > 1 {
                Cow::Owned(downsample(moving, dims, factor))
            } else {
                Cow::Borrowed(moving)
            };

            let (mut v1z, mut v1y, mut v1x, mut v2z, mut v2y, mut v2x) =
                if let Some(PrevLevelState {
                    forward: fwd,
                    inverse: inv,
                    dims: pd,
                }) = prev.take()
                {
                    (
                        upsample_field(&fwd.z, pd, ld, 0),
                        upsample_field(&fwd.y, pd, ld, 1),
                        upsample_field(&fwd.x, pd, ld, 2),
                        upsample_field(&inv.z, pd, ld, 0),
                        upsample_field(&inv.y, pd, ld, 1),
                        upsample_field(&inv.x, pd, ld, 2),
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

            // ── Per-level scratch (zero alloc inside the inner loop) ─────────
            let mut p1z = vec![0.0_f32; ln];
            let mut p1y = vec![0.0_f32; ln];
            let mut p1x = vec![0.0_f32; ln];
            let mut p2z = vec![0.0_f32; ln];
            let mut p2y = vec![0.0_f32; ln];
            let mut p2x = vec![0.0_f32; ln];
            let mut i_w_buf = vec![0.0_f32; ln];
            let mut j_w_buf = vec![0.0_f32; ln];
            let mut giz = vec![0.0_f32; ln];
            let mut giy = vec![0.0_f32; ln];
            let mut gix = vec![0.0_f32; ln];
            let mut gjz = vec![0.0_f32; ln];
            let mut gjy = vec![0.0_f32; ln];
            let mut gjx = vec![0.0_f32; ln];
            let mut scratch_ss_z = vec![0.0_f32; ln];
            let mut scratch_ss_y = vec![0.0_f32; ln];
            let mut scratch_ss_x = vec![0.0_f32; ln];

            for _ in 0..self.config.iterations_per_level[level] {
                total_iter += 1;
                scaling_and_squaring_into(
                    &v1z,
                    &v1y,
                    &v1x,
                    ld,
                    self.config.n_squarings,
                    &mut p1z,
                    &mut p1y,
                    &mut p1x,
                    &mut scratch_ss_z,
                    &mut scratch_ss_y,
                    &mut scratch_ss_x,
                );
                scaling_and_squaring_into(
                    &v2z,
                    &v2y,
                    &v2x,
                    ld,
                    self.config.n_squarings,
                    &mut p2z,
                    &mut p2y,
                    &mut p2x,
                    &mut scratch_ss_z,
                    &mut scratch_ss_y,
                    &mut scratch_ss_x,
                );
                warp_image_into(&f_ds, ld, &p1z, &p1y, &p1x, &mut i_w_buf);
                warp_image_into(&m_ds, ld, &p2z, &p2y, &p2x, &mut j_w_buf);
                compute_gradient_into(&i_w_buf, ld, ls, &mut giz, &mut giy, &mut gix);
                compute_gradient_into(&j_w_buf, ld, ls, &mut gjz, &mut gjy, &mut gjx);
                cc_forces_into(
                    &i_w_buf, &j_w_buf, &giz, &giy, &gix, ld, r, &mut u1z, &mut u1y, &mut u1x,
                );
                cc_forces_into(
                    &j_w_buf, &i_w_buf, &gjz, &gjy, &gjx, ld, r, &mut u2z, &mut u2y, &mut u2x,
                );

                normalize_forces_into(
                    &mut u1z,
                    &mut u1y,
                    &mut u1x,
                    &mut u2z,
                    &mut u2y,
                    &mut u2x,
                    self.config.gradient_step,
                );

                for i in 0..ln {
                    v1z[i] += u1z[i];
                    v1y[i] += u1y[i];
                    v1x[i] += u1x[i];
                    v2z[i] += u2z[i];
                    v2y[i] += u2y[i];
                    v2x[i] += u2x[i];
                }
                if self.config.sigma_smooth > 0.0 {
                    gaussian_smooth_field_inplace(
                        &mut v1z,
                        &mut v1y,
                        &mut v1x,
                        ld,
                        self.config.sigma_smooth,
                    );
                    gaussian_smooth_field_inplace(
                        &mut v2z,
                        &mut v2y,
                        &mut v2x,
                        ld,
                        self.config.sigma_smooth,
                    );
                }
                if self.config.enforce_inverse_consistency == InverseConsistency::Enforced {
                    compose_fields_into(
                        VectorField3D {
                            z: &v1z,
                            y: &v1y,
                            x: &v1x,
                        },
                        VectorField3D {
                            z: &v2z,
                            y: &v2y,
                            x: &v2x,
                        },
                        ld,
                        VectorFieldMut3D {
                            z: &mut c1z,
                            y: &mut c1y,
                            x: &mut c1x,
                        },
                    );
                    compose_fields_into(
                        VectorField3D {
                            z: &v2z,
                            y: &v2y,
                            x: &v2x,
                        },
                        VectorField3D {
                            z: &v1z,
                            y: &v1y,
                            x: &v1x,
                        },
                        ld,
                        VectorFieldMut3D {
                            z: &mut c2z,
                            y: &mut c2y,
                            x: &mut c2x,
                        },
                    );
                    for i in 0..ln {
                        v1z[i] = (v1z[i] - c1z[i]) * 0.5;
                        v1y[i] = (v1y[i] - c1y[i]) * 0.5;
                        v1x[i] = (v1x[i] - c1x[i]) * 0.5;
                        v2z[i] = (v2z[i] - c2z[i]) * 0.5;
                        v2y[i] = (v2y[i] - c2y[i]) * 0.5;
                        v2x[i] = (v2x[i] - c2x[i]) * 0.5;
                    }
                }
                final_cc = mean_local_cc(&i_w_buf, &j_w_buf, ld, r);
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
            prev = Some(PrevLevelState {
                forward: VelocityField::new(v1z, v1y, v1x),
                inverse: VelocityField::new(v2z, v2y, v2x),
                dims: ld,
            });
        }

        let PrevLevelState {
            forward: fwd,
            inverse: inv,
            ..
        } = prev.expect("at least one resolution level must succeed in multires SyN");
        let p1 = scaling_and_squaring(&fwd.z, &fwd.y, &fwd.x, dims, self.config.n_squarings);
        let p2 = scaling_and_squaring(&inv.z, &inv.y, &inv.x, dims, self.config.n_squarings);
        Ok(SyNResult {
            forward_field: fwd,
            inverse_field: inv,
            warped_fixed: warp_image(fixed, dims, &p1.z, &p1.y, &p1.x),
            warped_moving: warp_image(moving, dims, &p2.z, &p2.y, &p2.x),
            final_cc,
            num_iterations: total_iter,
        })
    }
}
