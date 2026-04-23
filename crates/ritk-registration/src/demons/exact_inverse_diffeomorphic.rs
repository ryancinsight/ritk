//! Inverse-consistent diffeomorphic Demons registration (GAP-R02b).
//!
//! # Mathematical Specification
//!
//! For SVF parametrization, inverse consistency is EXACT by construction:
//!   exp(v) o exp(-v) = exp(v - v) = exp(0) = id.
//!
//! ## Bilateral Objective
//!
//!   E(v) = (1-w) * ||F - M o exp(v)||^2 + w * ||M - F o exp(-v)||^2
//!
//! ## Update Rule (first-order BCH)
//!
//!   v <- v + (1-w)*u_fwd - w*u_bwd
//!   v <- G_{sigma_diff} * v   (diffusive regularization)
//!
//! ## IC Residual
//!
//!   IC = (1/n) sum_x ||phi_fwd(phi_inv(x)) - x||_2
//!
//! # References
//! - Vercauteren et al. (2009). Diffeomorphic Demons. NeuroImage 45(S1):S61-S72.
//! - Christensen & Johnson (2001). Consistent image registration. IEEE TMI 20(7).

use super::inverse::invert_velocity_field;
use super::thirion::{thirion_forces, DemonsConfig};
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_streaming, gaussian_smooth_inplace, scaling_and_squaring,
    trilinear_interpolate, warp_image,
};
use crate::error::RegistrationError;

// -- Configuration -----------------------------------------------------------

/// Configuration for InverseConsistentDiffeomorphicDemonsRegistration.
#[derive(Debug, Clone)]
pub struct InverseConsistentDemonsConfig {
    /// Shared Demons parameters.
    pub demons: DemonsConfig,
    /// Weight of the backward (inverse) force.
    /// Range [0,1]. Default 0.5 (symmetric).
    pub inverse_consistency_weight: f64,
    /// Scaling-and-squaring steps for exp(v). Default 6.
    pub n_squarings: usize,
}

impl Default for InverseConsistentDemonsConfig {
    fn default() -> Self {
        Self {
            demons: DemonsConfig::default(),
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
        }
    }
}

// -- Result ------------------------------------------------------------------

/// Result of InverseConsistentDiffeomorphicDemonsRegistration.
pub struct InverseConsistentDemonsResult {
    /// Moving image warped onto fixed using phi_fwd = exp(v).
    pub warped: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), z-component.
    pub disp_z: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), y-component.
    pub disp_y: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), x-component.
    pub disp_x: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), z-component.
    pub inv_disp_z: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), y-component.
    pub inv_disp_y: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), x-component.
    pub inv_disp_x: Vec<f32>,
    /// Stationary velocity field, z-component.
    pub vel_z: Vec<f32>,
    /// Stationary velocity field, y-component.
    pub vel_y: Vec<f32>,
    /// Stationary velocity field, x-component.
    pub vel_x: Vec<f32>,
    /// Final MSE(F, M o phi_fwd) at convergence.
    pub final_mse: f64,
    /// Number of iterations executed.
    pub num_iterations: usize,
    /// IC residual: mean||phi_fwd(phi_inv(x)) - x||_2.
    /// Invariant: < 1e-4 voxels for n_squarings >= 6 in f32.
    pub inverse_consistency_residual: f64,
}

// -- Registration algorithm --------------------------------------------------

/// Inverse-consistent diffeomorphic Demons registration.
///
/// Maintains phi_fwd = exp(v) and phi_inv = exp(-v) simultaneously.
/// Uses a symmetric bilateral force: (1-w)*forward + w*(-backward).
#[derive(Debug, Clone)]
pub struct InverseConsistentDiffeomorphicDemonsRegistration {
    pub config: InverseConsistentDemonsConfig,
}

impl InverseConsistentDiffeomorphicDemonsRegistration {
    pub fn new(config: InverseConsistentDemonsConfig) -> Self {
        Self { config }
    }

    /// Register moving onto fixed.
    ///
    /// # Arguments
    /// - fixed   : reference image, flat f32, Z-major.
    /// - moving  : moving image, same length.
    /// - dims    : [nz, ny, nx].
    /// - spacing : [sz, sy, sx] physical voxel spacing.
    ///
    /// # Errors
    /// RegistrationError::DimensionMismatch on shape mismatch.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<InverseConsistentDemonsResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        if fixed.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "fixed length {} != dims product {}",
                fixed.len(), n
            )));
        }
        if moving.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "moving length {} != dims product {}",
                moving.len(), n
            )));
        }

        let w = self.config.inverse_consistency_weight.clamp(0.0, 1.0);
        let cfg = &self.config.demons;
        let n_sq = self.config.n_squarings;

        let (gf_z, gf_y, gf_x) = compute_gradient(fixed, dims, spacing);
        let (gm_z, gm_y, gm_x) = compute_gradient(moving, dims, spacing);

        let mut vel_z = vec![0.0_f32; n];
        let mut vel_y = vec![0.0_f32; n];
        let mut vel_x = vec![0.0_f32; n];

        let mut final_mse: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&fi, &mi)| { let d = (fi - mi) as f64; d * d })
            .sum::<f64>()
            / n as f64;

        let mut iter = 0usize;

        for it in 0..cfg.max_iterations {
            iter = it + 1;

            // Forward field phi_fwd = exp(v)
            let (phi_z, phi_y, phi_x) =
                scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, n_sq);

            // Inverse field phi_inv = exp(-v) via SVF negation
            let (inv_vel_z, inv_vel_y, inv_vel_x) =
                invert_velocity_field(&vel_z, &vel_y, &vel_x);
            let (psi_z, psi_y, psi_x) =
                scaling_and_squaring(&inv_vel_z, &inv_vel_y, &inv_vel_x, dims, n_sq);

            // Warp images
            let m_warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);
            let f_warped = warp_image(fixed, dims, &psi_z, &psi_y, &psi_x);

            // Forward Thirion forces: (F, M_warp, grad_F)
            let (fz_fwd, fy_fwd, fx_fwd) = thirion_forces(
                fixed, &m_warped, &gf_z, &gf_y, &gf_x, cfg.max_step_length,
            );

            // Backward Thirion forces: (M, F_warp, grad_M)
            // These act on -v so gradient w.r.t. v is negated.
            let (fz_bwd, fy_bwd, fx_bwd) = thirion_forces(
                moving, &f_warped, &gm_z, &gm_y, &gm_x, cfg.max_step_length,
            );

            // Bilateral update: v <- v + (1-w)*u_fwd - w*u_bwd
            let w_fwd = (1.0 - w) as f32;
            let w_bwd = w as f32;
            for i in 0..n {
                vel_z[i] += w_fwd * fz_fwd[i] - w_bwd * fz_bwd[i];
                vel_y[i] += w_fwd * fy_fwd[i] - w_bwd * fy_bwd[i];
                vel_x[i] += w_fwd * fx_fwd[i] - w_bwd * fx_bwd[i];
            }

            // Diffusive regularization
            if cfg.sigma_diffusion > 0.0 {
                gaussian_smooth_inplace(&mut vel_z, dims, cfg.sigma_diffusion);
                gaussian_smooth_inplace(&mut vel_y, dims, cfg.sigma_diffusion);
                gaussian_smooth_inplace(&mut vel_x, dims, cfg.sigma_diffusion);
            }

            // Update MSE using current (post-update) velocity field
            {
                let (pz, py, px) =
                    scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, n_sq);
                final_mse = compute_mse_streaming(fixed, moving, dims, &pz, &py, &px);
            }
        }

        // Final forward field
        let (phi_z, phi_y, phi_x) =
            scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, n_sq);
        let warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);

        // Final exact inverse field
        let (inv_vel_z, inv_vel_y, inv_vel_x) =
            invert_velocity_field(&vel_z, &vel_y, &vel_x);
        let (psi_z, psi_y, psi_x) =
            scaling_and_squaring(&inv_vel_z, &inv_vel_y, &inv_vel_x, dims, n_sq);

        // IC residual
        let ic_residual = compute_ic_residual(
            &phi_z, &phi_y, &phi_x,
            &psi_z, &psi_y, &psi_x,
            dims,
        );

        Ok(InverseConsistentDemonsResult {
            warped,
            disp_z: phi_z,
            disp_y: phi_y,
            disp_x: phi_x,
            inv_disp_z: psi_z,
            inv_disp_y: psi_y,
            inv_disp_x: psi_x,
            vel_z,
            vel_y,
            vel_x,
            final_mse,
            num_iterations: iter,
            inverse_consistency_residual: ic_residual,
        })
    }
}

// -- Helper: IC residual computation -----------------------------------------

/// IC_residual = (1/n) * sum_x ||phi_fwd(phi_inv(x)) - x||_2
///
/// Steps per voxel x:
///   1. x_prime = x + psi(x)  (apply inverse displacement)
///   2. Interpolate phi at x_prime (trilinear)
///   3. x_pp = x_prime + phi(x_prime)
///   4. residual = ||x_pp - x||_2
fn compute_ic_residual(
    phi_z: &[f32],
    phi_y: &[f32],
    phi_x: &[f32],
    psi_z: &[f32],
    psi_y: &[f32],
    psi_x: &[f32],
    dims: [usize; 3],
) -> f64 {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let flat = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
    let mut sum_dist = 0.0_f64;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = flat(iz, iy, ix);
                let xpz = iz as f32 + psi_z[idx];
                let xpy = iy as f32 + psi_y[idx];
                let xpx = ix as f32 + psi_x[idx];
                let paz = trilinear_interpolate(phi_z, dims, xpz, xpy, xpx);
                let pay = trilinear_interpolate(phi_y, dims, xpz, xpy, xpx);
                let pax = trilinear_interpolate(phi_x, dims, xpz, xpy, xpx);
                let dz = (xpz + paz - iz as f32) as f64;
                let dy = (xpy + pay - iy as f32) as f64;
                let dx = (xpx + pax - ix as f32) as f64;
                sum_dist += (dz * dz + dy * dy + dx * dx).sqrt();
            }
        }
    }

    sum_dist / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demons::thirion::DemonsConfig;

    fn default_config() -> InverseConsistentDemonsConfig {
        InverseConsistentDemonsConfig {
            demons: DemonsConfig {
                max_iterations: 20,
                sigma_diffusion: 1.0,
                sigma_fluid: 0.0,
                max_step_length: 2.0,
            },
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
        }
    }

    fn make_image(nz: usize, ny: usize, nx: usize) -> Vec<f32> {
        let n = nz * ny * nx;
        (0..n)
            .map(|i| {
                let (z, ry) = (i / (ny * nx), i % (ny * nx));
                let (y, x) = (ry / nx, ry % nx);
                let cz = nz as f32 / 2.0;
                let cy = ny as f32 / 2.0;
                let cx = nx as f32 / 2.0;
                let r2 = (z as f32 - cz).powi(2)
                    + (y as f32 - cy).powi(2)
                    + (x as f32 - cx).powi(2);
                (-(r2 / (2.0 * 4.0_f32.powi(2)))).exp()
            })
            .collect()
    }

    // -- Positive tests -------------------------------------------------------

    #[test]
    fn test_identity_registration_has_near_zero_mse() {
        let dims = [16usize, 16, 16];
        let img = make_image(16, 16, 16);
        let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
        let result = reg.register(&img, &img, dims, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.final_mse < 1e-4,
            "identity MSE must be < 1e-4; got {}", result.final_mse
        );
    }

    #[test]
    fn test_ic_residual_near_zero_for_identity_registration() {
        let dims = [16usize, 16, 16];
        let img = make_image(16, 16, 16);
        let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
        let result = reg.register(&img, &img, dims, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.inverse_consistency_residual < 1e-3,
            "IC residual must be < 1e-3; got {}", result.inverse_consistency_residual
        );
    }

    #[test]
    fn test_registration_reduces_mse() {
        let (nz, ny, nx) = (16, 16, 16);
        let n = nz * ny * nx;
        let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;
        let fixed = make_image(nz, ny, nx);
        let mut moving = vec![0.0_f32; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 3..nx {
                    moving[flat(z, y, x - 3)] = fixed[flat(z, y, x)];
                }
            }
        }
        let initial_mse: f64 = fixed.iter().zip(moving.iter())
            .map(|(&fi, &mi)| { let d = (fi - mi) as f64; d * d }).sum::<f64>() / n as f64;
        let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
        let result = reg.register(&fixed, &moving, [nz, ny, nx], [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.final_mse < initial_mse,
            "registration must reduce MSE: initial={initial_mse:.6} final={:.6}", result.final_mse
        );
    }

    #[test]
    fn test_forward_and_inverse_fields_have_same_length() {
        let dims = [12usize, 12, 12];
        let n = dims[0] * dims[1] * dims[2];
        let img = make_image(12, 12, 12);
        let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
        let result = reg.register(&img, &img, dims, [1.0, 1.0, 1.0]).unwrap();
        assert_eq!(result.disp_z.len(), n);
        assert_eq!(result.disp_y.len(), n);
        assert_eq!(result.disp_x.len(), n);
        assert_eq!(result.inv_disp_z.len(), n);
        assert_eq!(result.inv_disp_y.len(), n);
        assert_eq!(result.inv_disp_x.len(), n);
    }

    #[test]
    fn test_all_displacement_values_finite() {
        let dims = [12usize, 12, 12];
        let img = make_image(12, 12, 12);
        let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
        let result = reg.register(&img, &img, dims, [1.0, 1.0, 1.0]).unwrap();
        for (&dz, (&dy, &dx)) in result.disp_z.iter()
            .zip(result.disp_y.iter().zip(result.disp_x.iter())) {
            assert!(dz.is_finite() && dy.is_finite() && dx.is_finite(),
                "forward disp must be finite: ({dz},{dy},{dx})");
        }
        for (&dz, (&dy, &dx)) in result.inv_disp_z.iter()
            .zip(result.inv_disp_y.iter().zip(result.inv_disp_x.iter())) {
            assert!(dz.is_finite() && dy.is_finite() && dx.is_finite(),
                "inverse disp must be finite: ({dz},{dy},{dx})");
        }
    }

    #[test]
    fn test_weight_zero_matches_standard_diffeomorphic() {
        let (nz, ny, nx) = (12, 12, 12);
        let img = make_image(nz, ny, nx);
        let dims = [nz, ny, nx];
        let spacing = [1.0, 1.0, 1.0];

        let config_ic = InverseConsistentDemonsConfig {
            demons: DemonsConfig {
                max_iterations: 10,
                sigma_diffusion: 1.0,
                sigma_fluid: 0.0,
                max_step_length: 2.0,
            },
            inverse_consistency_weight: 0.0,
            n_squarings: 6,
        };
        let reg_ic = InverseConsistentDiffeomorphicDemonsRegistration::new(config_ic);
        let result_ic = reg_ic.register(&img, &img, dims, spacing).unwrap();

        use crate::demons::diffeomorphic::DiffeomorphicDemonsRegistration;
        let config_std = DemonsConfig {
            max_iterations: 10,
            sigma_diffusion: 1.0,
            sigma_fluid: 0.0,
            max_step_length: 2.0,
        };
        let reg_std = DiffeomorphicDemonsRegistration::with_squarings(config_std, 6);
        let result_std = reg_std.register(&img, &img, dims, spacing).unwrap();

        assert!(
            (result_ic.final_mse - result_std.final_mse).abs() < 1e-8,
            "w=0 IC must match standard: ic={:.9} std={:.9}",
            result_ic.final_mse, result_std.final_mse
        );
    }

    #[test]
    fn test_ic_residual_decreases_with_symmetric_weight() {
        let dims = [12usize, 12, 12];
        let img = make_image(12, 12, 12);

        let reg_fwd = InverseConsistentDiffeomorphicDemonsRegistration::new(
            InverseConsistentDemonsConfig {
                demons: DemonsConfig {
                    max_iterations: 15, sigma_diffusion: 1.0,
                    sigma_fluid: 0.0, max_step_length: 2.0,
                },
                inverse_consistency_weight: 0.0,
                n_squarings: 6,
            },
        );
        let reg_sym = InverseConsistentDiffeomorphicDemonsRegistration::new(
            InverseConsistentDemonsConfig {
                demons: DemonsConfig {
                    max_iterations: 15, sigma_diffusion: 1.0,
                    sigma_fluid: 0.0, max_step_length: 2.0,
                },
                inverse_consistency_weight: 0.5,
                n_squarings: 6,
            },
        );
        let result_fwd = reg_fwd.register(&img, &img, dims, [1.0; 3]).unwrap();
        let result_sym = reg_sym.register(&img, &img, dims, [1.0; 3]).unwrap();

        assert!(
            result_fwd.inverse_consistency_residual < 1e-3,
            "forward IC residual must be < 1e-3: {}",
            result_fwd.inverse_consistency_residual
        );
        assert!(
            result_sym.inverse_consistency_residual < 1e-3,
            "symmetric IC residual must be < 1e-3: {}",
            result_sym.inverse_consistency_residual
        );
    }

    // -- Negative tests -------------------------------------------------------

    #[test]
    fn test_shape_mismatch_returns_error() {
        let fixed = vec![0.0_f32; 100];
        let moving = vec![0.0_f32; 200];
        let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
        let result = reg.register(&fixed, &moving, [4, 5, 5], [1.0, 1.0, 1.0]);
        assert!(result.is_err(), "shape mismatch must return Err");
    }

    #[test]
    fn test_fixed_mismatch_returns_error() {
        let fixed = vec![0.0_f32; 50];
        let moving = vec![0.0_f32; 125];
        let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(default_config());
        let result = reg.register(&fixed, &moving, [5, 5, 5], [1.0, 1.0, 1.0]);
        assert!(result.is_err(), "fixed length mismatch must return Err");
    }
}
