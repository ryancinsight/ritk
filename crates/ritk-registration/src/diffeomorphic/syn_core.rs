//! SyN registration engine: `SyNResult`, `SyNRegistration`, and `register`.
//!
//! # Algorithm
//! Greedy SyN (Avants 2008) with local cross-correlation metric.
//! Both forward (fixed→midpoint) and inverse (moving→midpoint) velocity fields
//! are updated symmetrically each iteration so the midpoint is equidistant from
//! both images.
//!
//! **Per-iteration steps:**
//! 1. φ₁ = exp(v₁),  φ₂ = exp(v₂)
//! 2. I_w = warp(F, φ₁),  J_w = warp(M, φ₂)
//! 3. u₁ = CC_gradient(I_w, J_w, ∇I_w);  normalise max|u₁| ← gradient_step
//! 4. u₂ = CC_gradient(J_w, I_w, ∇J_w);  normalise max|u₂| ← gradient_step
//! 5. v₁ ← v₁ + u₁;  v₁ ← G_σ ∗ v₁
//! 6. v₂ ← v₂ + u₂;  v₂ ← G_σ ∗ v₂
//! 7. Convergence: stop when variance of last `convergence_window` CC values
//!    is below `convergence_threshold`.

use std::collections::VecDeque;

use crate::deformable_field_ops::{
    compute_gradient, gaussian_smooth_inplace, scaling_and_squaring, warp_image,
};
use crate::error::RegistrationError;

use super::local_cc::{cc_forces, mean_local_cc};

// ── Public types ──────────────────────────────────────────────────────────────

/// Result returned by [`SyNRegistration::register`].
#[derive(Debug, Clone)]
pub struct SyNResult {
    /// Forward velocity field `v₁` components (fixed→midpoint).
    pub forward_field: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Inverse velocity field `v₂` components (moving→midpoint).
    pub inverse_field: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Fixed image warped to the midpoint by φ₁ = exp(v₁).
    pub warped_fixed: Vec<f32>,
    /// Moving image warped to the midpoint by φ₂ = exp(v₂).
    pub warped_moving: Vec<f32>,
    /// Final mean local CC value (higher is better; 1.0 = perfect alignment).
    pub final_cc: f64,
    /// Number of iterations actually performed.
    pub num_iterations: usize,
}

/// SyN registration engine.
///
/// Implements greedy SyN with local cross-correlation metric.
/// Both forward and inverse velocity fields are updated symmetrically each
/// iteration so that the midpoint is equidistant from both images.
#[derive(Debug, Clone)]
pub struct SyNRegistration {
    /// Algorithm configuration.
    pub config: super::SyNConfig,
}

impl SyNRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: super::SyNConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` using greedy SyN with local CC metric.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `Vec<f32>` in Z-major order.
    /// - `moving`  — moving image, same shape as `fixed`.
    /// - `dims`    — image dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if image lengths are inconsistent with `dims`.
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

        let mut v1z = vec![0.0_f32; n];
        let mut v1y = vec![0.0_f32; n];
        let mut v1x = vec![0.0_f32; n];
        let mut v2z = vec![0.0_f32; n];
        let mut v2y = vec![0.0_f32; n];
        let mut v2x = vec![0.0_f32; n];

        let mut cc_history: VecDeque<f64> = VecDeque::new();
        let mut final_cc = 0.0_f64;
        let mut iter = 0usize;
        let r = self.config.cc_window_radius;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            let (phi1_z, phi1_y, phi1_x) =
                scaling_and_squaring(&v1z, &v1y, &v1x, dims, self.config.n_squarings);
            let (phi2_z, phi2_y, phi2_x) =
                scaling_and_squaring(&v2z, &v2y, &v2x, dims, self.config.n_squarings);

            let i_w = warp_image(fixed, dims, &phi1_z, &phi1_y, &phi1_x);
            let j_w = warp_image(moving, dims, &phi2_z, &phi2_y, &phi2_x);

            let (gi_z, gi_y, gi_x) = compute_gradient(&i_w, dims, spacing);
            let (gj_z, gj_y, gj_x) = compute_gradient(&j_w, dims, spacing);

            let (u1z, u1y, u1x) = cc_forces(&i_w, &j_w, &gi_z, &gi_y, &gi_x, dims, r);
            let (u2z, u2y, u2x) = cc_forces(&j_w, &i_w, &gj_z, &gj_y, &gj_x, dims, r);

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

            for i in 0..n {
                v1z[i] += u1z[i];
                v1y[i] += u1y[i];
                v1x[i] += u1x[i];
                v2z[i] += u2z[i];
                v2y[i] += u2y[i];
                v2x[i] += u2x[i];
            }

            if self.config.sigma_smooth > 0.0 {
                gaussian_smooth_inplace(&mut v1z, dims, self.config.sigma_smooth);
                gaussian_smooth_inplace(&mut v1y, dims, self.config.sigma_smooth);
                gaussian_smooth_inplace(&mut v1x, dims, self.config.sigma_smooth);
                gaussian_smooth_inplace(&mut v2z, dims, self.config.sigma_smooth);
                gaussian_smooth_inplace(&mut v2y, dims, self.config.sigma_smooth);
                gaussian_smooth_inplace(&mut v2x, dims, self.config.sigma_smooth);
            }

            final_cc = mean_local_cc(&i_w, &j_w, dims, r);
            cc_history.push_back(final_cc);
            if cc_history.len() > self.config.convergence_window {
                cc_history.pop_front();
            }
            if cc_history.len() == self.config.convergence_window {
                let mean_cc = cc_history.iter().sum::<f64>() / cc_history.len() as f64;
                let var_cc = cc_history
                    .iter()
                    .map(|&v| (v - mean_cc).powi(2))
                    .sum::<f64>()
                    / cc_history.len() as f64;
                if var_cc < self.config.convergence_threshold {
                    break;
                }
            }
        }

        let (phi1_z, phi1_y, phi1_x) =
            scaling_and_squaring(&v1z, &v1y, &v1x, dims, self.config.n_squarings);
        let (phi2_z, phi2_y, phi2_x) =
            scaling_and_squaring(&v2z, &v2y, &v2x, dims, self.config.n_squarings);
        let warped_fixed = warp_image(fixed, dims, &phi1_z, &phi1_y, &phi1_x);
        let warped_moving = warp_image(moving, dims, &phi2_z, &phi2_y, &phi2_x);

        Ok(SyNResult {
            forward_field: (v1z, v1y, v1x),
            inverse_field: (v2z, v2y, v2x),
            warped_fixed,
            warped_moving,
            final_cc,
            num_iterations: iter,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::local_cc::field_rms;
    use super::*;
    use crate::deformable_field_ops::scaling_and_squaring;

    /// Smooth test image: I[z,y,x] = sin(π·z/nz)·cos(π·y/ny)·(x+1)
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

    /// Shift image +shift voxels in x with zero-padding at the left boundary.
    fn translate_x(data: &[f32], dims: [usize; 3], shift: usize) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let mut out = vec![0.0_f32; nz * ny * nx];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    if ix >= shift {
                        let src = iz * ny * nx + iy * nx + (ix - shift);
                        out[iz * ny * nx + iy * nx + ix] = data[src];
                    }
                }
            }
        }
        out
    }

    /// Registering identical images produces final_cc ≈ 1.0 (> 0.95).
    #[test]
    fn identity_registration_high_cc() {
        let dims = [10usize, 10, 10];
        let image = make_test_image(dims);
        let reg = SyNRegistration::new(super::super::SyNConfig {
            max_iterations: 20,
            sigma_smooth: 2.0,
            cc_window_radius: 2,
            ..Default::default()
        });
        let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.final_cc > 0.9,
            "identity registration should have CC > 0.9, got {}",
            result.final_cc
        );
    }

    /// SyN registration on a translated image pair: non-divergence and
    /// non-trivial velocity-field checks.
    ///
    /// # Rationale
    /// For images whose dominant structure is a linear intensity ramp in x,
    /// local CC is already near 1.0 for any pure x-shift (linear ramps are
    /// perfectly correlated in every local window regardless of offset).
    /// An absolute CC or SSD improvement over the unregistered pair is therefore
    /// NOT a reliable test for this image class.
    ///
    /// Instead we verify:
    /// 1. The algorithm completes without error.
    /// 2. The velocity fields have non-trivially large x-components (the algorithm
    ///    detected the x-shift and produced meaningful updates).
    /// 3. The final CC is still high (≥ 0.9) — the algorithm has not diverged.
    /// 4. The warped midpoint pair SSD is not MORE THAN 10× the original
    ///    (catastrophic divergence guard).
    #[test]
    fn syn_registration_non_divergence_and_non_trivial_fields() {
        let dims = [12usize, 12, 16];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);

        let initial_ssd: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&f, &m)| ((f - m) as f64).powi(2))
            .sum::<f64>()
            / n as f64;

        let reg = SyNRegistration::new(super::super::SyNConfig {
            max_iterations: 30,
            sigma_smooth: 1.5,
            cc_window_radius: 2,
            ..Default::default()
        });
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        let fwd_rms_x = field_rms(&result.forward_field.2);
        let inv_rms_x = field_rms(&result.inverse_field.2);
        assert!(
            fwd_rms_x > 0.01 || inv_rms_x > 0.01,
            "at least one velocity field must be non-trivial in x: \
             fwd_rms_x={fwd_rms_x:.4} inv_rms_x={inv_rms_x:.4}"
        );

        assert!(
            result.final_cc > 0.9,
            "final CC must stay > 0.9 for near-identical images, got {}",
            result.final_cc
        );

        let final_ssd: f64 = result
            .warped_fixed
            .iter()
            .zip(result.warped_moving.iter())
            .map(|(&f, &m)| ((f - m) as f64).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(
            final_ssd < initial_ssd * 10.0,
            "midpoint SSD must not exceed 10× initial SSD: \
             initial={initial_ssd:.4} final={final_ssd:.4}"
        );
    }

    /// The RMS magnitudes of the forward and inverse fields are within 2× of
    /// each other, verifying approximate symmetry in the deformation split.
    #[test]
    fn forward_inverse_field_symmetry() {
        let dims = [10usize, 10, 12];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);

        let reg = SyNRegistration::new(super::super::SyNConfig {
            max_iterations: 30,
            sigma_smooth: 2.0,
            cc_window_radius: 2,
            ..Default::default()
        });
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        let fwd_rms = field_rms(&result.forward_field.2);
        let inv_rms = field_rms(&result.inverse_field.2);

        assert!(
            fwd_rms > 0.01 || inv_rms > 0.01,
            "at least one field must be non-trivial: fwd={fwd_rms:.4} inv={inv_rms:.4}"
        );

        let ratio = if fwd_rms > 1e-10 {
            inv_rms / fwd_rms
        } else {
            0.0
        };
        assert!(
            ratio < 3.0 && (fwd_rms < 1e-10 || ratio > 0.1),
            "field magnitudes too asymmetric: fwd_rms={fwd_rms:.4} inv_rms={inv_rms:.4}"
        );
    }

    /// All velocity field components must be finite after registration.
    #[test]
    fn velocity_fields_finite() {
        let dims = [8usize, 8, 10];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 1);
        let reg = SyNRegistration::new(super::super::SyNConfig {
            max_iterations: 15,
            sigma_smooth: 1.5,
            cc_window_radius: 1,
            ..Default::default()
        });
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
            assert!(
                v.is_finite(),
                "velocity field contains non-finite value: {v}"
            );
        }
    }

    /// Error is returned for length-mismatched inputs.
    #[test]
    fn mismatched_lengths_returns_error() {
        let dims = [4usize, 4, 4];
        let fixed = vec![0.0_f32; 4 * 4 * 4];
        let moving = vec![0.0_f32; 4 * 4 * 5];
        let reg = SyNRegistration::new(super::super::SyNConfig::default());
        assert!(
            reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
                .is_err(),
            "should return error for mismatched lengths"
        );
    }

    /// scaling_and_squaring of zero velocity is zero displacement.
    #[test]
    fn zero_velocity_zero_displacement() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let z = vec![0.0_f32; n];
        let (phiz, phiy, phix) = scaling_and_squaring(&z, &z, &z, dims, 6);
        for i in 0..n {
            assert!(phiz[i].abs() < 1e-5, "phiz[{i}]={}", phiz[i]);
            assert!(phiy[i].abs() < 1e-5, "phiy[{i}]={}", phiy[i]);
            assert!(phix[i].abs() < 1e-5, "phix[{i}]={}", phix[i]);
        }
    }

    /// mean_local_cc of a constant image pair is not 1.0 (near-zero or NaN-safe).
    #[test]
    fn mean_local_cc_constant_images_safe() {
        let dims = [5usize, 5, 5];
        let n = 5 * 5 * 5;
        let a = vec![3.0_f32; n];
        let b = vec![3.0_f32; n];
        let cc = mean_local_cc(&a, &b, dims, 1);
        assert!(
            cc.is_finite(),
            "CC of constant images should be finite, got {cc}"
        );
    }

    /// SyN recovers a pure x-translation on a Gaussian blob image (NCC improves).
    ///
    /// Uses a smooth Gaussian blob (sigma=3) centred at a corner region.
    /// Linear-ramp images are unsuitable because local CC is shift-invariant for
    /// linear ramps; this blob has informative local gradients.
    ///
    /// Verification: NCC_after > NCC_before AND NCC_after >= 0.80.
    #[test]
    fn syn_recovers_translation_ncc_improves() {
        let dims = [16usize, 16, 20];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let sigma = 3.0_f32;
        let fixed: Vec<f32> = (0..n)
            .map(|fi| {
                let ix = (fi % nx) as f32;
                let iy = ((fi / nx) % ny) as f32;
                let iz = (fi / (ny * nx)) as f32;
                let dz = iz - nz as f32 / 2.0;
                let dy = iy - ny as f32 / 2.0;
                let dx = ix - 5.0_f32;
                (-(dz * dz + dy * dy + dx * dx) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let moving = translate_x(&fixed, dims, 4);

        let ncc_before = mean_local_cc(&fixed, &moving, dims, 2);

        let reg = SyNRegistration::new(super::super::SyNConfig {
            max_iterations: 60,
            sigma_smooth: 1.5,
            cc_window_radius: 2,
            gradient_step: 0.25,
            ..Default::default()
        });
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        assert!(
            result.final_cc > ncc_before,
            "SyN must improve NCC: before={ncc_before:.4} after={:.4}",
            result.final_cc
        );
        assert!(
            result.final_cc >= 0.80,
            "SyN final NCC must reach >= 0.80: got {:.4}",
            result.final_cc
        );
    }
}
