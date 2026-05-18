//! SyN registration engine: `SyNResult`, `SyNRegistration`, and `register`.
//!
//! # Algorithm
//! Greedy SyN (Avants 2008) with local cross-correlation metric.
//! Both forward (fixed→midpoint) and inverse (moving→midpoint) velocity fields
//! are updated symmetrically each iteration so the midpoint is equidistant from
//! both images.
//!
//! **Per-iteration steps:**
//! 1. φ₁ = exp(v₁), φ₂ = exp(v₂)
//! 2. I_w = warp(F, φ₁), J_w = warp(M, φ₂)
//! 3. u₁ = CC_gradient(I_w, J_w, ∇I_w); normalise max|u₁| ← gradient_step
//! 4. u₂ = CC_gradient(J_w, I_w, ∇J_w); normalise max|u₂| ← gradient_step
//! 5. v₁ ← v₁ + u₁; v₁ ← G_σ ∗ v₁
//! 6. v₂ ← v₂ + u₂; v₂ ← G_σ ∗ v₂
//! 7. Convergence: stop when variance of last `convergence_window` CC values
//!    is below `convergence_threshold`.
//!
//! # Memory discipline
//! All scratch buffers are pre-allocated before the iteration loop.
//! The loop body performs **zero heap allocations**; all `_into` variants
//! write into caller-provided buffers. Total pre-allocation: ~27n f32
//! (6 velocity + 6 displacement + 3 scaling-and-squaring scratch +
//!  2 warped images + 6 gradient + 6 CC forces + 1 smooth scratch = 30n).

use std::collections::VecDeque;

use super::local_cc::{cc_forces_into, mean_local_cc};
use crate::deformable_field_ops::{
    compute_gradient_into, gaussian_smooth_with_scratch, scaling_and_squaring_into, warp_image_into,
};
use crate::error::RegistrationError;

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
    /// - `fixed` — reference image, flat `Vec<f32>` in Z-major order.
    /// - `moving` — moving image, same shape as `fixed`.
    /// - `dims` — image dimensions `[nz, ny, nx]`.
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

        // ── Velocity fields (output) ────────────────────────────────────────
        let mut v1z = vec![0.0_f32; n];
        let mut v1y = vec![0.0_f32; n];
        let mut v1x = vec![0.0_f32; n];
        let mut v2z = vec![0.0_f32; n];
        let mut v2y = vec![0.0_f32; n];
        let mut v2x = vec![0.0_f32; n];

        // ── Pre-allocated scratch buffers (zero alloc inside the loop) ──────
        let mut phi1_z = vec![0.0_f32; n];
        let mut phi1_y = vec![0.0_f32; n];
        let mut phi1_x = vec![0.0_f32; n];
        let mut phi2_z = vec![0.0_f32; n];
        let mut phi2_y = vec![0.0_f32; n];
        let mut phi2_x = vec![0.0_f32; n];
        let mut scratch_ss_z = vec![0.0_f32; n];
        let mut scratch_ss_y = vec![0.0_f32; n];
        let mut scratch_ss_x = vec![0.0_f32; n];
        let mut i_w = vec![0.0_f32; n];
        let mut j_w = vec![0.0_f32; n];
        let mut gi_z = vec![0.0_f32; n];
        let mut gi_y = vec![0.0_f32; n];
        let mut gi_x = vec![0.0_f32; n];
        let mut gj_z = vec![0.0_f32; n];
        let mut gj_y = vec![0.0_f32; n];
        let mut gj_x = vec![0.0_f32; n];
        let mut u1z = vec![0.0_f32; n];
        let mut u1y = vec![0.0_f32; n];
        let mut u1x = vec![0.0_f32; n];
        let mut u2z = vec![0.0_f32; n];
        let mut u2y = vec![0.0_f32; n];
        let mut u2x = vec![0.0_f32; n];
        let mut smooth_tmp = vec![0.0_f32; n];

        let mut cc_history: VecDeque<f64> = VecDeque::new();
        let mut final_cc = 0.0_f64;
        let mut iter = 0usize;
        let r = self.config.cc_window_radius;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // exp(v) via scaling-and-squaring (zero alloc)
            scaling_and_squaring_into(
                &v1z,
                &v1y,
                &v1x,
                dims,
                self.config.n_squarings,
                &mut phi1_z,
                &mut phi1_y,
                &mut phi1_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            scaling_and_squaring_into(
                &v2z,
                &v2y,
                &v2x,
                dims,
                self.config.n_squarings,
                &mut phi2_z,
                &mut phi2_y,
                &mut phi2_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );

            // Warp images (zero alloc)
            warp_image_into(fixed, dims, &phi1_z, &phi1_y, &phi1_x, &mut i_w);
            warp_image_into(moving, dims, &phi2_z, &phi2_y, &phi2_x, &mut j_w);

            // Compute gradients (zero alloc)
            compute_gradient_into(&i_w, dims, spacing, &mut gi_z, &mut gi_y, &mut gi_x);
            compute_gradient_into(&j_w, dims, spacing, &mut gj_z, &mut gj_y, &mut gj_x);

            // CC forces (zero alloc)
            cc_forces_into(
                &i_w, &j_w, &gi_z, &gi_y, &gi_x, dims, r, &mut u1z, &mut u1y, &mut u1x,
            );
            cc_forces_into(
                &j_w, &i_w, &gj_z, &gj_y, &gj_x, dims, r, &mut u2z, &mut u2y, &mut u2x,
            );

            // Normalise u₁ so max|u₁| = gradient_step
            let max_u1 = u1z
                .iter()
                .chain(u1y.iter())
                .chain(u1x.iter())
                .map(|&v| (v as f64).abs())
                .fold(0.0_f64, f64::max);
            if max_u1 > 1e-10 {
                let s = (self.config.gradient_step / max_u1) as f32;
                u1z.iter_mut().for_each(|v| *v *= s);
                u1y.iter_mut().for_each(|v| *v *= s);
                u1x.iter_mut().for_each(|v| *v *= s);
            }

            // Normalise u₂ so max|u₂| = gradient_step
            let max_u2 = u2z
                .iter()
                .chain(u2y.iter())
                .chain(u2x.iter())
                .map(|&v| (v as f64).abs())
                .fold(0.0_f64, f64::max);
            if max_u2 > 1e-10 {
                let s = (self.config.gradient_step / max_u2) as f32;
                u2z.iter_mut().for_each(|v| *v *= s);
                u2y.iter_mut().for_each(|v| *v *= s);
                u2x.iter_mut().for_each(|v| *v *= s);
            }

            // Accumulate forces into velocity fields
            for i in 0..n {
                v1z[i] += u1z[i];
                v1y[i] += u1y[i];
                v1x[i] += u1x[i];
                v2z[i] += u2z[i];
                v2y[i] += u2y[i];
                v2x[i] += u2x[i];
            }

            // Gaussian smooth (zero alloc with shared scratch)
            if self.config.sigma_smooth > 0.0 {
                let sigma = self.config.sigma_smooth;
                gaussian_smooth_with_scratch(&mut v1z, dims, sigma, &mut smooth_tmp);
                gaussian_smooth_with_scratch(&mut v1y, dims, sigma, &mut smooth_tmp);
                gaussian_smooth_with_scratch(&mut v1x, dims, sigma, &mut smooth_tmp);
                gaussian_smooth_with_scratch(&mut v2z, dims, sigma, &mut smooth_tmp);
                gaussian_smooth_with_scratch(&mut v2y, dims, sigma, &mut smooth_tmp);
                gaussian_smooth_with_scratch(&mut v2x, dims, sigma, &mut smooth_tmp);
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

        // Final displacement fields (zero alloc, reuse scratch)
        scaling_and_squaring_into(
            &v1z,
            &v1y,
            &v1x,
            dims,
            self.config.n_squarings,
            &mut phi1_z,
            &mut phi1_y,
            &mut phi1_x,
            &mut scratch_ss_z,
            &mut scratch_ss_y,
            &mut scratch_ss_x,
        );
        scaling_and_squaring_into(
            &v2z,
            &v2y,
            &v2x,
            dims,
            self.config.n_squarings,
            &mut phi2_z,
            &mut phi2_y,
            &mut phi2_x,
            &mut scratch_ss_z,
            &mut scratch_ss_y,
            &mut scratch_ss_x,
        );
        warp_image_into(fixed, dims, &phi1_z, &phi1_y, &phi1_x, &mut i_w);
        warp_image_into(moving, dims, &phi2_z, &phi2_y, &phi2_x, &mut j_w);

        Ok(SyNResult {
            forward_field: (v1z, v1y, v1x),
            inverse_field: (v2z, v2y, v2x),
            warped_fixed: i_w,
            warped_moving: j_w,
            final_cc,
            num_iterations: iter,
        })
    }
}

#[cfg(test)]
mod tests;
