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
//! write into caller-provided buffers. Total pre-allocation: ~30n f32
//! (6 velocity + 6 displacement + 3 scaling-and-squaring scratch +
//!  2 warped images + 6 gradient + 6 CC forces + 1 smooth scratch = 30n).

mod buffers;

use std::collections::VecDeque;

use super::local_cc::{cc_forces_into, mean_local_cc};
use crate::deformable_field_ops::{
    compute_gradient_into, gaussian_smooth_field_inplace_with_scratch, normalize_forces_into,
    scaling_and_squaring_into, warp_image_into, VelocityField,
};
use crate::error::RegistrationError;
use buffers::SyNBuffers;

// ── Public types ──────────────────────────────────────────────────────────────

/// Result returned by [`SyNRegistration::register`].
#[derive(Debug, Clone)]
pub struct SyNResult {
    /// Forward velocity field `v₁` components (fixed→midpoint), in (z, y, x) order.
    pub forward_field: VelocityField,
    /// Inverse velocity field `v₂` components (moving→midpoint), in (z, y, x) order.
    pub inverse_field: VelocityField,
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

        let mut buf = SyNBuffers::new(n);

        let mut cc_history: VecDeque<f64> = VecDeque::new();
        let mut final_cc = 0.0_f64;
        let mut iter = 0usize;
        let r = self.config.cc_window_radius;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // exp(v) via scaling-and-squaring (zero alloc)
            scaling_and_squaring_into(
                &buf.v1z,
                &buf.v1y,
                &buf.v1x,
                dims.into(),
                self.config.n_squarings,
                &mut buf.phi1_z,
                &mut buf.phi1_y,
                &mut buf.phi1_x,
                &mut buf.scratch_ss_z,
                &mut buf.scratch_ss_y,
                &mut buf.scratch_ss_x,
            );
            scaling_and_squaring_into(
                &buf.v2z,
                &buf.v2y,
                &buf.v2x,
                dims.into(),
                self.config.n_squarings,
                &mut buf.phi2_z,
                &mut buf.phi2_y,
                &mut buf.phi2_x,
                &mut buf.scratch_ss_z,
                &mut buf.scratch_ss_y,
                &mut buf.scratch_ss_x,
            );

            // Warp images (zero alloc)
            warp_image_into(
                fixed,
                dims.into(),
                &buf.phi1_z,
                &buf.phi1_y,
                &buf.phi1_x,
                &mut buf.i_w,
            );
            warp_image_into(
                moving,
                dims.into(),
                &buf.phi2_z,
                &buf.phi2_y,
                &buf.phi2_x,
                &mut buf.j_w,
            );

            // Compute gradients (zero alloc)
            compute_gradient_into(
                &buf.i_w,
                dims.into(),
                spacing,
                &mut buf.gi_z,
                &mut buf.gi_y,
                &mut buf.gi_x,
            );
            compute_gradient_into(
                &buf.j_w,
                dims.into(),
                spacing,
                &mut buf.gj_z,
                &mut buf.gj_y,
                &mut buf.gj_x,
            );

            // CC forces (zero alloc)
            cc_forces_into(
                &buf.i_w,
                &buf.j_w,
                &buf.gi_z,
                &buf.gi_y,
                &buf.gi_x,
                dims,
                r,
                &mut buf.u1z,
                &mut buf.u1y,
                &mut buf.u1x,
            );
            cc_forces_into(
                &buf.j_w,
                &buf.i_w,
                &buf.gj_z,
                &buf.gj_y,
                &buf.gj_x,
                dims,
                r,
                &mut buf.u2z,
                &mut buf.u2y,
                &mut buf.u2x,
            );

            // Normalise forces so max|u₁| = max|u₂| = gradient_step
            normalize_forces_into(
                &mut buf.u1z,
                &mut buf.u1y,
                &mut buf.u1x,
                &mut buf.u2z,
                &mut buf.u2y,
                &mut buf.u2x,
                self.config.gradient_step,
            );

            // Accumulate forces into velocity fields
            for i in 0..n {
                buf.v1z[i] += buf.u1z[i];
                buf.v1y[i] += buf.u1y[i];
                buf.v1x[i] += buf.u1x[i];
                buf.v2z[i] += buf.u2z[i];
                buf.v2y[i] += buf.u2y[i];
                buf.v2x[i] += buf.u2x[i];
            }

            // Gaussian smooth (zero alloc with shared scratch)
            if self.config.sigma_smooth > 0.0 {
                let sigma = self.config.sigma_smooth;
                gaussian_smooth_field_inplace_with_scratch(
                    &mut buf.v1z,
                    &mut buf.v1y,
                    &mut buf.v1x,
                    dims.into(),
                    sigma,
                    &mut buf.smooth_tmp,
                );
                gaussian_smooth_field_inplace_with_scratch(
                    &mut buf.v2z,
                    &mut buf.v2y,
                    &mut buf.v2x,
                    dims.into(),
                    sigma,
                    &mut buf.smooth_tmp,
                );
            }

            final_cc = mean_local_cc(&buf.i_w, &buf.j_w, dims, r);
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
            &buf.v1z,
            &buf.v1y,
            &buf.v1x,
            dims.into(),
            self.config.n_squarings,
            &mut buf.phi1_z,
            &mut buf.phi1_y,
            &mut buf.phi1_x,
            &mut buf.scratch_ss_z,
            &mut buf.scratch_ss_y,
            &mut buf.scratch_ss_x,
        );
        scaling_and_squaring_into(
            &buf.v2z,
            &buf.v2y,
            &buf.v2x,
            dims.into(),
            self.config.n_squarings,
            &mut buf.phi2_z,
            &mut buf.phi2_y,
            &mut buf.phi2_x,
            &mut buf.scratch_ss_z,
            &mut buf.scratch_ss_y,
            &mut buf.scratch_ss_x,
        );
        warp_image_into(
            fixed,
            dims.into(),
            &buf.phi1_z,
            &buf.phi1_y,
            &buf.phi1_x,
            &mut buf.i_w,
        );
        warp_image_into(
            moving,
            dims.into(),
            &buf.phi2_z,
            &buf.phi2_y,
            &buf.phi2_x,
            &mut buf.j_w,
        );

        Ok(SyNResult {
            forward_field: VelocityField::new(buf.v1z, buf.v1y, buf.v1x),
            inverse_field: VelocityField::new(buf.v2z, buf.v2y, buf.v2x),
            warped_fixed: buf.i_w,
            warped_moving: buf.j_w,
            final_cc,
            num_iterations: iter,
        })
    }
}

#[cfg(test)]
mod tests;
