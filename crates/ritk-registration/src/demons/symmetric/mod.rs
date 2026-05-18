//! Symmetric Demons deformable image registration.
//!
//! # Mathematical Specification
//!
//! The Symmetric Demons variant (Pennec et al. 1999, extended) uses gradient
//! information from **both** the fixed and warped moving images to compute
//! registration forces. This makes the algorithm approximately symmetric with
//! respect to swapping fixed and moving images.
//!
//! **Symmetric force at voxel p:**
//!
//! f(p) = (F(p) − M_w(p)) · [∇F(p) + ∇M_w(p)] /
//! (|∇F(p) + ∇M_w(p)|² / 4 + (F(p) − M_w(p))² / σₓ² + ε)
//!
//! where:
//! - M_w(p) = M(p + D(p)) — current warp of M
//! - ∇F(p) — gradient of the fixed image (constant)
//! - ∇M_w(p) — gradient of the warped moving image (recomputed each iteration)
//! - σₓ — max_step_length parameter
//! - ε = 1e-5 — numerical floor
//!
//! The |∇F + ∇M_w|² / 4 denominator term (dividing by 4 instead of 1) comes
//! from the symmetric formulation where the combined gradient is the average of
//! the two individual gradients, so the effective gradient magnitude is halved.
//!
//! **Per-iteration update:**
//! 1. Warp M with current D → M_w
//! 2. Compute ∇F (fixed, cached) and ∇M_w (recomputed each iteration)
//! 3. Compute symmetric forces f
//! 4. Clamp |f| ≤ max_step_length
//! 5. Optional fluid regularisation: smooth f with G_{σ_fluid}
//! 6. Accumulate: D ← D + f
//! 7. Diffusive regularisation: D ← G_{σ_diff} ∗ D
//! 8. Compute MSE = mean((F − M_w)²) (reuses M_w from step 1)
//!
//! # Memory discipline
//! All scratch buffers are pre-allocated before the iteration loop.
//! The loop body performs **zero heap allocations**; `_into` variants of
//! image warp, gradient, force, and Gaussian smoothing write into
//! caller-provided buffers. Total pre-allocation: ~14n f32
//! (3 displacement + 1 warped + 3 fixed gradient + 3 moving gradient +
//! 3 forces + 1 smooth scratch = 14n).
//!
//! # Symmetry Property
//! When fixed and moving are swapped, the force direction reverses. More
//! precisely: for images F and M with displacement D_{FM}, and images M and F
//! with displacement D_{MF}, we expect D_{FM} ≈ −D_{MF} for small deformations.
//!
//! # References
//! - Pennec, X., Cachier, P. & Ayache, N. (1999). Understanding the
//!   "Demon's Algorithm": 3D Non-Rigid Registration by Gradient Descent.
//!   *MICCAI*, LNCS 1679:597–605.
//! - Cachier, P., Bardinet, E., Dormont, D., Pennec, X. & Ayache, N. (2003).
//!   Iconic feature based nonrigid registration: the PASHA algorithm.
//!   *CVIU* 89(2–3):272–298.

use super::config::{DemonsConfig, DemonsResult};
use crate::deformable_field_ops::{
    compute_gradient_into, gaussian_smooth_with_scratch, warp_image_into,
};
use crate::error::RegistrationError;

// ── Public types ──────────────────────────────────────────────────────────────

/// Symmetric Demons registration.
///
/// Extends the classic Thirion Demons by incorporating gradient information
/// from both the fixed and the warped moving images. The resulting forces
/// are approximately symmetric: swapping fixed and moving produces forces of
/// opposite sign.
#[derive(Debug, Clone)]
pub struct SymmetricDemonsRegistration {
    /// Algorithm configuration (shared with Thirion Demons).
    pub config: DemonsConfig,
}

impl SymmetricDemonsRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: DemonsConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` using symmetric optical-flow forces.
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
    ) -> Result<DemonsResult, RegistrationError> {
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

        // Initialise displacement field to zero.
        let mut disp_z = vec![0.0_f32; n];
        let mut disp_y = vec![0.0_f32; n];
        let mut disp_x = vec![0.0_f32; n];

        // Pre-compute fixed-image gradient (constant across iterations).
        let mut grad_fz = vec![0.0_f32; n];
        let mut grad_fy = vec![0.0_f32; n];
        let mut grad_fx = vec![0.0_f32; n];
        compute_gradient_into(
            fixed,
            dims,
            spacing,
            &mut grad_fz,
            &mut grad_fy,
            &mut grad_fx,
        );

        let sigma_x = self.config.max_step_length;
        let sigma_x2 = sigma_x * sigma_x;

        // ── Pre-allocated scratch buffers (zero alloc inside the loop) ──
        let mut m_warped = vec![0.0_f32; n];
        let mut grad_mz = vec![0.0_f32; n];
        let mut grad_my = vec![0.0_f32; n];
        let mut grad_mx = vec![0.0_f32; n];
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];
        let mut smooth_tmp = vec![0.0_f32; n];

        // Initial MSE: D = 0 (identity) — displacement buffers already zero.
        let mut final_mse: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&f, &m)| ((f - m) as f64).powi(2))
            .sum::<f64>()
            / n as f64;

        let mut iter = 0usize;
        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // 1. Warp moving with current displacement.
            warp_image_into(moving, dims, &disp_z, &disp_y, &disp_x, &mut m_warped);

            // 2. Compute gradient of the warped moving image (changes each iteration).
            compute_gradient_into(
                &m_warped,
                dims,
                spacing,
                &mut grad_mz,
                &mut grad_my,
                &mut grad_mx,
            );

            // 3. Compute symmetric forces into pre-allocated buffers.
            symmetric_forces_into(
                fixed,
                &m_warped,
                &grad_fz,
                &grad_fy,
                &grad_fx,
                &grad_mz,
                &grad_my,
                &grad_mx,
                sigma_x2,
                self.config.max_step_length,
                &mut fz,
                &mut fy,
                &mut fx,
            );

            // 4. Optional fluid regularisation (smooth forces before accumulation).
            if self.config.sigma_fluid > 0.0 {
                gaussian_smooth_with_scratch(
                    &mut fz,
                    dims,
                    self.config.sigma_fluid,
                    &mut smooth_tmp,
                );
                gaussian_smooth_with_scratch(
                    &mut fy,
                    dims,
                    self.config.sigma_fluid,
                    &mut smooth_tmp,
                );
                gaussian_smooth_with_scratch(
                    &mut fx,
                    dims,
                    self.config.sigma_fluid,
                    &mut smooth_tmp,
                );
            }

            // 5. Accumulate displacement field.
            for i in 0..n {
                disp_z[i] += fz[i];
                disp_y[i] += fy[i];
                disp_x[i] += fx[i];
            }

            // 6. Diffusive regularisation (smooth total field).
            if self.config.sigma_diffusion > 0.0 {
                gaussian_smooth_with_scratch(
                    &mut disp_z,
                    dims,
                    self.config.sigma_diffusion,
                    &mut smooth_tmp,
                );
                gaussian_smooth_with_scratch(
                    &mut disp_y,
                    dims,
                    self.config.sigma_diffusion,
                    &mut smooth_tmp,
                );
                gaussian_smooth_with_scratch(
                    &mut disp_x,
                    dims,
                    self.config.sigma_diffusion,
                    &mut smooth_tmp,
                );
            }

            // 7. Compute MSE from m_warped (already computed at step 1).
            final_mse = fixed
                .iter()
                .zip(m_warped.iter())
                .map(|(&f, &m)| ((f - m) as f64).powi(2))
                .sum::<f64>()
                / n as f64;
        }

        let mut warped = vec![0.0_f32; n];
        warp_image_into(moving, dims, &disp_z, &disp_y, &disp_x, &mut warped);

        Ok(DemonsResult {
            warped,
            disp_z,
            disp_y,
            disp_x,
            vel_z: None,
            vel_y: None,
            vel_x: None,
            final_mse,
            num_iterations: iter,
        })
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Write symmetric Demons forces into caller-provided buffers.
///
/// Force formula:
/// f(p) = diff · (gF + gM_w) / (|gF + gM_w|² / 4 + diff² / σₓ² + ε)
///
/// where `diff = F(p) − M_w(p)`, `gF = ∇F(p)`, `gM_w = ∇M_w(p)`.
///
/// Performs zero heap allocation. All output buffers must have length `fixed.len()`.
#[allow(clippy::too_many_arguments)]
fn symmetric_forces_into(
    fixed: &[f32],
    m_warped: &[f32],
    grad_fz: &[f32],
    grad_fy: &[f32],
    grad_fx: &[f32],
    grad_mz: &[f32],
    grad_my: &[f32],
    grad_mx: &[f32],
    sigma_x2: f32,
    max_step_length: f32,
    fz: &mut [f32],
    fy: &mut [f32],
    fx: &mut [f32],
) {
    let n = fixed.len();
    let max2 = max_step_length * max_step_length;

    for i in 0..n {
        let diff = fixed[i] - m_warped[i];
        let cgz = grad_fz[i] + grad_mz[i];
        let cgy = grad_fy[i] + grad_my[i];
        let cgx = grad_fx[i] + grad_mx[i];
        let grad_sq_over4 = (cgz * cgz + cgy * cgy + cgx * cgx) * 0.25;
        let denom = grad_sq_over4 + diff * diff / sigma_x2 + 1e-5;
        let scale = diff / denom;

        let mut fzi = scale * cgz;
        let mut fyi = scale * cgy;
        let mut fxi = scale * cgx;

        let mag2 = fzi * fzi + fyi * fyi + fxi * fxi;
        if mag2 > max2 {
            let s = max_step_length / mag2.sqrt();
            fzi *= s;
            fyi *= s;
            fxi *= s;
        }

        fz[i] = fzi;
        fy[i] = fyi;
        fx[i] = fxi;
    }
}

#[cfg(test)]
mod tests;
