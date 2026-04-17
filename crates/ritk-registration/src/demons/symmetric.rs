//! Symmetric Demons deformable image registration.
//!
//! # Mathematical Specification
//!
//! The Symmetric Demons variant (Pennec et al. 1999, extended) uses gradient
//! information from **both** the fixed and warped moving images to compute
//! registration forces.  This makes the algorithm approximately symmetric with
//! respect to swapping fixed and moving images.
//!
//! **Symmetric force at voxel p:**
//!
//!   f(p) = (F(p) − M_w(p)) · [∇F(p) + ∇M_w(p)] /
//!           (|∇F(p) + ∇M_w(p)|² / 4 + (F(p) − M_w(p))² / σₓ² + ε)
//!
//! where:
//! - M_w(p) = M(p + D(p))  — current warp of M
//! - ∇F(p)                — gradient of the fixed image (constant)
//! - ∇M_w(p)              — gradient of the warped moving image (recomputed each iteration)
//! - σₓ                   — max_step_length parameter
//! - ε = 1e-5             — numerical floor
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
//! 8. Compute MSE = mean((F − warp(M, D))²)
//!
//! # Symmetry Property
//! When fixed and moving are swapped, the force direction reverses.  More
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

use super::thirion::{DemonsConfig, DemonsResult};
use crate::deformable_field_ops::{
    compute_gradient, compute_gradient_into, compute_mse_streaming, gaussian_smooth_inplace,
    warp_image, warp_image_into,
};
use crate::error::RegistrationError;

// ── Public types ──────────────────────────────────────────────────────────────

/// Symmetric Demons registration.
///
/// Extends the classic Thirion Demons by incorporating gradient information
/// from both the fixed and the warped moving images.  The resulting forces
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
        let (grad_fz, grad_fy, grad_fx) = compute_gradient(fixed, dims, spacing);

        let sigma_x = self.config.max_step_length;
        let sigma_x2 = sigma_x * sigma_x;

        let mut m_warped = vec![0.0_f32; n];
        let mut grad_mz = vec![0.0_f32; n];
        let mut grad_my = vec![0.0_f32; n];
        let mut grad_mx = vec![0.0_f32; n];
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];

        let mut final_mse = compute_mse_streaming(fixed, moving, dims, &disp_z, &disp_y, &disp_x);
        let mut iter = 0usize;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // 1. Warp moving with current displacement into a reusable buffer.
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

            // 3. Compute symmetric forces into reusable buffers.
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
                gaussian_smooth_inplace(&mut fz, dims, self.config.sigma_fluid);
                gaussian_smooth_inplace(&mut fy, dims, self.config.sigma_fluid);
                gaussian_smooth_inplace(&mut fx, dims, self.config.sigma_fluid);
            }

            // 5. Accumulate displacement field.
            for i in 0..n {
                disp_z[i] += fz[i];
                disp_y[i] += fy[i];
                disp_x[i] += fx[i];
            }

            // 6. Diffusive regularisation (smooth total field).
            if self.config.sigma_diffusion > 0.0 {
                gaussian_smooth_inplace(&mut disp_z, dims, self.config.sigma_diffusion);
                gaussian_smooth_inplace(&mut disp_y, dims, self.config.sigma_diffusion);
                gaussian_smooth_inplace(&mut disp_x, dims, self.config.sigma_diffusion);
            }

            // 7. Compute current MSE without materialising another warped image.
            final_mse = compute_mse_streaming(fixed, moving, dims, &disp_z, &disp_y, &disp_x);
        }

        let warped = warp_image(moving, dims, &disp_z, &disp_y, &disp_x);

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

/// Compute symmetric Demons forces.
///
/// Force formula:
///   f(p) = diff · (gF + gM_w) / (|gF + gM_w|² / 4 + diff² / σₓ² + ε)
///
/// where `diff = F(p) − M_w(p)`, `gF = ∇F(p)`, `gM_w = ∇M_w(p)`.
///
/// The factor of 4 in the denominator arises from using the combined gradient
/// (sum rather than average): the effective per-image gradient contribution is
/// (gF + gM_w) / 2, so |avg_grad|² = |gF + gM_w|² / 4.
/// Compute symmetric Demons forces into caller-provided buffers.
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

        // Combined gradient (fixed + warped-moving).
        let cgz = grad_fz[i] + grad_mz[i];
        let cgy = grad_fy[i] + grad_my[i];
        let cgx = grad_fx[i] + grad_mx[i];

        // Denominator: |combined_grad|² / 4 + diff² / σₓ² + ε
        let grad_sq_over4 = (cgz * cgz + cgy * cgy + cgx * cgx) * 0.25;
        let denom = grad_sq_over4 + diff * diff / sigma_x2 + 1e-5;

        let scale = diff / denom;
        let mut fzi = scale * cgz;
        let mut fyi = scale * cgy;
        let mut fxi = scale * cgx;

        // Clamp per-voxel magnitude.
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




// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Registering identical images must yield near-zero MSE.
    #[test]
    fn identity_registration_near_zero_mse() {
        let dims = [8usize, 8, 8];
        let image = make_test_image(dims);
        let reg = SymmetricDemonsRegistration::new(DemonsConfig {
            max_iterations: 20,
            ..Default::default()
        });
        let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.final_mse < 1e-3,
            "identity MSE should be < 1e-3, got {}",
            result.final_mse
        );
    }

    /// MSE must decrease after registration of translated images.
    #[test]
    fn registration_reduces_mse() {
        let dims = [10usize, 10, 14];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);

        let initial_mse: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&f, &m)| ((f - m) as f64).powi(2))
            .sum::<f64>()
            / n as f64;

        let reg = SymmetricDemonsRegistration::new(DemonsConfig {
            max_iterations: 50,
            ..Default::default()
        });
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        assert!(
            result.final_mse < initial_mse,
            "MSE should decrease: initial={initial_mse:.6} final={:.6}",
            result.final_mse
        );
        assert!(
            result.final_mse < initial_mse * 0.5,
            "MSE should decrease by ≥50%: initial={initial_mse:.6} final={:.6}",
            result.final_mse
        );
    }

    /// Approximate symmetry: register(F, M) and register(M, F) produce
    /// displacements that are approximately negatives of each other.
    ///
    /// # Invariant verified
    /// For the x-displacement in the interior region:
    ///   |mean(disp_x_FM) + mean(disp_x_MF)| < |mean(disp_x_FM)| × 0.5
    ///
    /// This is a loose symmetry check appropriate for discrete finite-difference
    /// implementations with boundary effects.
    #[test]
    fn approximate_symmetry_fm_vs_mf() {
        let dims = [8usize, 8, 12];
        let [nz, ny, nx] = dims;
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);

        let reg = SymmetricDemonsRegistration::new(DemonsConfig {
            max_iterations: 30,
            ..Default::default()
        });

        let res_fm = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();
        let res_mf = reg
            .register(&moving, &fixed, dims, [1.0, 1.0, 1.0])
            .unwrap();

        // Compute mean interior disp_x for FM and MF.
        let mut sum_fm = 0.0_f64;
        let mut sum_mf = 0.0_f64;
        let mut count = 0usize;
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 2..nx - 2 {
                    let fi = iz * ny * nx + iy * nx + ix;
                    sum_fm += res_fm.disp_x[fi] as f64;
                    sum_mf += res_mf.disp_x[fi] as f64;
                    count += 1;
                }
            }
        }
        let mean_fm = sum_fm / count as f64;
        let mean_mf = sum_mf / count as f64;

        // FM and MF displacements should have opposite signs.
        assert!(
            mean_fm * mean_mf < 0.0,
            "FM and MF mean disp_x should have opposite signs: \
             mean_fm={mean_fm:.4} mean_mf={mean_mf:.4}"
        );

        // Magnitude of (FM + MF) should be less than 50% of FM magnitude.
        let asymmetry = (mean_fm + mean_mf).abs();
        let scale = mean_fm.abs().max(1e-6);
        assert!(
            asymmetry < scale * 0.5,
            "Asymmetry too large: |FM+MF|={asymmetry:.4} vs |FM|={:.4}",
            mean_fm.abs()
        );
    }

    /// All displacement field components must be finite.
    #[test]
    fn displacement_field_finite() {
        let dims = [6usize, 6, 8];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 1);
        let reg = SymmetricDemonsRegistration::new(DemonsConfig {
            max_iterations: 15,
            ..Default::default()
        });
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();
        for (&dz, (&dy, &dx)) in result
            .disp_z
            .iter()
            .zip(result.disp_y.iter().zip(result.disp_x.iter()))
        {
            assert!(dz.is_finite(), "disp_z non-finite: {dz}");
            assert!(dy.is_finite(), "disp_y non-finite: {dy}");
            assert!(dx.is_finite(), "disp_x non-finite: {dx}");
        }
    }

    /// Error is returned for mismatched image lengths.
    #[test]
    fn mismatched_lengths_returns_error() {
        let dims = [4usize, 4, 4];
        let fixed = vec![0.0_f32; 4 * 4 * 4];
        let moving = vec![0.0_f32; 4 * 4 * 5];
        let reg = SymmetricDemonsRegistration::new(DemonsConfig::default());
        assert!(
            reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
                .is_err(),
            "should return error for mismatched lengths"
        );
    }
}
