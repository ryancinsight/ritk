//! Thirion Demons deformable image registration.
//!
//! # Mathematical Specification
//!
//! Given fixed image F and moving image M, the Demons algorithm computes a
//! displacement field D : ℤ³ → ℝ³ that warps M toward F.
//!
//! **Optical-flow force at voxel p** (Thirion 1998):
//!
//!   f(p) = (F(p) − M_w(p)) · ∇F(p) / (|∇F(p)|² + (F(p)−M_w(p))²/σₓ² + ε)
//!
//! where:
//! - M_w(p) = M(p + D(p))  — current warp of M
//! - ∇F(p)               — gradient of the fixed image
//! - σₓ                  — max_step_length parameter (intensity normalisation)
//! - ε = 1e-5             — numerical floor
//!
//! **Per-iteration update:**
//! 1. Warp M with current D → M_w
//! 2. Compute forces f from (F, M_w, ∇F)
//! 3. Clamp |f| ≤ max_step_length
//! 4. Optional fluid regularisation: smooth f with G_{σ_fluid}
//! 5. Accumulate: D ← D + f
//! 6. Diffusive regularisation: D ← G_{σ_diff} ∗ D
//! 7. Compute MSE = mean((F − warp(M, D))²) for convergence tracking
//!
//! # References
//! - Thirion, J.-P. (1998). Image matching as a diffusion process: an analogy
//!   with Maxwell's demons. *Medical Image Analysis* 2(3):243–260.

use crate::deformable_field_ops::{compute_gradient, gaussian_smooth_inplace, warp_image};
use crate::error::RegistrationError;

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for Demons-family registration algorithms.
#[derive(Debug, Clone)]
pub struct DemonsConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Standard deviation (in voxels) of the Gaussian applied to the total
    /// displacement field after each iteration (diffusive regularisation).
    /// Set to 0.0 to disable.
    pub sigma_diffusion: f64,
    /// Standard deviation (in voxels) of the Gaussian applied to the *force
    /// update* before adding it to the displacement field (fluid regularisation).
    /// Set to 0.0 to disable.
    pub sigma_fluid: f64,
    /// Maximum per-voxel step length in voxel units.  Forces whose magnitude
    /// exceeds this value are rescaled to exactly `max_step_length`.
    pub max_step_length: f32,
}

impl Default for DemonsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            sigma_diffusion: 1.5,
            sigma_fluid: 0.0,
            max_step_length: 2.0,
        }
    }
}

/// Result returned by a Demons-family registration.
#[derive(Debug, Clone)]
pub struct DemonsResult {
    /// Warped moving image (same shape as fixed image).
    pub warped: Vec<f32>,
    /// Z-component of the final displacement field (voxel units).
    pub disp_z: Vec<f32>,
    /// Y-component of the final displacement field.
    pub disp_y: Vec<f32>,
    /// X-component of the final displacement field.
    pub disp_x: Vec<f32>,
    /// Mean-squared error between fixed and warped moving at the final iteration.
    pub final_mse: f64,
    /// Actual number of iterations performed (may be less than `max_iterations`
    /// if convergence was reached).
    pub num_iterations: usize,
}

/// Thirion Demons registration (classic variant).
///
/// Computes a dense displacement field by iterating optical-flow-like forces
/// derived from image intensity difference and the fixed-image gradient.
#[derive(Debug, Clone)]
pub struct ThirionDemonsRegistration {
    /// Algorithm configuration.
    pub config: DemonsConfig,
}

impl ThirionDemonsRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: DemonsConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` and return the displacement field and
    /// warped moving image.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `Vec<f32>` in Z-major order.
    /// - `moving`  — moving image, same shape as `fixed`.
    /// - `dims`    — image dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]` (used for gradient).
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if `fixed` and `moving` have different
    /// lengths or `dims` are inconsistent.
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
        let (grad_z, grad_y, grad_x) = compute_gradient(fixed, dims, spacing);

        let sigma_x = self.config.max_step_length;
        let sigma_x2 = sigma_x * sigma_x;

        let mut final_mse = compute_mse(fixed, moving, dims, &disp_z, &disp_y, &disp_x);
        let mut iter = 0usize;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // 1. Warp moving with current displacement.
            let m_warped = warp_image(moving, dims, &disp_z, &disp_y, &disp_x);

            // 2. Compute optical-flow forces.
            let mut fz = vec![0.0_f32; n];
            let mut fy = vec![0.0_f32; n];
            let mut fx = vec![0.0_f32; n];

            for i in 0..n {
                let diff = fixed[i] - m_warped[i];
                let gz = grad_z[i];
                let gy = grad_y[i];
                let gx = grad_x[i];
                let grad_sq = gz * gz + gy * gy + gx * gx;
                let denom = grad_sq + diff * diff / sigma_x2 + 1e-5;
                let scale = diff / denom;
                fz[i] = scale * gz;
                fy[i] = scale * gy;
                fx[i] = scale * gx;
            }

            // 3. Clamp force magnitude.
            clamp_field_magnitude(&mut fz, &mut fy, &mut fx, self.config.max_step_length);

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

            // 7. Compute current MSE.
            final_mse = compute_mse(fixed, moving, dims, &disp_z, &disp_y, &disp_x);
        }

        let warped = warp_image(moving, dims, &disp_z, &disp_y, &disp_x);

        Ok(DemonsResult {
            warped,
            disp_z,
            disp_y,
            disp_x,
            final_mse,
            num_iterations: iter,
        })
    }
}

// ── Shared helpers (also used by diffeomorphic.rs and symmetric.rs) ───────────

/// Compute MSE = mean((F(p) − M_w(p))²) where M_w = warp(M, D).
pub(super) fn compute_mse(
    fixed: &[f32],
    moving: &[f32],
    dims: [usize; 3],
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
) -> f64 {
    let warped = warp_image(moving, dims, dz, dy, dx);
    fixed
        .iter()
        .zip(warped.iter())
        .map(|(&f, &m)| ((f - m) as f64).powi(2))
        .sum::<f64>()
        / fixed.len() as f64
}

/// Compute optical-flow Thirion forces given pre-computed fixed-image gradient.
///
/// Returns `(fz, fy, fx)` force components.  Forces whose magnitude exceeds
/// `max_step_length` are rescaled.
pub(super) fn thirion_forces(
    fixed: &[f32],
    m_warped: &[f32],
    grad_z: &[f32],
    grad_y: &[f32],
    grad_x: &[f32],
    max_step_length: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = fixed.len();
    let sigma_x2 = max_step_length * max_step_length;

    let mut fz = vec![0.0_f32; n];
    let mut fy = vec![0.0_f32; n];
    let mut fx = vec![0.0_f32; n];

    for i in 0..n {
        let diff = fixed[i] - m_warped[i];
        let gz = grad_z[i];
        let gy = grad_y[i];
        let gx = grad_x[i];
        let grad_sq = gz * gz + gy * gy + gx * gx;
        let denom = grad_sq + diff * diff / sigma_x2 + 1e-5;
        let scale = diff / denom;
        fz[i] = scale * gz;
        fy[i] = scale * gy;
        fx[i] = scale * gx;
    }

    clamp_field_magnitude(&mut fz, &mut fy, &mut fx, max_step_length);
    (fz, fy, fx)
}

/// Clamp per-voxel displacement/force magnitude to `max_length`.
fn clamp_field_magnitude(fz: &mut [f32], fy: &mut [f32], fx: &mut [f32], max_length: f32) {
    let max2 = max_length * max_length;
    for i in 0..fz.len() {
        let mag2 = fz[i] * fz[i] + fy[i] * fy[i] + fx[i] * fx[i];
        if mag2 > max2 {
            let scale = max_length / mag2.sqrt();
            fz[i] *= scale;
            fy[i] *= scale;
            fx[i] *= scale;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a smooth test image: I[z,y,x] = sin(z/nz·π)·cos(y/ny·π)·x
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

    /// Shift an image +shift voxels in x with zero-padding at the left boundary.
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

    /// Registering identical images must produce near-zero MSE.
    #[test]
    fn identity_registration_near_zero_mse() {
        let dims = [8usize, 8, 8];
        let image = make_test_image(dims);
        let reg = ThirionDemonsRegistration::new(DemonsConfig {
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

    /// MSE must be lower after registration than before.
    #[test]
    fn registration_reduces_mse() {
        let dims = [10usize, 10, 14];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let fixed = make_test_image(dims);
        // Shift moving by +2 voxels in x.
        let moving = translate_x(&fixed, dims, 2);

        let initial_mse: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&f, &m)| ((f - m) as f64).powi(2))
            .sum::<f64>()
            / n as f64;

        let reg = ThirionDemonsRegistration::new(DemonsConfig {
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
            "MSE should decrease by at least 50%: initial={initial_mse:.6} final={:.6}",
            result.final_mse
        );
    }

    /// Translation recovery: moving = translate(fixed, +2 voxels in x).
    ///
    /// Under the forward-warp convention `warped(p) = moving(p + D(p))`,
    /// aligning `moving[ix] = fixed[ix−2]` requires `D_x = +2` so that
    /// `warped(ix) = moving(ix + 2) = fixed(ix + 2 − 2) = fixed(ix)`.
    /// Therefore the mean interior disp_x must be **positive**.
    #[test]
    fn translation_recovery_direction() {
        let dims = [8usize, 8, 12];
        let [nz, ny, nx] = dims;
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);

        let reg = ThirionDemonsRegistration::new(DemonsConfig {
            max_iterations: 50,
            ..Default::default()
        });
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        // Compute mean disp_x in interior (exclude 2-voxel boundary).
        let mut sum_dx = 0.0_f64;
        let mut count = 0usize;
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 2..nx - 2 {
                    let fi = iz * ny * nx + iy * nx + ix;
                    sum_dx += result.disp_x[fi] as f64;
                    count += 1;
                }
            }
        }
        let mean_dx = sum_dx / count as f64;

        // Forward-warp convention: warped(ix) = moving(ix + D_x).
        // moving[ix] = fixed[ix-2], so we need D_x = +2 to sample from ix+2.
        // The mean interior disp_x must therefore be positive (≈ +2).
        assert!(
            mean_dx > 0.0,
            "mean interior disp_x should be positive (≈+2) for forward-warp convention, got {mean_dx:.4}"
        );
    }

    /// Registering a constant image against itself produces zero displacement.
    #[test]
    fn constant_image_zero_forces() {
        let dims = [6usize, 6, 6];
        let n = 6 * 6 * 6;
        let image = vec![42.0_f32; n];
        let reg = ThirionDemonsRegistration::new(DemonsConfig {
            max_iterations: 10,
            ..Default::default()
        });
        let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();

        let max_disp = result
            .disp_x
            .iter()
            .chain(result.disp_y.iter())
            .chain(result.disp_z.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);

        // With sigma_diffusion > 0 and zero forces, displacement stays near 0.
        assert!(
            max_disp < 1e-4,
            "constant image: max displacement should be ~0, got {max_disp}"
        );
    }

    /// Error is returned when fixed and moving have different lengths.
    #[test]
    fn mismatched_lengths_returns_error() {
        let dims = [4usize, 4, 4];
        let fixed = vec![0.0_f32; 4 * 4 * 4];
        let moving = vec![0.0_f32; 4 * 4 * 5]; // wrong length
        let reg = ThirionDemonsRegistration::new(DemonsConfig::default());
        assert!(
            reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
                .is_err(),
            "should return error for mismatched lengths"
        );
    }

    /// Mean displacement magnitude is finite after registration.
    #[test]
    fn displacement_field_finite() {
        let dims = [6usize, 6, 8];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 1);
        let reg = ThirionDemonsRegistration::new(DemonsConfig {
            max_iterations: 20,
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
            assert!(dz.is_finite(), "disp_z contains non-finite value: {dz}");
            assert!(dy.is_finite(), "disp_y contains non-finite value: {dy}");
            assert!(dx.is_finite(), "disp_x contains non-finite value: {dx}");
        }
    }
}
